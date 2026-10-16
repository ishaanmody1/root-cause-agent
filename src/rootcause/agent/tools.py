"""Agent tools over a scenario's sensor data, as seen at diagnosis time.

``ToolContext`` holds the data visible to the agent (truncated at ``t_now``), the detector
output that triggered it, the operator event log, and the topology. Every tool returns a
compact JSON-serialisable dict. Bad inputs raise ``ToolError``, which the loop turns into an
``is_error`` tool result so the model can recover.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from rootcause.agent.schema import Diagnosis, strict_schema
from rootcause.detect.detector import Detection, DetectorConfig, ewma, robust_baseline
from rootcause.sim.faults import Event
from rootcause.sim.simulator import Scenario
from rootcause.sim.topology import LineParams, Topology, get_topology, sensor_id


class ToolError(Exception):
    pass


def _r(x: float | None, nd: int = 2) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


class ToolContext:
    def __init__(
        self,
        data: pd.DataFrame,
        detection: Detection,
        events: list[Event],
        t_now: int,
        params: LineParams | None = None,
        topology: Topology | None = None,
        baseline_ticks: int | None = None,
    ) -> None:
        self.t_now = int(t_now)
        self.full = data
        self.data = data[data["t"] <= self.t_now].reset_index(drop=True)
        self.detection = detection
        self.events = [e for e in events if e.tick <= self.t_now]
        self.params = params or LineParams()
        self.topology = topology or get_topology(self.params)
        self.baseline_ticks = baseline_ticks or detection.config.baseline_ticks
        self.call_count = 0
        self.tool_specs = build_tool_specs(self.topology)

    @classmethod
    def from_scenario(
        cls, sc: Scenario, detection: Detection, grace_ticks: int = 40
    ) -> ToolContext:
        last = int(sc.data["t"].iloc[-1])
        if detection.triggered and detection.alarm_tick is not None:
            t_now = min(detection.alarm_tick + grace_ticks, last)
        else:
            t_now = last
        return cls(
            data=sc.data,
            detection=detection,
            events=sc.spec.events,
            t_now=t_now,
            params=sc.params,
            topology=get_topology(sc.params),
            baseline_ticks=sc.spec.baseline_ticks,
        )

    # ------------------------------------------------------------------ helpers
    def _series(self, stage: str, sensor: str) -> tuple[str, np.ndarray]:
        stages = self.topology.stages
        if stage not in stages:
            raise ToolError(f"unknown stage '{stage}'. Stages: {stages}")
        sensors = self.topology.sensors[stage]
        if sensor not in sensors:
            raise ToolError(f"stage '{stage}' has no sensor '{sensor}'. It has: {sensors}")
        sid = sensor_id(stage, sensor)
        if sid not in self.data.columns:
            raise ToolError(f"no data column for {sid}")
        return sid, self.data[sid].to_numpy(dtype=float)

    def _baseline_obj(self, sid: str):
        b = self.detection.baselines.get(sid)
        if b is None or b.n == 0:
            b = robust_baseline(self.data[sid].to_numpy(dtype=float)[: self.baseline_ticks])
        return b

    def _baseline(self, sid: str) -> tuple[float, float]:
        b = self._baseline_obj(sid)
        return b.mean, max(b.std, abs(b.mean) * 1e-3, 1e-9)

    def _varies_when_healthy(self, sid: str) -> bool:
        return not self._baseline_obj(sid).quantised

    def _zero_is_abnormal(self, sid: str) -> bool:
        b = self._baseline_obj(sid)
        return (not b.had_zeros) and abs(b.mean) > 5 * max(b.std, 1e-9)

    def _clamp(self, t_start: int | None, t_end: int | None) -> tuple[int, int]:
        a = 0 if t_start is None else max(0, int(t_start))
        b = self.t_now if t_end is None else min(self.t_now, int(t_end))
        if a > b:
            raise ToolError(f"t_start ({a}) must be <= t_end ({b}); data ends at t={self.t_now}")
        return a, b

    def detection_summary(self) -> str:
        d = self.detection
        if not d.triggered or d.first_alarm is None:
            return "No alarm (detector did not trigger)."
        confirmed = [a for a in d.alarms if a.alarm_tick <= self.t_now]
        lines = [f"First alarm (what paged the operator): {d.first_alarm.describe()}"]
        if len(confirmed) > 1:
            lines.append(f"All alarms confirmed by t={self.t_now}, ordered by when they began:")
            lines += [f"  - {a.describe()}" for a in confirmed]
        else:
            lines.append("No other sensor has alarmed yet.")
        return "\n".join(lines)

    # -------------------------------------------------------------------- tools
    def get_line_topology(self) -> dict[str, Any]:
        return self.topology.model_dump()

    def query_sensor(
        self,
        stage: str,
        sensor: str,
        t_start: int | None = None,
        t_end: int | None = None,
        max_points: int | None = 40,
    ) -> dict[str, Any]:
        sid, x = self._series(stage, sensor)
        a, b = self._clamp(t_start, t_end)
        seg = x[a : b + 1]
        ticks = np.arange(a, b + 1)
        n_pts = max(4, min(int(max_points or 40), 200))
        if len(seg) > n_pts:
            edges = np.linspace(0, len(seg), n_pts + 1).astype(int)
            pts = []
            for i in range(n_pts):
                chunk = seg[edges[i] : edges[i + 1]]
                tchunk = ticks[edges[i] : edges[i + 1]]
                if len(chunk) == 0:
                    continue
                valid = chunk[~np.isnan(chunk)]
                pts.append(
                    {
                        "t": int(tchunk[0]),
                        "value": _r(float(valid.mean())) if len(valid) else None,
                    }
                )
            resolution = f"bucket means over ~{len(seg) / n_pts:.1f} ticks"
        else:
            pts = [{"t": int(t), "value": _r(v)} for t, v in zip(ticks, seg, strict=True)]
            resolution = "raw"
        mu, sd = self._baseline(sid)
        valid = seg[~np.isnan(seg)]
        return {
            "sensor": sid,
            "unit": self.topology.units.get(sensor, ""),
            "window": [a, b],
            "resolution": resolution,
            "baseline_mean": _r(mu),
            "baseline_std": _r(sd),
            "stats": {
                "mean": _r(valid.mean()) if len(valid) else None,
                "min": _r(valid.min()) if len(valid) else None,
                "max": _r(valid.max()) if len(valid) else None,
                "nan_count": int(np.isnan(seg).sum()),
            },
            "points": pts,
        }

    def compare_to_baseline(
        self, stage: str, sensor: str, window_ticks: int | None = 30
    ) -> dict[str, Any]:
        sid, x = self._series(stage, sensor)
        w = max(5, min(int(window_ticks or 30), 200))
        a = max(self.baseline_ticks, self.t_now - w + 1)
        seg = x[a : self.t_now + 1]
        valid = seg[~np.isnan(seg)]
        mu, sd = self._baseline(sid)
        nan_frac = float(np.isnan(seg).mean()) if len(seg) else 1.0
        if len(valid) < 3:
            return {
                "sensor": sid,
                "window": [a, self.t_now],
                "baseline_mean": _r(mu),
                "baseline_std": _r(sd),
                "nan_fraction": _r(nan_frac),
                "verdict": "missing",
            }
        recent_mean = float(valid.mean())
        shift_sigma = (recent_mean - mu) / sd
        pct = (recent_mean - mu) / mu * 100 if abs(mu) > 1e-9 else None
        tt = np.arange(a, self.t_now + 1)[~np.isnan(seg)]
        slope = float(np.polyfit(tt, valid, 1)[0]) * 10 if len(valid) >= 5 else 0.0
        flat = len(valid) >= 10 and float(np.std(valid)) == 0.0 and self._varies_when_healthy(sid)
        if flat:
            verdict = "flatline"
        elif np.all(valid == 0.0) and self._zero_is_abnormal(sid):
            verdict = "stuck_zero"
        elif shift_sigma > 3:
            verdict = "elevated"
        elif shift_sigma < -3:
            verdict = "depressed"
        elif abs(shift_sigma) > 1.5:
            verdict = "slightly_" + ("elevated" if shift_sigma > 0 else "depressed")
        else:
            verdict = "normal"
        return {
            "sensor": sid,
            "unit": self.topology.units.get(sensor, ""),
            "window": [a, self.t_now],
            "baseline_mean": _r(mu),
            "baseline_std": _r(sd),
            "recent_mean": _r(recent_mean),
            "recent_min": _r(valid.min()),
            "recent_max": _r(valid.max()),
            "shift_sigma": _r(shift_sigma, 1),
            "pct_change": _r(pct, 1),
            "slope_per_10_ticks": _r(slope, 3),
            "nan_fraction": _r(nan_frac),
            "verdict": verdict,
        }

    def find_first_deviation(
        self,
        z_threshold: float | None = 2.5,
        persistence: int | None = 3,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> dict[str, Any]:
        thr = float(z_threshold or 2.5)
        if thr < 1.0:
            raise ToolError("z_threshold must be >= 1.0; lower values flag noise")
        k = max(1, int(persistence or 3))
        a, b = self._clamp(t_start, t_end)
        a = max(a, self.baseline_ticks)
        cfg = DetectorConfig()
        devs: list[dict[str, Any]] = []
        quiet: list[str] = []
        for stage in self.topology.stages:
            for sensor in self.topology.sensors[stage]:
                sid, x = self._series(stage, sensor)
                seg = x[a : b + 1]
                ticks = np.arange(a, b + 1)
                mu, sd = self._baseline(sid)
                entry: dict[str, Any] | None = None
                nan_run = _first_run_len(np.isnan(seg), cfg.nan_run_threshold)
                if nan_run is not None:
                    entry = {"direction": "missing", "first_tick": int(ticks[nan_run])}
                else:
                    zero_run = _first_run_len(seg == 0.0, k)
                    same = np.concatenate([[False], np.diff(seg) == 0.0])
                    flat_run = _first_run_len(same, cfg.flatline_window)
                    if zero_run is not None and self._zero_is_abnormal(sid):
                        entry = {"direction": "stuck_zero", "first_tick": int(ticks[zero_run])}
                    elif flat_run is not None and self._varies_when_healthy(sid):
                        entry = {
                            "direction": "flatline",
                            "first_tick": int(ticks[max(flat_run - 1, 0)]),
                        }
                    else:
                        z = ewma((seg - mu) / sd, cfg.ewma_alpha)
                        hi = _first_run_len(z > thr, k)
                        lo = _first_run_len(z < -thr, k)
                        cands = [(i, "high") for i in [hi] if i is not None] + [
                            (i, "low") for i in [lo] if i is not None
                        ]
                        if cands:
                            i, direction = min(cands)
                            entry = {
                                "direction": direction,
                                "first_tick": int(ticks[i]),
                                "confirmed_tick": int(ticks[min(i + k - 1, len(ticks) - 1)]),
                                "current_z": _r(float(z[-1]), 1),
                                "peak_abs_z": _r(float(np.nanmax(np.abs(z))), 1),
                            }
                if entry is None:
                    quiet.append(sid)
                else:
                    devs.append({"sensor": sid, "stage": stage, "measure": sensor, **entry})
        devs.sort(key=lambda d: d["first_tick"])
        max_list = 30
        out: dict[str, Any] = {
            "t_now": self.t_now,
            "window_searched": [a, b],
            "z_threshold": thr,
            "persistence": k,
            "deviations_in_order_of_onset": devs[:max_list],
            "n_deviating": len(devs),
            "sensors_still_normal": quiet if len(quiet) <= max_list else quiet[:max_list],
            "n_still_normal": len(quiet),
            "note": "first_tick is when the smoothed z-score first stayed beyond the threshold. "
            "Noisy sensors deviate later than quiet ones for the same physical change, so use "
            "the physics and the topology to order causes.",
        }
        if len(devs) > max_list:
            out["note"] += f" Only the first {max_list} of {len(devs)} deviations are listed."
        return out

    def list_recent_events(
        self, t_start: int | None = None, t_end: int | None = None
    ) -> dict[str, Any]:
        a, b = self._clamp(t_start, t_end)
        evs = [
            {
                "tick": e.tick,
                "kind": e.kind,
                "stage": e.stage,
                "description": e.description,
            }
            for e in self.events
            if a <= e.tick <= b
        ]
        return {"window": [a, b], "events": evs, "count": len(evs)}

    def check_sensor_health(self, stage: str, sensor: str) -> dict[str, Any]:
        sid, x = self._series(stage, sensor)
        recent = x[max(self.baseline_ticks, self.t_now - 59) : self.t_now + 1]
        valid = recent[~np.isnan(recent)]
        mu, sd = self._baseline(sid)
        flags: list[str] = []
        if np.isnan(recent).mean() > 0.2:
            flags.append("missing_data")
        if len(valid) >= 10 and float(np.std(valid)) == 0.0 and self._varies_when_healthy(sid):
            flags.append("flatline")
        if len(valid) and np.all(valid == 0.0) and self._zero_is_abnormal(sid):
            flags.append("stuck_zero")
        if not self._varies_when_healthy(sid):
            flags.append("quantised_or_controlled_channel")
        if len(valid) and (valid.min() < 0 and mu > 0):
            flags.append("negative_reading")
        if len(valid) and abs(float(valid.mean()) - mu) > 10 * sd:
            flags.append("far_outside_baseline")
        coupled = self._coupled_sensors(stage, sensor)
        coupled_status = []
        for cs_stage, cs_sensor in coupled:
            c = self.compare_to_baseline(cs_stage, cs_sensor, 30)
            coupled_status.append(
                {
                    "sensor": c["sensor"],
                    "verdict": c["verdict"],
                    "shift_sigma": c.get("shift_sigma"),
                }
            )
        own = self.compare_to_baseline(stage, sensor, 30)
        return {
            "sensor": sid,
            "recent_window": [max(self.baseline_ticks, self.t_now - 59), self.t_now],
            "nan_fraction": _r(float(np.isnan(recent).mean())),
            "flags": flags,
            "own_verdict": own["verdict"],
            "own_shift_sigma": own.get("shift_sigma"),
            "physically_coupled_signals": coupled_status,
            "interpretation": (
                "If this sensor is abnormal while every physically coupled signal is normal, "
                "suspect a sensor fault rather than a process fault."
            ),
        }

    def _coupled_sensors(self, stage: str, sensor: str) -> list[tuple[str, str]]:
        """Same-stage sensors plus the immediately upstream and downstream stages' sensors.

        Capped so the tool result stays small on lines with many sensors: when a neighbour
        stage has more than a few sensors, keep the ones that moved most.
        """
        topo = self.topology
        out: list[tuple[str, str]] = [(stage, m) for m in topo.sensors[stage] if m != sensor]
        neighbours = topo.upstream.get(stage, []) + topo.downstream(stage)
        for nb in neighbours:
            cands = [(nb, m) for m in topo.sensors[nb]]
            if len(cands) > 4:
                scored = []
                for st, m in cands:
                    c = self.compare_to_baseline(st, m, 30)
                    scored.append((abs(c.get("shift_sigma") or 0.0), (st, m)))
                scored.sort(reverse=True)
                cands = [sm for _, sm in scored[:4]]
            out.extend(cands)
        return out[:12]

    # ---------------------------------------------------------------- dispatch
    def call_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        self.call_count += 1
        fn = {
            "get_line_topology": self.get_line_topology,
            "query_sensor": self.query_sensor,
            "compare_to_baseline": self.compare_to_baseline,
            "find_first_deviation": self.find_first_deviation,
            "list_recent_events": self.list_recent_events,
            "check_sensor_health": self.check_sensor_health,
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'"}, True
        try:
            clean = {k: v for k, v in (args or {}).items() if v is not None}
            return fn(**clean), False
        except ToolError as e:
            return {"error": str(e)}, True
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}, True
        except Exception as e:  # noqa: BLE001 - surface anything else to the model
            return {"error": f"{type(e).__name__}: {e}"}, True


def _first_run_len(mask: np.ndarray, k: int) -> int | None:
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k:
            return i - k + 1
    return None


def _nullable(t: str, **extra: Any) -> dict[str, Any]:
    return {"type": [t, "null"], **extra}


def build_tool_specs(topology: Topology) -> list[dict[str, Any]]:
    """Strict tool definitions with stage/sensor enums taken from the topology."""
    stages = list(topology.stages)
    sensor_names = topology.all_sensor_names()
    specs: list[dict[str, Any]] = [
        {
            "name": "get_line_topology",
            "description": "Stage order (with parallel branches and merges), sensors per stage, "
            "typical lags, and the causal rules linking signals. Call once at the start.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "required": [],
            },
        },
        {
            "name": "find_first_deviation",
            "description": "For every sensor, the first tick after the baseline window at which "
            "its smoothed z-score stayed beyond z_threshold for `persistence` ticks, plus "
            "missing / flatline / stuck-zero detection. Returns deviations ordered by onset "
            "and the sensors that are still normal. Defaults (2.5, 3) are more sensitive than "
            "the alarm detector.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "z_threshold": _nullable("number", description="Default 2.5. Minimum 1.0."),
                    "persistence": _nullable(
                        "integer", description="Consecutive ticks. Default 3."
                    ),
                    "t_start": _nullable(
                        "integer", description="Search start tick (default: baseline end)."
                    ),
                    "t_end": _nullable("integer", description="Search end tick (default: now)."),
                },
                "additionalProperties": False,
                "required": ["z_threshold", "persistence", "t_start", "t_end"],
            },
        },
        {
            "name": "compare_to_baseline",
            "description": "Recent window of one sensor vs its baseline: shift in sigma units, "
            "percent change, trend slope, NaN fraction, and a verdict (normal / "
            "slightly_elevated / elevated / depressed / missing / flatline / stuck_zero).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": stages},
                    "sensor": {"type": "string", "enum": sensor_names},
                    "window_ticks": _nullable(
                        "integer", description="Recent window length. Default 30."
                    ),
                },
                "additionalProperties": False,
                "required": ["stage", "sensor", "window_ticks"],
            },
        },
        {
            "name": "query_sensor",
            "description": "Time series for one sensor over a tick range (downsampled to "
            "max_points bucket means), with summary stats and the baseline mean/std for "
            "reference. Use it to see the shape and timing of a deviation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": stages},
                    "sensor": {"type": "string", "enum": sensor_names},
                    "t_start": _nullable("integer"),
                    "t_end": _nullable("integer"),
                    "max_points": _nullable("integer", description="4-200, default 40."),
                },
                "additionalProperties": False,
                "required": ["stage", "sensor", "t_start", "t_end", "max_points"],
            },
        },
        {
            "name": "list_recent_events",
            "description": "Operator / maintenance event log (setpoint changes, maintenance, "
            "shift changes) within a tick range. Setpoint changes have real physical effects; "
            "the rest are informational. May be empty.",
            "input_schema": {
                "type": "object",
                "properties": {"t_start": _nullable("integer"), "t_end": _nullable("integer")},
                "additionalProperties": False,
                "required": ["t_start", "t_end"],
            },
        },
        {
            "name": "check_sensor_health",
            "description": "Data-quality flags for one sensor (missing, flatline, stuck zero, far "
            "outside baseline) and the verdicts of its physically coupled signals (same stage "
            "and neighbouring stages), to separate sensor faults from process faults.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": stages},
                    "sensor": {"type": "string", "enum": sensor_names},
                },
                "additionalProperties": False,
                "required": ["stage", "sensor"],
            },
        },
        {
            "name": "submit_diagnosis",
            "description": "Submit the final structured diagnosis. Call exactly once, when the "
            "evidence is gathered. This ends the investigation.",
            "input_schema": strict_schema(Diagnosis),
        },
    ]
    for spec in specs:
        spec["strict"] = True
    return specs


# Default specs for the simulated line (kept for convenience and tests).
TOOL_SPECS: list[dict[str, Any]] = build_tool_specs(get_topology())
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def tool_specs_json(topology: Topology | None = None) -> str:
    return json.dumps(build_tool_specs(topology) if topology else TOOL_SPECS, indent=2)
