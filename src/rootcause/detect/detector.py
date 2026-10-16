"""Cheap, deterministic anomaly detection. No LLM involved.

Per sensor: robust baseline (median / MAD) over the first ``baseline_ticks``, then an
EWMA-smoothed z-score that must exceed ``z_threshold`` for ``persistence`` consecutive
ticks. Separate rules catch missing data (NaN runs), flatlines, and stuck-at-zero.

The detector reports every sensor that eventually alarms, in order of when the alarm
condition *started*, plus the first alarm that would have paged a human. The naive
baseline blames the first alarm's stage; the agent is triggered by it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from rootcause.sim.topology import split_sensor_id

AlarmKind = Literal["high", "low", "missing", "flatline", "stuck_zero"]


class DetectorConfig(BaseModel):
    baseline_ticks: int = 200
    z_threshold: float = 4.0
    persistence: int = 5
    ewma_alpha: float = 0.3
    nan_run_threshold: int = 5
    flatline_window: int = 15
    # Floor on the robust std so near-constant sensors do not produce infinite z.
    min_sigma_rel: float = 1e-3


class Baseline(BaseModel):
    mean: float
    std: float
    n: int
    # Fraction of consecutive identical values in the baseline. Controlled or lab-value
    # channels sit exactly constant for long stretches, so flatline / stuck-zero rules only
    # apply to sensors that actually vary when healthy.
    repeat_frac: float = 0.0
    had_zeros: bool = False

    @property
    def quantised(self) -> bool:
        return self.repeat_frac > 0.3


class SensorAlarm(BaseModel):
    sensor: str
    stage: str
    kind: AlarmKind
    onset_tick: int  # first tick of the run that produced the alarm
    alarm_tick: int  # tick at which the persistence rule confirmed it
    z_at_alarm: float | None = None
    value_at_alarm: float | None = None

    def describe(self) -> str:
        if self.kind in ("high", "low"):
            return (
                f"{self.sensor} {self.kind} (z={self.z_at_alarm:+.1f}) from t={self.onset_tick}, "
                f"confirmed t={self.alarm_tick}"
            )
        return f"{self.sensor} {self.kind} from t={self.onset_tick}, confirmed t={self.alarm_tick}"


class Detection(BaseModel):
    triggered: bool
    config: DetectorConfig
    baselines: dict[str, Baseline]
    alarms: list[SensorAlarm] = Field(default_factory=list)  # sorted by onset_tick
    first_alarm: SensorAlarm | None = None  # earliest *confirmed* alarm (what pages a human)
    alarm_tick: int | None = None

    def alarms_by_stage(self) -> dict[str, list[SensorAlarm]]:
        out: dict[str, list[SensorAlarm]] = {}
        for a in self.alarms:
            out.setdefault(a.stage, []).append(a)
        return out


def robust_baseline(x: np.ndarray) -> Baseline:
    v = x[~np.isnan(x)]
    if len(v) == 0:
        return Baseline(mean=float("nan"), std=float("nan"), n=0)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad == 0.0:
        mad = float(np.std(v))
    repeat = float(np.mean(np.diff(v) == 0.0)) if len(v) > 1 else 1.0
    return Baseline(
        mean=med, std=mad, n=int(len(v)), repeat_frac=repeat, had_zeros=bool(np.any(v == 0.0))
    )


def _first_run(mask: np.ndarray, k: int) -> tuple[int, int] | None:
    """Return (run_start, confirm_index) of the first run of ``k`` consecutive Trues."""
    if k <= 0 or len(mask) == 0:
        return None
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k:
            return i - k + 1, i
    return None


def ewma(z: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(z)
    acc = 0.0
    for i, v in enumerate(z):
        if np.isnan(v):
            out[i] = np.nan
            continue
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def detect(data: pd.DataFrame, config: DetectorConfig | None = None) -> Detection:
    cfg = config or DetectorConfig()
    sensors = [c for c in data.columns if c != "t"]
    ticks = data["t"].to_numpy()
    baselines: dict[str, Baseline] = {}
    alarms: list[SensorAlarm] = []

    for sid in sensors:
        x = data[sid].to_numpy(dtype=float)
        b = robust_baseline(x[: cfg.baseline_ticks])
        baselines[sid] = b
        stage, _ = split_sensor_id(sid)
        post = x[cfg.baseline_ticks :]
        post_ticks = ticks[cfg.baseline_ticks :]
        if b.n == 0:
            continue

        # Missing data.
        hit = _first_run(np.isnan(post), cfg.nan_run_threshold)
        if hit:
            s, e = hit
            alarms.append(
                SensorAlarm(
                    sensor=sid,
                    stage=stage,
                    kind="missing",
                    onset_tick=int(post_ticks[s]),
                    alarm_tick=int(post_ticks[e]),
                )
            )
            continue  # a sensor with no data has nothing else to say

        # Stuck at zero.
        hit = _first_run(post == 0.0, cfg.persistence)
        if hit and abs(b.mean) > 5 * max(b.std, 1e-9) and not b.had_zeros:
            s, e = hit
            alarms.append(
                SensorAlarm(
                    sensor=sid,
                    stage=stage,
                    kind="stuck_zero",
                    onset_tick=int(post_ticks[s]),
                    alarm_tick=int(post_ticks[e]),
                    value_at_alarm=0.0,
                )
            )
            continue

        # Flatline: identical consecutive values over a window (noise makes this impossible
        # for a healthy sensor).
        same = np.concatenate([[False], np.diff(post) == 0.0])
        hit = _first_run(same, cfg.flatline_window) if not b.quantised else None
        if hit:
            s, e = hit
            alarms.append(
                SensorAlarm(
                    sensor=sid,
                    stage=stage,
                    kind="flatline",
                    onset_tick=int(post_ticks[max(s - 1, 0)]),
                    alarm_tick=int(post_ticks[e]),
                    value_at_alarm=float(post[e]),
                )
            )
            continue

        sigma = max(b.std, abs(b.mean) * cfg.min_sigma_rel, 1e-9)
        z = ewma((post - b.mean) / sigma, cfg.ewma_alpha)
        hi = _first_run(z > cfg.z_threshold, cfg.persistence)
        lo = _first_run(z < -cfg.z_threshold, cfg.persistence)
        cands = [(h, "high") for h in [hi] if h] + [(x, "low") for x in [lo] if x]
        if cands:
            (s, e), kind = min(cands, key=lambda c: c[0][1])
            alarms.append(
                SensorAlarm(
                    sensor=sid,
                    stage=stage,
                    kind=kind,  # type: ignore[arg-type]
                    onset_tick=int(post_ticks[s]),
                    alarm_tick=int(post_ticks[e]),
                    z_at_alarm=float(z[e]),
                    value_at_alarm=float(post[e]),
                )
            )

    alarms.sort(key=lambda a: (a.onset_tick, a.alarm_tick))
    first = min(alarms, key=lambda a: (a.alarm_tick, a.onset_tick)) if alarms else None
    return Detection(
        triggered=first is not None,
        config=cfg,
        baselines=baselines,
        alarms=alarms,
        first_alarm=first,
        alarm_tick=first.alarm_tick if first else None,
    )
