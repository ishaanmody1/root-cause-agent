"""Random but reproducible scenario generation.

``generate_suite`` produces a balanced mix of fault kinds, including:

* clogs at upstream stages whose first visible symptom is downstream (the misleading case
  the project exists to solve),
* sensor faults that mimic process faults on a single signal,
* normal runs, with and without benign operator events, for false-positive measurement.
"""

from __future__ import annotations

import numpy as np

from rootcause.sim.faults import Event, Fault, ScenarioSpec
from rootcause.sim.topology import SENSORS, STAGES

SCENARIO_KINDS: list[str] = [
    "clog",
    "overheating",
    "sensor_drift",
    "sensor_failure",
    "supply_drop",
    "normal",
    "normal_setpoint_change",
]

_MAINTENANCE_TEXT = [
    "Routine lubrication of {stage} drive, no process impact expected.",
    "Operator inspected {stage} guard panel; line kept running.",
    "Filter housing on {stage} visually checked, not replaced.",
]


def _onset(rng: np.random.Generator, spec_ticks: int, baseline: int) -> int:
    lo = baseline + 60
    hi = spec_ticks - 150
    return int(rng.integers(lo, hi))


def _benign_events(rng: np.random.Generator, n_ticks: int, onset: int | None) -> list[Event]:
    events: list[Event] = []
    if rng.random() < 0.5:
        stage = str(rng.choice(STAGES))
        # Sometimes place the distractor suspiciously close to the fault onset.
        if onset is not None and rng.random() < 0.5:
            tick = int(np.clip(onset + rng.integers(-15, 15), 1, n_ticks - 1))
        else:
            tick = int(rng.integers(20, n_ticks - 20))
        events.append(
            Event(
                tick=tick,
                kind="maintenance",
                stage=stage,
                description=str(rng.choice(_MAINTENANCE_TEXT)).format(stage=stage),
            )
        )
    if rng.random() < 0.3:
        tick = int(rng.integers(20, n_ticks - 20))
        events.append(
            Event(tick=tick, kind="shift_change", description="Shift change: crew B took over.")
        )
    return sorted(events, key=lambda e: e.tick)


def make_scenario(kind: str, seed: int, n_ticks: int = 600, baseline_ticks: int = 200) -> ScenarioSpec:
    rng = np.random.default_rng(seed)
    sid = f"{kind}-{seed:04d}"
    fault: Fault | None = None
    events: list[Event] = []
    tags = [kind]

    match kind:
        case "clog":
            # Bias towards upstream stages: those produce the misleading downstream symptoms.
            stage = str(rng.choice(["feed", "heater", "press"], p=[0.45, 0.3, 0.25]))
            fault = Fault(
                fault_type="clog",
                stage=stage,
                onset_tick=_onset(rng, n_ticks, baseline_ticks),
                severity=float(rng.uniform(0.15, 0.5)),
                ramp_ticks=int(rng.integers(30, 90)),
            )
        case "overheating":
            stage = str(rng.choice(["heater", "press", "feed"], p=[0.6, 0.25, 0.15]))
            fault = Fault(
                fault_type="overheating",
                stage=stage,
                onset_tick=_onset(rng, n_ticks, baseline_ticks),
                severity=float(rng.uniform(0.12, 0.35)),
                ramp_ticks=int(rng.integers(20, 80)),
            )
        case "supply_drop":
            fault = Fault(
                fault_type="supply_drop",
                stage="feed",
                onset_tick=_onset(rng, n_ticks, baseline_ticks),
                severity=float(rng.uniform(0.1, 0.35)),
                ramp_ticks=int(rng.integers(10, 80)),
            )
        case "sensor_drift":
            stage = str(rng.choice(STAGES))
            sensor = str(rng.choice(SENSORS[stage]))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            fault = Fault(
                fault_type="sensor_drift",
                stage=stage,
                sensor=sensor,
                onset_tick=_onset(rng, n_ticks, baseline_ticks),
                severity=sign * float(rng.uniform(0.08, 0.3)),
                ramp_ticks=int(rng.integers(40, 120)),
            )
        case "sensor_failure":
            stage = str(rng.choice(STAGES))
            sensor = str(rng.choice(SENSORS[stage]))
            fault = Fault(
                fault_type="sensor_failure",
                stage=stage,
                sensor=sensor,
                onset_tick=_onset(rng, n_ticks, baseline_ticks),
                severity=0.0,
                ramp_ticks=0,
                failure_mode=str(rng.choice(["nan", "flatline", "zero"])),  # type: ignore[arg-type]
            )
        case "normal":
            pass
        case "normal_setpoint_change":
            stage = str(rng.choice(["heater", "press"], p=[0.7, 0.3]))
            mult = float(rng.choice([0.92, 0.94, 1.06, 1.08, 1.1]))
            tick = _onset(rng, n_ticks, baseline_ticks)
            events.append(
                Event(
                    tick=tick,
                    kind="setpoint_change",
                    stage=stage,
                    heat_multiplier=mult,
                    description=(
                        f"Operator changed {stage} heat setpoint to {mult:.0%} of nominal "
                        f"(planned recipe adjustment)."
                    ),
                )
            )
            tags.append("benign_event")
        case _:
            raise ValueError(f"unknown scenario kind: {kind}")

    onset = fault.onset_tick if fault else (events[0].tick if events else None)
    events = sorted(events + _benign_events(rng, n_ticks, onset), key=lambda e: e.tick)
    return ScenarioSpec(
        scenario_id=sid,
        seed=seed,
        n_ticks=n_ticks,
        baseline_ticks=baseline_ticks,
        fault=fault,
        events=events,
        tags=tags,
    )


def generate_suite(n: int, seed: int = 0, kinds: list[str] | None = None) -> list[ScenarioSpec]:
    """Round-robin over kinds so the suite is balanced; seeds are deterministic in ``seed``."""
    kinds = kinds or SCENARIO_KINDS
    rng = np.random.default_rng(seed)
    specs: list[ScenarioSpec] = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        specs.append(make_scenario(kind, seed=int(rng.integers(0, 9999))))
    return specs
