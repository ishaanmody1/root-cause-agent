"""Inject labelled sensor faults into real data.

A sensor fault changes one measured signal and nothing else, so it can be injected into
real data with full ground truth even though the plant physics are unknown. This gives a
quantitative check that the agent separates instrument problems from process problems in
the presence of real-world drift and artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rootcause.sim.faults import Fault, GroundTruth
from rootcause.sim.topology import Topology


@dataclass
class RealScenario:
    scenario_id: str
    data: pd.DataFrame
    topology: Topology
    truth: GroundTruth
    baseline_ticks: int
    fault: Fault | None = None


def inject_sensor_fault(
    data: pd.DataFrame,
    topology: Topology,
    rng: np.random.Generator,
    baseline_ticks: int,
    kind: str | None = None,
    stage: str | None = None,
    sensor: str | None = None,
    onset: int | None = None,
    scenario_id: str | None = None,
) -> RealScenario:
    kind = kind or str(rng.choice(["sensor_drift", "sensor_failure"]))
    # Avoid output-measurement channels as injection targets: several are near zero or
    # already glitchy in the raw data, which makes labels ambiguous.
    cands = [
        (s, m) for s in topology.stages for m in topology.sensors[s] if not s.endswith("Output")
    ]
    if stage is None or sensor is None:
        stage, sensor = cands[int(rng.integers(len(cands)))]
    sid = f"{stage}.{sensor}"
    n = len(data)
    onset = onset if onset is not None else int(rng.integers(baseline_ticks + 30, n - 120))
    df = data.copy()
    x = df[sid].to_numpy(dtype=float).copy()
    nominal = float(np.nanmedian(x[:baseline_ticks]))
    scale = max(abs(nominal), float(np.nanstd(x[:baseline_ticks])) * 10, 1e-6)
    if kind == "sensor_drift":
        sev = float(rng.choice([-1, 1])) * float(rng.uniform(0.08, 0.3))
        ramp_ticks = int(rng.integers(30, 120))
        fault = Fault(
            fault_type="sensor_drift",
            stage=stage,
            sensor=sensor,
            onset_tick=onset,
            severity=sev,
            ramp_ticks=ramp_ticks,
        )
        ramp = np.array([fault.ramp(int(t)) for t in range(n)])
        x = x + sev * scale * ramp
    else:
        mode = str(rng.choice(["nan", "flatline", "zero"]))
        fault = Fault(
            fault_type="sensor_failure",
            stage=stage,
            sensor=sensor,
            onset_tick=onset,
            severity=0.0,
            ramp_ticks=0,
            failure_mode=mode,
        )  # type: ignore[arg-type]
        if mode == "nan":
            x[onset:] = np.nan
        elif mode == "zero":
            x[onset:] = 0.0
        else:
            x[onset:] = x[onset - 1]
    df[sid] = x
    truth = GroundTruth(
        fault_type=fault.fault_type,
        root_cause_stage=stage,
        sensor=sensor,
        onset_tick=onset,
        severity=fault.severity,
        description=fault.describe() + " (injected into real data)",
    )
    return RealScenario(
        scenario_id=scenario_id or f"real-{kind}-{stage}.{sensor}-{onset}",
        data=df,
        topology=topology,
        truth=truth,
        baseline_ticks=baseline_ticks,
        fault=fault,
    )
