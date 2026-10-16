"""Naive baseline: blame the stage where the first alarm fired, guess the type from the signal."""

from __future__ import annotations

from rootcause.detect.detector import Detection
from rootcause.evals.types import Prediction
from rootcause.sim.topology import split_sensor_id


def naive_prediction(det: Detection) -> Prediction:
    if not det.triggered or det.first_alarm is None:
        return Prediction(
            status="no_fault", root_cause_stage=None, fault_type="none", confidence=0.7
        )
    a = det.first_alarm
    stage, measure = split_sensor_id(a.sensor)
    sensor: str | None = None
    match (measure, a.kind):
        case (_, "missing" | "flatline" | "stuck_zero"):
            fault, sensor = "sensor_failure", a.sensor
        case ("flow", "low"):
            fault = "supply_drop" if stage == "feed" else "clog"
        case ("pressure", "high") | ("pressure", "low") | ("quality", "low"):
            fault = "clog"
        case ("temperature", "high") | ("power", "high"):
            fault = "overheating"
        case _:
            fault, sensor = "sensor_drift", a.sensor
    return Prediction(
        status="fault_diagnosed",
        root_cause_stage=stage,
        fault_type=fault,  # type: ignore[arg-type]
        affected_sensor=sensor,
        confidence=0.7,
    )
