"""Fault and event specifications with known ground truth."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FaultType = Literal[
    "clog", "overheating", "sensor_drift", "sensor_failure", "supply_drop", "none"
]

FAULT_TYPES: list[str] = ["clog", "overheating", "sensor_drift", "sensor_failure", "supply_drop"]


class Fault(BaseModel):
    """A single injected fault. ``stage``/``sensor`` identify where the fault physically is."""

    fault_type: FaultType
    stage: str | None = None
    sensor: str | None = None
    onset_tick: int
    # Severity semantics by type:
    #   clog: fractional resistance increase (0.15-0.6)
    #   overheating: fractional heat-input increase (0.1-0.4)
    #   supply_drop: fractional supply decrease (0.1-0.4)
    #   sensor_drift: fractional drift of the measured value at full ramp (signed)
    #   sensor_failure: ignored (see failure_mode)
    severity: float = 0.3
    # Ticks to reach full severity; 0 = step change.
    ramp_ticks: int = 40
    failure_mode: Literal["nan", "flatline", "zero"] | None = None

    def ramp(self, t: int) -> float:
        if t < self.onset_tick:
            return 0.0
        if self.ramp_ticks <= 0:
            return 1.0
        return min(1.0, (t - self.onset_tick) / self.ramp_ticks)

    def describe(self) -> str:
        match self.fault_type:
            case "clog":
                return (
                    f"Clog at {self.stage}: resistance rises {self.severity:.0%} over "
                    f"{self.ramp_ticks} ticks from t={self.onset_tick}."
                )
            case "overheating":
                return (
                    f"Overheating at {self.stage}: heat input rises {self.severity:.0%} over "
                    f"{self.ramp_ticks} ticks from t={self.onset_tick}."
                )
            case "supply_drop":
                return (
                    f"Upstream supply drop into feed: supply falls {self.severity:.0%} over "
                    f"{self.ramp_ticks} ticks from t={self.onset_tick}."
                )
            case "sensor_drift":
                return (
                    f"Sensor drift on {self.stage}.{self.sensor}: reading drifts "
                    f"{self.severity:+.0%} of nominal over {self.ramp_ticks} ticks from "
                    f"t={self.onset_tick}. No physical effect."
                )
            case "sensor_failure":
                return (
                    f"Sensor failure on {self.stage}.{self.sensor} ({self.failure_mode}) from "
                    f"t={self.onset_tick}. No physical effect."
                )
            case _:
                return "No fault."


class Event(BaseModel):
    """An operational event visible to the agent via ``list_recent_events``.

    ``setpoint_change`` has a real physical effect (heat input multiplier at a stage).
    ``maintenance`` and ``shift_change`` are benign distractors with no effect.
    """

    tick: int
    kind: Literal["setpoint_change", "maintenance", "shift_change"]
    stage: str | None = None
    description: str
    heat_multiplier: float | None = None


class GroundTruth(BaseModel):
    fault_type: FaultType
    root_cause_stage: str | None
    sensor: str | None = None
    onset_tick: int | None = None
    severity: float | None = None
    description: str
    # True when the detector's first alarm fired at a stage other than the root cause.
    # Filled in after detection runs; None until then.
    misleading: bool | None = None


class ScenarioSpec(BaseModel):
    scenario_id: str
    seed: int
    n_ticks: int = 600
    baseline_ticks: int = 200
    fault: Fault | None = None
    events: list[Event] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
