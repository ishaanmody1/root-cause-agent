from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rootcause.agent.schema import DiagFaultType, Diagnosis, Status
from rootcause.sim.faults import GroundTruth


class Prediction(BaseModel):
    status: Status
    root_cause_stage: str | None
    fault_type: DiagFaultType
    affected_sensor: str | None = None
    confidence: float

    @classmethod
    def from_diagnosis(cls, d: Diagnosis) -> Prediction:
        return cls(
            status=d.status,
            root_cause_stage=d.root_cause_stage,
            fault_type=d.fault_type,
            affected_sensor=d.affected_sensor,
            confidence=d.confidence,
        )


class Score(BaseModel):
    stage_correct: bool
    type_correct: bool
    fully_correct: bool
    abstained: bool  # insufficient_evidence
    false_positive: bool  # said fault on a no-fault scenario
    confident_error: bool  # wrong with confidence >= 0.75


def score(pred: Prediction, truth: GroundTruth) -> Score:
    abstained = pred.status == "insufficient_evidence"
    if truth.fault_type == "none":
        stage_ok = pred.status == "no_fault"
        type_ok = pred.status == "no_fault" and pred.fault_type == "none"
        fp = pred.status == "fault_diagnosed"
    else:
        stage_ok = (
            pred.status == "fault_diagnosed" and pred.root_cause_stage == truth.root_cause_stage
        )
        type_ok = pred.status == "fault_diagnosed" and pred.fault_type == truth.fault_type
        fp = False
    fully = stage_ok and type_ok
    return Score(
        stage_correct=stage_ok,
        type_correct=type_ok,
        fully_correct=fully,
        abstained=abstained,
        false_positive=fp,
        confident_error=(not fully) and (not abstained) and pred.confidence >= 0.75,
    )


class SystemResult(BaseModel):
    prediction: Prediction
    score: Score
    status: str = "ok"
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    run_path: str | None = None
    dashboard_path: str | None = None


class ScenarioResult(BaseModel):
    scenario_id: str
    kind: str
    truth: GroundTruth
    detector_triggered: bool
    first_alarm_sensor: str | None
    first_alarm_stage: str | None
    alarm_tick: int | None
    detection_delay: int | None
    misleading: bool | None
    t_now: int | None
    systems: dict[str, SystemResult] = Field(default_factory=dict)


class EvalSummary(BaseModel):
    n_scenarios: int
    n_triggered: int
    detector: dict[str, Any]
    systems: dict[str, dict[str, Any]]
    config: dict[str, Any]
