"""Structured diagnosis output and a helper to turn pydantic models into strict tool schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["fault_diagnosed", "no_fault", "insufficient_evidence"]
DiagFaultType = Literal[
    "clog", "overheating", "sensor_drift", "sensor_failure", "supply_drop", "none"
]


class Evidence(BaseModel):
    signal: str = Field(
        description="Sensor id as the tools report it ('<stage>.<sensor>'), or 'events'."
    )
    observation: str = Field(description="What the data shows, with ticks and magnitudes.")
    why_it_matters: str = Field(description="How this observation supports the diagnosis.")


class Hypothesis(BaseModel):
    root_cause_stage: str | None = Field(
        description="One of this line's stage names (from get_line_topology), or null."
    )
    fault_type: DiagFaultType
    confidence: float = Field(description="0-1 probability this hypothesis is correct.")
    why_less_likely: str


class Diagnosis(BaseModel):
    status: Status = Field(
        description="fault_diagnosed when a physical or sensor fault is identified; no_fault when "
        "the alarm is explained by normal operation or a logged operator event; "
        "insufficient_evidence when the data cannot support a conclusion."
    )
    root_cause_stage: str | None = Field(
        description="One of this line's stage names (from get_line_topology), or null. For a "
        "supply drop use the first stage."
    )
    fault_type: DiagFaultType
    affected_sensor: str | None = Field(
        description="For sensor_drift / sensor_failure: the faulty sensor id. Otherwise null."
    )
    confidence: float = Field(
        description="0-1 probability that root_cause_stage AND fault_type are both correct. "
        "Be calibrated: 0.9+ only when several independent, physically coupled signals agree."
    )
    evidence: list[Evidence] = Field(description="Ordered evidence chain, earliest signal first.")
    alternative_hypotheses: list[Hypothesis] = Field(
        description="Other explanations considered, with why they are less likely."
    )
    recommended_action: str = Field(
        description="Concrete next step for the reliability engineer: what to inspect, where."
    )
    requires_downtime: bool = Field(description="Whether the action needs the line stopped.")
    reasoning_summary: str = Field(
        description="Plain-language explanation of the causal chain from root cause to alarm."
    )

    def one_line(self) -> str:
        if self.status == "fault_diagnosed":
            where = self.root_cause_stage or "?"
            sensor = f" ({self.affected_sensor})" if self.affected_sensor else ""
            return f"{self.fault_type} at {where}{sensor}, confidence {self.confidence:.2f}"
        return f"{self.status}, confidence {self.confidence:.2f}"


_STRIP_KEYS = {"title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema -> strict tool input schema.

    Inlines ``$ref``s, sets ``additionalProperties: false`` and makes every property required
    (optional fields are expressed as nullable types instead), which is what strict tool use
    expects. Numeric bounds are stripped and enforced in Python after parsing.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"].rsplit("/", 1)[-1]
            return walk(defs[ref])
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _STRIP_KEYS:
                continue
            out[k] = walk(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out

    return walk(raw)
