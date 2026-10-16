import anthropic
from anthropic.types import Message, TextBlock, Usage

from rootcause.agent import AgentConfig, DiagnosisAgent, FakeRuleBasedBackend, ToolContext
from rootcause.detect import detect
from rootcause.sim import make_scenario, simulate


def test_rule_backend_reaches_correct_diagnosis(clog_scenario):
    sc, det = clog_scenario
    run = DiagnosisAgent(FakeRuleBasedBackend(), AgentConfig(model="fake-rule-based")).run(
        ToolContext.from_scenario(sc, det), sc.scenario_id
    )
    assert run.status == "ok"
    assert run.diagnosis.root_cause_stage == "heater" and run.diagnosis.fault_type == "clog"
    kinds = [s.kind for s in run.trace]
    assert "model_call" in kinds and "tool_call" in kinds
    assert run.usage.tool_calls >= 4 and run.usage.cost_usd == 0.0


def test_budget_exhaustion_forces_submit(clog_scenario):
    sc, det = clog_scenario
    run = DiagnosisAgent(
        FakeRuleBasedBackend(), AgentConfig(model="fake-rule-based", max_steps=1)
    ).run(ToolContext.from_scenario(sc, det), sc.scenario_id)
    assert run.status == "budget_exhausted"
    assert run.diagnosis is not None
    assert any(s.kind == "nudge" for s in run.trace)


class _Silent:
    """Backend that never calls tools: exercises the nudge path and the synthesized fallback."""

    model = "silent"

    def create(self, *, system, tools, messages, max_tokens):
        return Message(
            id="m",
            type="message",
            role="assistant",
            model="silent",
            content=[TextBlock(type="text", text="I think it is the press.")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=5),
        )


def test_model_that_never_submits_yields_insufficient_evidence(clog_scenario):
    sc, det = clog_scenario
    run = DiagnosisAgent(_Silent(), AgentConfig(model="silent")).run(
        ToolContext.from_scenario(sc, det), "x"
    )
    assert run.status == "error"
    assert run.diagnosis.status == "insufficient_evidence"
    assert sum(s.kind == "nudge" for s in run.trace) >= 2


class _Broken:
    model = "broken"

    def create(self, **kw):
        raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]


def test_api_errors_are_captured(clog_scenario):
    sc, det = clog_scenario
    run = DiagnosisAgent(_Broken(), AgentConfig(model="broken")).run(
        ToolContext.from_scenario(sc, det), "x"
    )
    assert run.status == "error" and "connection" in (run.error or "")
    assert run.diagnosis.status == "insufficient_evidence"


def test_no_fault_scenario_handled():
    sc = simulate(make_scenario("normal_setpoint_change", 8))
    det = detect(sc.data)
    assert det.triggered
    run = DiagnosisAgent(FakeRuleBasedBackend(), AgentConfig(model="fake-rule-based")).run(
        ToolContext.from_scenario(sc, det), sc.scenario_id
    )
    assert run.diagnosis.status == "no_fault"
