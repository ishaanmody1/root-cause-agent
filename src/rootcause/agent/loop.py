"""Hand-rolled agent loop with budgets, tracing, and graceful failure.

The loop owns: the message history, tool dispatch, step/cost budgets, token accounting,
a JSON-serialisable trace of every model call and tool call, and the fallbacks when the
model refuses, runs out of tokens, or never submits a diagnosis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic
from anthropic.types import Message
from pydantic import BaseModel, Field, ValidationError

from rootcause.agent.pricing import estimate_cost_usd
from rootcause.agent.prompts import SYSTEM_PROMPT, user_prompt
from rootcause.agent.schema import Diagnosis
from rootcause.agent.tools import TOOL_SPECS, ToolContext

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AgentConfig(BaseModel):
    model: str = "claude-sonnet-5-5"
    max_steps: int = 12  # model calls per diagnosis
    max_cost_usd: float = 0.50
    max_tokens: int = 8000
    effort: Effort | None = "medium"
    grace_ticks: int = 25  # how long after the alarm the diagnosis is requested
    tool_result_max_chars: int = 8000
    prompt_cache: bool = True


class TraceStep(BaseModel):
    step: int
    kind: Literal["model_call", "tool_call", "nudge", "error"]
    wall_time: float
    latency_s: float | None = None
    stop_reason: str | None = None
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    is_error: bool = False
    usage: dict[str, int] | None = None
    cost_usd: float | None = None


class RunUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


RunStatus = Literal["ok", "budget_exhausted", "refused", "error"]


class DiagnosisRun(BaseModel):
    scenario_id: str
    model: str
    status: RunStatus
    diagnosis: Diagnosis | None
    t_now: int
    usage: RunUsage
    trace: list[TraceStep] = Field(default_factory=list)
    error: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    def write_trace_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for s in self.trace:
                f.write(s.model_dump_json() + "\n")


class ModelBackend(Protocol):
    """Anything that answers like ``client.messages.create``."""

    model: str

    def create(
        self,
        *,
        system: list[dict[str, Any]] | str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> Message: ...


class AnthropicBackend:
    def __init__(
        self, model: str, effort: Effort | None, client: anthropic.Anthropic | None = None
    ):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def create(self, *, system, tools, messages, max_tokens) -> Message:
        kwargs: dict[str, Any] = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            **kwargs,
        )


NUDGE_SUBMIT = (
    "You have not called submit_diagnosis. Finish now by calling submit_diagnosis with your "
    "best assessment (use status insufficient_evidence if the data does not support a "
    "conclusion)."
)
NUDGE_BUDGET = (
    "BUDGET EXHAUSTED. No more investigation is possible. Call submit_diagnosis now with your "
    "best current assessment and an honest confidence, or status insufficient_evidence."
)
NUDGE_TRUNCATED = (
    "Your previous response was cut off by the token limit. Continue more concisely and "
    "call the tools you need, or submit_diagnosis."
)


def _usage_dict(msg: Message) -> dict[str, int]:
    u = msg.usage
    return {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


class DiagnosisAgent:
    def __init__(self, backend: ModelBackend, config: AgentConfig | None = None):
        self.backend = backend
        self.cfg = config or AgentConfig()

    def _system(self) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": SYSTEM_PROMPT}
        if self.cfg.prompt_cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def run(self, ctx: ToolContext, scenario_id: str = "scenario") -> DiagnosisRun:
        cfg = self.cfg
        t0 = time.perf_counter()
        usage = RunUsage()
        trace: list[TraceStep] = []
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": user_prompt(ctx.detection_summary(), ctx.t_now, ctx.baseline_ticks),
            }
        ]
        diagnosis: Diagnosis | None = None
        status: RunStatus = "ok"
        error: str | None = None
        nudges = 0
        budget_nudged = False
        tools = TOOL_SPECS

        step = 0
        while True:
            step += 1
            over_steps = step > cfg.max_steps
            over_cost = usage.cost_usd >= cfg.max_cost_usd
            if (over_steps or over_cost) and not budget_nudged:
                # One last call, submit-only.
                budget_nudged = True
                status = "budget_exhausted"
                tools = [t for t in TOOL_SPECS if t["name"] == "submit_diagnosis"]
                messages.append({"role": "user", "content": NUDGE_BUDGET})
                trace.append(
                    TraceStep(step=step, kind="nudge", wall_time=time.time(), text=NUDGE_BUDGET)
                )
            elif (over_steps or over_cost) and budget_nudged:
                break

            t_call = time.perf_counter()
            try:
                resp = self.backend.create(
                    system=self._system(),
                    tools=tools,
                    messages=messages,
                    max_tokens=cfg.max_tokens,
                )
            except anthropic.RateLimitError as e:
                status, error = "error", f"rate limited after SDK retries: {e}"
                trace.append(
                    TraceStep(
                        step=step, kind="error", wall_time=time.time(), text=error, is_error=True
                    )
                )
                break
            except anthropic.APIStatusError as e:
                status, error = "error", f"API error {e.status_code}: {e.message}"
                trace.append(
                    TraceStep(
                        step=step, kind="error", wall_time=time.time(), text=error, is_error=True
                    )
                )
                break
            except anthropic.APIConnectionError as e:
                status, error = "error", f"connection error: {e}"
                trace.append(
                    TraceStep(
                        step=step, kind="error", wall_time=time.time(), text=error, is_error=True
                    )
                )
                break
            latency = time.perf_counter() - t_call

            u = _usage_dict(resp)
            cost = estimate_cost_usd(self.backend.model, **u)
            usage.input_tokens += u["input_tokens"]
            usage.output_tokens += u["output_tokens"]
            usage.cache_read_input_tokens += u["cache_read_input_tokens"]
            usage.cache_creation_input_tokens += u["cache_creation_input_tokens"]
            usage.cost_usd += cost
            usage.model_calls += 1

            text = "\n".join(b.text for b in resp.content if b.type == "text") or None
            trace.append(
                TraceStep(
                    step=step,
                    kind="model_call",
                    wall_time=time.time(),
                    latency_s=round(latency, 3),
                    stop_reason=resp.stop_reason,
                    text=text,
                    usage=u,
                    cost_usd=round(cost, 6),
                )
            )

            if resp.stop_reason == "refusal":
                status = "refused"
                details = getattr(resp, "stop_details", None)
                error = f"model refused: {getattr(details, 'explanation', None) or 'no details'}"
                break

            # Echo the assistant turn back verbatim (including thinking blocks).
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]

            if resp.stop_reason == "pause_turn":
                continue

            if not tool_uses:
                nudges += 1
                if nudges > 2:
                    status, error = "error", "model ended without calling submit_diagnosis"
                    break
                nudge = NUDGE_TRUNCATED if resp.stop_reason == "max_tokens" else NUDGE_SUBMIT
                messages.append({"role": "user", "content": nudge})
                trace.append(TraceStep(step=step, kind="nudge", wall_time=time.time(), text=nudge))
                continue

            results: list[dict[str, Any]] = []
            for tu in tool_uses:
                args = tu.input if isinstance(tu.input, dict) else {}
                is_error = False
                if tu.name == "submit_diagnosis":
                    try:
                        cand = Diagnosis.model_validate(args)
                        cand.confidence = min(1.0, max(0.0, cand.confidence))
                        for h in cand.alternative_hypotheses:
                            h.confidence = min(1.0, max(0.0, h.confidence))
                        diagnosis = cand
                        out: dict[str, Any] = {"accepted": True}
                    except ValidationError as e:
                        is_error = True
                        out = {"error": f"invalid diagnosis: {e.errors()[:3]}"}
                else:
                    out, is_error = ctx.call_tool(tu.name, args)
                usage.tool_calls += 1
                payload = json.dumps(out, separators=(",", ":"))
                if len(payload) > cfg.tool_result_max_chars:
                    payload = payload[: cfg.tool_result_max_chars] + '..."TRUNCATED"}'
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": payload,
                        "is_error": is_error,
                    }
                )
                trace.append(
                    TraceStep(
                        step=step,
                        kind="tool_call",
                        wall_time=time.time(),
                        tool_name=tu.name,
                        tool_input=args,
                        tool_output=payload[:2000],
                        is_error=is_error,
                    )
                )
            messages.append({"role": "user", "content": results})

            if diagnosis is not None:
                break
            if resp.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": NUDGE_TRUNCATED})
                trace.append(
                    TraceStep(step=step, kind="nudge", wall_time=time.time(), text=NUDGE_TRUNCATED)
                )

        usage.latency_s = round(time.perf_counter() - t0, 3)
        usage.cost_usd = round(usage.cost_usd, 6)

        if diagnosis is None:
            diagnosis = Diagnosis(
                status="insufficient_evidence",
                root_cause_stage=None,
                fault_type="none",
                affected_sensor=None,
                confidence=0.0,
                evidence=[],
                alternative_hypotheses=[],
                recommended_action="Agent did not reach a diagnosis; review the trace and re-run "
                "with a larger budget or escalate to a human.",
                requires_downtime=False,
                reasoning_summary=f"No diagnosis submitted. Run status: {status}. {error or ''}".strip(),
            )

        return DiagnosisRun(
            scenario_id=scenario_id,
            model=self.backend.model,
            status=status,
            diagnosis=diagnosis,
            t_now=ctx.t_now,
            usage=usage,
            trace=trace,
            error=error,
        )
