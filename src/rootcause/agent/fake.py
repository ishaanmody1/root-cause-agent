"""A deterministic, rule-based ``ModelBackend``.

Two jobs:
1. Lets the whole pipeline (loop, tools, tracing, budgets, evals, UI) run offline and in CI
   with no API key.
2. Serves as the "rule-based tracer" baseline in the eval table: it walks the topology
   upstream from the first flow deviation, which is the obvious hand-written alternative to
   an LLM agent. It is deliberately simple and brittle on the ambiguous cases.

It speaks the Anthropic ``Message`` type so the loop cannot tell it apart from the real API.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from rootcause.sim.topology import SENSORS, STAGES


def _msg(blocks: list[Any], stop_reason: str, out_tokens: int = 50) -> Message:
    return Message(
        id="msg_fake",
        type="message",
        role="assistant",
        model="fake-rule-based",
        content=blocks,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        usage=Usage(input_tokens=0, output_tokens=out_tokens),
    )


def _tool(name: str, args: dict[str, Any], i: int) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=f"toolu_fake_{name}_{i}", name=name, input=args)


def _last_tool_results(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool_use_id -> parsed result from the most recent user message."""
    if not messages or messages[-1]["role"] != "user":
        return {}
    content = messages[-1]["content"]
    if not isinstance(content, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for blk in content:
        if isinstance(blk, dict) and blk.get("type") == "tool_result":
            try:
                out[blk["tool_use_id"]] = json.loads(blk["content"])
            except (json.JSONDecodeError, TypeError):
                out[blk["tool_use_id"]] = {}
    return out


class FakeRuleBasedBackend:
    model = "fake-rule-based"

    def __init__(self) -> None:
        self.phase = 0
        self.devs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.t_now: int = 0
        self.compare: dict[str, dict[str, Any]] = {}
        self.plan: dict[str, Any] = {}

    # The loop constructs one backend per run; keep state per instance.
    def create(self, *, system, tools, messages, max_tokens) -> Message:
        results = _last_tool_results(messages)
        tool_names = {t["name"] for t in tools}
        if tool_names == {"submit_diagnosis"} and self.phase < 3:
            # Budget nudge: submit whatever we have.
            return self._submit()
        if self.phase == 0:
            self.phase = 1
            return _msg(
                [
                    _tool("get_line_topology", {}, 0),
                    _tool(
                        "find_first_deviation",
                        {"z_threshold": 2.5, "persistence": 3, "t_start": None, "t_end": None},
                        1,
                    ),
                    _tool("list_recent_events", {"t_start": None, "t_end": None}, 2),
                ],
                "tool_use",
            )
        if self.phase == 1:
            for r in results.values():
                if "deviations_in_order_of_onset" in r:
                    self.devs = r["deviations_in_order_of_onset"]
                    self.t_now = r["t_now"]
                if "events" in r:
                    self.events = r["events"]
            self.plan = self._reason()
            confirm = self.plan.get("confirm", [])
            self.phase = 2
            if not confirm:
                return self._submit()
            return _msg(
                [
                    _tool(
                        "compare_to_baseline",
                        {"stage": s, "sensor": m, "window_ticks": 30},
                        i,
                    )
                    for i, (s, m) in enumerate(confirm)
                ],
                "tool_use",
            )
        if self.phase == 2:
            for r in results.values():
                if "sensor" in r:
                    self.compare[r["sensor"]] = r
            self._refine()
            return self._submit()
        self.phase = 4
        return _msg([TextBlock(type="text", text="Diagnosis submitted.")], "end_turn", 5)

    # ------------------------------------------------------------------ rules
    def _dev(self, sid: str) -> dict[str, Any] | None:
        for d in self.devs:
            if d["sensor"] == sid:
                return d
        return None

    def _dir(self, sid: str) -> str | None:
        d = self._dev(sid)
        return d["direction"] if d else None

    def _reason(self) -> dict[str, Any]:
        plan: dict[str, Any] = {"confirm": []}
        failures = [d for d in self.devs if d["direction"] in ("missing", "flatline", "stuck_zero")]
        if failures:
            f = failures[0]
            plan.update(
                status="fault_diagnosed",
                stage=f["stage"],
                fault="sensor_failure",
                sensor=f["sensor"],
                conf=0.9,
                why=f"{f['sensor']} reads {f['direction']} from t={f['first_tick']}.",
            )
            return plan

        low_flow_stages = [s for s in STAGES if self._dir(f"{s}.flow") == "low"]
        if low_flow_stages:
            k = low_flow_stages[0]
            i = STAGES.index(k)
            nxt = STAGES[i + 1] if i + 1 < len(STAGES) else None
            if k == "feed":
                propagated = nxt is not None and self._dir(f"{nxt}.flow") == "low"
                if self._dir("feed.pressure") == "high":
                    plan.update(
                        status="fault_diagnosed",
                        stage="feed",
                        fault="sensor_drift",
                        sensor="feed.flow",
                        conf=0.55,
                        why="feed.flow low but feed.pressure high: inconsistent with a real inlet drop.",
                    )
                elif propagated or self._dir("feed.pressure") == "low":
                    plan.update(
                        status="fault_diagnosed",
                        stage="feed",
                        fault="supply_drop",
                        conf=0.85,
                        why="feed.flow (inlet = supply) fell first and the drop propagated downstream.",
                    )
                    plan["confirm"] = [("feed", "pressure"), ("heater", "flow")]
                else:
                    plan.update(
                        status="fault_diagnosed",
                        stage="feed",
                        fault="sensor_drift",
                        sensor="feed.flow",
                        conf=0.6,
                        why="Only feed.flow moved; downstream flows and feed pressure are normal.",
                    )
                    plan["confirm"] = [("feed", "pressure"), ("heater", "flow")]
                return plan
            up = STAGES[i - 1]
            coupled = (
                (nxt is not None and self._dir(f"{nxt}.flow") == "low")
                or self._dir(f"{k}.pressure") == "low"
                or self._dir(f"{up}.pressure") == "high"
                or self._dir(f"{up}.temperature") == "high"
                or self._dir(f"{k}.temperature") == "high"
            )
            if coupled:
                conf = 0.6
                conf += 0.1 if self._dir(f"{up}.pressure") == "high" else 0.0
                conf += 0.1 if self._dir(f"{up}.temperature") == "high" else 0.0
                conf += 0.1 if (nxt and self._dir(f"{nxt}.flow") == "low") else 0.0
                plan.update(
                    status="fault_diagnosed",
                    stage=up,
                    fault="clog",
                    conf=min(conf, 0.9),
                    why=f"{k}.flow (inlet) dropped while {up}.flow stayed normal, so the restriction is inside {up}.",
                )
                confirm = [(up, m) for m in SENSORS[up] if m in ("pressure", "temperature")]
                plan["confirm"] = confirm + [(k, "flow")]
            else:
                plan.update(
                    status="fault_diagnosed",
                    stage=k,
                    fault="sensor_drift",
                    sensor=f"{k}.flow",
                    conf=0.6,
                    why=f"Only {k}.flow moved; coupled signals are normal.",
                )
                plan["confirm"] = [(k, m) for m in SENSORS[k] if m != "flow"]
            return plan

        thermal = [
            s
            for s in STAGES
            if self._dir(f"{s}.temperature") in ("high", "low")
            or self._dir(f"{s}.power") in ("high", "low")
        ]
        if thermal:
            k = thermal[0]
            first_tick = min(d["first_tick"] for d in self.devs if d["stage"] == k)
            for e in self.events:
                if (
                    e["kind"] == "setpoint_change"
                    and e["stage"] == k
                    and 0 <= first_tick - e["tick"] <= 120
                ):
                    plan.update(
                        status="no_fault",
                        stage=None,
                        fault="none",
                        conf=0.8,
                        why=f"Setpoint change at {k} at t={e['tick']} precedes the deviation "
                        "and explains it.",
                    )
                    return plan
            hot = [
                s
                for s in thermal
                if self._dir(f"{s}.temperature") == "high" or self._dir(f"{s}.power") == "high"
            ]
            if not hot:
                # Cooling with no event: a process cannot "under-heat" in this model.
                d = self.devs[0]
                plan.update(
                    status="fault_diagnosed",
                    stage=d["stage"],
                    fault="sensor_drift",
                    sensor=d["sensor"],
                    conf=0.5,
                    why=f"{d['sensor']} fell with no logged setpoint change and no flow change.",
                )
                return plan
            k = hot[0]
            downstream_hot = len(hot) > 1
            single = len(self.devs) == 1
            if k == "heater" and self._dir("heater.power") == "high":
                plan.update(
                    status="fault_diagnosed",
                    stage="heater",
                    fault="overheating",
                    conf=0.85,
                    why="heater.power and heater.temperature rose with flows unchanged.",
                )
                plan["confirm"] = [("heater", "temperature"), ("press", "temperature")]
            elif downstream_hot or not single:
                plan.update(
                    status="fault_diagnosed",
                    stage=k,
                    fault="overheating",
                    conf=0.7,
                    why=f"{k}.temperature rose first and downstream temperatures followed with flows normal.",
                )
                plan["confirm"] = [(k, "temperature")]
            else:
                plan.update(
                    status="fault_diagnosed",
                    stage=k,
                    fault="sensor_drift",
                    sensor=f"{k}.temperature" if self._dir(f"{k}.temperature") else f"{k}.power",
                    conf=0.6,
                    why="Only one temperature signal moved and nothing downstream followed.",
                )
            return plan

        if len(self.devs) == 1:
            d = self.devs[0]
            plan.update(
                status="fault_diagnosed",
                stage=d["stage"],
                fault="sensor_drift",
                sensor=d["sensor"],
                conf=0.6,
                why=f"Only {d['sensor']} deviated ({d['direction']}); coupled signals normal.",
            )
            return plan
        if not self.devs:
            plan.update(
                status="insufficient_evidence",
                stage=None,
                fault="none",
                conf=0.2,
                why="No sensor deviates at the sensitive threshold.",
            )
            return plan
        # Several deviations without a flow or temperature story.
        d = self.devs[0]
        plan.update(
            status="insufficient_evidence",
            stage=d["stage"],
            fault="none",
            conf=0.3,
            why="Deviations do not match a single propagation pattern.",
        )
        return plan

    def _refine(self) -> None:
        p = self.plan
        if p.get("fault") == "clog" and p.get("stage"):
            up = p["stage"]
            pr = self.compare.get(f"{up}.pressure", {}).get("verdict")
            tp = self.compare.get(f"{up}.temperature", {}).get("verdict")
            if pr == "elevated" or tp == "elevated":
                p["conf"] = min(0.92, p["conf"] + 0.05)
            elif pr == "depressed":
                # Pressure fell at the suspected stage: the problem is further upstream.
                i = STAGES.index(up)
                p["stage"] = STAGES[i - 1] if i > 0 else "feed"
                p["fault"] = "clog" if i > 0 else "supply_drop"
                p["conf"] = 0.5

    def _submit(self) -> Message:
        p = self.plan or {
            "status": "insufficient_evidence",
            "stage": None,
            "fault": "none",
            "conf": 0.1,
            "why": "No reasoning performed.",
        }
        evidence = [
            {
                "signal": d["sensor"],
                "observation": f"{d['direction']} from t={d['first_tick']}",
                "why_it_matters": "Part of the observed propagation pattern.",
            }
            for d in self.devs[:6]
        ]
        self.phase = 3
        diag = {
            "status": p["status"],
            "root_cause_stage": p.get("stage"),
            "fault_type": p["fault"],
            "affected_sensor": p.get("sensor"),
            "confidence": float(p["conf"]),
            "evidence": evidence,
            "alternative_hypotheses": [],
            "recommended_action": f"Inspect {p.get('stage') or 'the line'} for {p['fault']}."
            if p["status"] == "fault_diagnosed"
            else "Keep monitoring; no action yet.",
            "requires_downtime": p["fault"] in ("clog", "overheating"),
            "reasoning_summary": p["why"],
        }
        return _msg([_tool("submit_diagnosis", diag, 9)], "tool_use", 200)
