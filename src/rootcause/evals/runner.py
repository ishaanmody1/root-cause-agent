"""Eval runner: generate scenarios, detect, diagnose with each system, score, report."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.console import Console

from rootcause.agent.fake import FakeRuleBasedBackend
from rootcause.agent.loop import AgentConfig, AnthropicBackend, DiagnosisAgent, DiagnosisRun
from rootcause.agent.tools import ToolContext
from rootcause.detect.detector import DetectorConfig, detect
from rootcause.evals.baselines import naive_prediction
from rootcause.evals.metrics import summarise_detector, summarise_system
from rootcause.evals.report import render_markdown, render_scenario_table
from rootcause.evals.types import EvalSummary, Prediction, ScenarioResult, SystemResult, score
from rootcause.sim.scenarios import SCENARIO_KINDS, generate_suite
from rootcause.sim.simulator import Scenario, simulate
from rootcause.ui.dashboard import render_dashboard

Backend = Literal["none", "anthropic"]


class EvalConfig(BaseModel):
    n_scenarios: int = 42
    seed: int = 0
    kinds: list[str] = Field(default_factory=lambda: list(SCENARIO_KINDS))
    llm_backend: Backend = "none"  # "anthropic" adds the LLM agent column
    agent: AgentConfig = Field(default_factory=AgentConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    concurrency: int = 4
    out_dir: Path = Path("runs")
    write_dashboards: bool = True


def _system_result(
    run: DiagnosisRun, truth, run_path: Path | None, dash: Path | None
) -> SystemResult:
    pred = Prediction.from_diagnosis(run.diagnosis)  # type: ignore[arg-type]
    return SystemResult(
        prediction=pred,
        score=score(pred, truth),
        status=run.status,
        model_calls=run.usage.model_calls,
        tool_calls=run.usage.tool_calls,
        input_tokens=run.usage.input_tokens,
        output_tokens=run.usage.output_tokens,
        cache_read_input_tokens=run.usage.cache_read_input_tokens,
        cost_usd=run.usage.cost_usd,
        latency_s=run.usage.latency_s,
        error=run.error,
        run_path=str(run_path) if run_path else None,
        dashboard_path=str(dash) if dash else None,
    )


def evaluate_scenario(
    sc: Scenario, cfg: EvalConfig, run_dir: Path, console: Console | None = None
) -> ScenarioResult:
    det = detect(sc.data, cfg.detector)
    truth = sc.truth
    first = det.first_alarm
    onset = (
        truth.onset_tick
        if truth.onset_tick is not None
        else (
            sc.spec.events[0].tick
            if sc.spec.events and sc.spec.events[0].kind == "setpoint_change"
            else None
        )
    )
    misleading = None
    if det.triggered and first is not None and truth.root_cause_stage is not None:
        misleading = first.stage != truth.root_cause_stage
    truth.misleading = misleading
    res = ScenarioResult(
        scenario_id=sc.scenario_id,
        kind=sc.spec.tags[0],
        truth=truth,
        detector_triggered=det.triggered,
        first_alarm_sensor=first.sensor if first else None,
        first_alarm_stage=first.stage if first else None,
        alarm_tick=det.alarm_tick,
        detection_delay=(det.alarm_tick - onset)
        if (det.alarm_tick is not None and onset is not None)
        else None,
        misleading=misleading,
        t_now=None,
    )
    if not det.triggered:
        return res

    ctx = ToolContext.from_scenario(sc, det, grace_ticks=cfg.agent.grace_ticks)
    res.t_now = ctx.t_now

    # Naive baseline.
    naive = naive_prediction(det)
    res.systems["naive"] = SystemResult(prediction=naive, score=score(naive, truth))

    # Rule-based tracer through the real loop.
    rule_cfg = cfg.agent.model_copy(update={"model": "fake-rule-based"})
    rule_run = DiagnosisAgent(FakeRuleBasedBackend(), rule_cfg).run(
        ToolContext.from_scenario(sc, det, grace_ticks=cfg.agent.grace_ticks), sc.scenario_id
    )
    rule_path = run_dir / "rule_tracer" / f"{sc.scenario_id}.json"
    rule_run.write(rule_path)
    res.systems["rule_tracer"] = _system_result(rule_run, truth, rule_path, None)

    # LLM agent.
    if cfg.llm_backend == "anthropic":
        backend = AnthropicBackend(cfg.agent.model, cfg.agent.effort)
        run = DiagnosisAgent(backend, cfg.agent).run(ctx, sc.scenario_id)
        name = f"agent:{cfg.agent.model}"
        run_path = run_dir / "agent" / f"{sc.scenario_id}.json"
        run.write(run_path)
        run.write_trace_jsonl(run_dir / "agent" / f"{sc.scenario_id}.trace.jsonl")
        dash = None
        if cfg.write_dashboards:
            dash = run_dir / "dashboards" / f"{sc.scenario_id}.html"
            render_dashboard(sc, det, run, dash)
        res.systems[name] = _system_result(run, truth, run_path, dash)
    elif cfg.write_dashboards:
        dash = run_dir / "dashboards" / f"{sc.scenario_id}.html"
        render_dashboard(sc, det, rule_run, dash)
        res.systems["rule_tracer"].dashboard_path = str(dash)

    if console:
        parts = []
        for n, s in res.systems.items():
            mark = "✅" if s.score.fully_correct else ("⚠️" if s.score.abstained else "❌")
            parts.append(f"{n}={mark}")
        console.print(
            f"[dim]{sc.scenario_id:30s}[/] {truth.fault_type}@{truth.root_cause_stage or '-':7s} alarm={first.sensor if first else '-':18s} {' '.join(parts)}"
        )
    return res


def run_eval(
    cfg: EvalConfig, console: Console | None = None
) -> tuple[EvalSummary, list[ScenarioResult], Path]:
    console = console or Console()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = cfg.agent.model if cfg.llm_backend == "anthropic" else "offline"
    run_dir = cfg.out_dir / f"{stamp}-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = generate_suite(cfg.n_scenarios, cfg.seed, cfg.kinds)
    scenarios = [simulate(s) for s in specs]
    console.print(f"[bold]Evaluating {len(scenarios)} scenarios -> {run_dir}[/]")

    workers = cfg.concurrency if cfg.llm_backend == "anthropic" else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(
                ex.map(lambda sc: evaluate_scenario(sc, cfg, run_dir, console), scenarios)
            )
    else:
        results = [evaluate_scenario(sc, cfg, run_dir, console) for sc in scenarios]

    names: list[str] = []
    for r in results:
        for n in r.systems:
            if n not in names:
                names.append(n)
    summary = EvalSummary(
        n_scenarios=len(results),
        n_triggered=sum(r.detector_triggered for r in results),
        detector=summarise_detector(results),
        systems={n: summarise_system(n, results) for n in names},
        config=json.loads(cfg.model_dump_json()),
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    (run_dir / "results.json").write_text(
        json.dumps([json.loads(r.model_dump_json()) for r in results], indent=2)
    )
    md = (
        render_markdown(summary, results)
        + "\n\n### Per-scenario\n\n"
        + render_scenario_table(results)
    )
    (run_dir / "REPORT.md").write_text(md)
    return summary, results, run_dir


def load_results(run_dir: Path) -> tuple[EvalSummary, list[ScenarioResult]]:
    summary = EvalSummary.model_validate_json((run_dir / "summary.json").read_text())
    raw: list[dict[str, Any]] = json.loads((run_dir / "results.json").read_text())
    return summary, [ScenarioResult.model_validate(r) for r in raw]
