"""Run detection + diagnosis on real data: natural alarms and injected sensor faults."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from rich.console import Console

from rootcause.agent.loop import AgentConfig, AnthropicBackend, DiagnosisAgent, DiagnosisRun
from rootcause.agent.tools import ToolContext
from rootcause.detect.detector import Detection, DetectorConfig, detect
from rootcause.evals.types import Prediction, Score, score
from rootcause.realdata.adapter import DEFAULT_CSV, load_real
from rootcause.realdata.inject import RealScenario, inject_sensor_fault
from rootcause.sim.faults import GroundTruth
from rootcause.ui.dashboard import render_dashboard

REAL_DETECTOR = DetectorConfig(baseline_ticks=180, z_threshold=4.0, persistence=5, ewma_alpha=0.3)


def naive_real(det: Detection) -> Prediction:
    """Generic naive baseline: blame the first alarm; sensor-looking kinds -> sensor_failure."""
    if not det.triggered or det.first_alarm is None:
        return Prediction(
            status="no_fault", root_cause_stage=None, fault_type="none", confidence=0.7
        )
    a = det.first_alarm
    fault = "sensor_failure" if a.kind in ("missing", "flatline", "stuck_zero") else "sensor_drift"
    return Prediction(
        status="fault_diagnosed",
        root_cause_stage=a.stage,
        fault_type=fault,  # type: ignore[arg-type]
        affected_sensor=a.sensor if fault != "none" else None,
        confidence=0.7,
    )


class RealResult(BaseModel):
    scenario_id: str
    truth: GroundTruth | None
    detector_triggered: bool
    first_alarm: str | None
    alarm_tick: int | None
    t_now: int | None
    naive: Prediction | None = None
    naive_score: Score | None = None
    agent: Prediction | None = None
    agent_score: Score | None = None
    agent_status: str | None = None
    agent_cost_usd: float | None = None
    agent_tool_calls: int | None = None
    dashboard: str | None = None
    run_path: str | None = None
    agent_summary: str | None = None


def diagnose_real(
    sc: RealScenario,
    cfg: AgentConfig,
    out_dir: Path,
    backend: str = "anthropic",
    detector: DetectorConfig = REAL_DETECTOR,
    alarm_index: int = 0,
    console: Console | None = None,
) -> RealResult:
    det = detect(sc.data, detector)
    res = RealResult(
        scenario_id=sc.scenario_id,
        truth=sc.truth if sc.fault else None,
        detector_triggered=det.triggered,
        first_alarm=det.first_alarm.describe() if det.first_alarm else None,
        alarm_tick=det.alarm_tick,
        t_now=None,
    )
    if not det.triggered:
        return res
    if alarm_index > 0 and alarm_index < len(det.alarms):
        # Investigate a later alarm episode: pretend earlier alarms were acknowledged.
        chosen = sorted(det.alarms, key=lambda a: a.alarm_tick)[alarm_index]
        det = det.model_copy(update={"first_alarm": chosen, "alarm_tick": chosen.alarm_tick})
    ctx = ToolContext(
        data=sc.data,
        detection=det,
        events=[],
        t_now=min(det.alarm_tick + cfg.grace_ticks, int(sc.data["t"].iloc[-1])),  # type: ignore[operator]
        topology=sc.topology,
        baseline_ticks=sc.baseline_ticks,
    )
    res.t_now = ctx.t_now
    res.naive = naive_real(det)
    if sc.fault:
        res.naive_score = score(res.naive, sc.truth)
    if backend != "anthropic":
        return res
    run: DiagnosisRun = DiagnosisAgent(AnthropicBackend(cfg.model, cfg.effort), cfg).run(
        ctx, sc.scenario_id
    )
    run_path = out_dir / f"{sc.scenario_id}.json"
    run.write(run_path)
    run.write_trace_jsonl(out_dir / f"{sc.scenario_id}.trace.jsonl")
    dash = out_dir / f"{sc.scenario_id}.html"
    render_dashboard(sc, det, run, dash)
    assert run.diagnosis is not None
    res.agent = Prediction.from_diagnosis(run.diagnosis)
    res.agent_status = run.status
    res.agent_cost_usd = run.usage.cost_usd
    res.agent_tool_calls = run.usage.tool_calls
    res.dashboard = str(dash)
    res.run_path = str(run_path)
    res.agent_summary = run.diagnosis.reasoning_summary
    if sc.fault:
        res.agent_score = score(res.agent, sc.truth)
    if console:
        mark = ""
        if res.agent_score:
            mark = (
                " ✅"
                if res.agent_score.fully_correct
                else (" ⚠️" if res.agent_score.abstained else " ❌")
            )
        console.print(
            f"[dim]{sc.scenario_id}[/] -> {run.diagnosis.one_line()}{mark}  ${run.usage.cost_usd:.3f}"
        )
    return res


def natural_scenario(csv: Path = DEFAULT_CSV, tick_seconds: float = 10.0) -> RealScenario:
    df, topo = load_real(csv, tick_seconds)
    return RealScenario(
        scenario_id="real-natural",
        data=df,
        topology=topo,
        truth=GroundTruth(
            fault_type="none",
            root_cause_stage=None,
            description="Unlabelled real run; no ground truth.",
        ),
        baseline_ticks=REAL_DETECTOR.baseline_ticks,
    )


def injected_suite(
    n: int, seed: int, csv: Path = DEFAULT_CSV, tick_seconds: float = 10.0
) -> list[RealScenario]:
    df, topo = load_real(csv, tick_seconds)
    rng = np.random.default_rng(seed)
    kinds = ["sensor_drift", "sensor_failure"]
    return [
        inject_sensor_fault(df, topo, rng, REAL_DETECTOR.baseline_ticks, kind=kinds[i % 2])
        for i in range(n)
    ]


def summarise(results: list[RealResult]) -> dict[str, Any]:
    labelled = [r for r in results if r.truth is not None and r.detector_triggered]

    def acc(getter) -> float | None:
        vals = [getter(r) for r in labelled]
        vals = [v for v in vals if v is not None]
        return round(100 * sum(vals) / len(vals), 1) if vals else None

    return {
        "n": len(results),
        "n_labelled_alarms": len(labelled),
        "naive_sensor_correct_pct": acc(
            lambda r: (
                r.naive_score.stage_correct
                and r.naive.affected_sensor == f"{r.truth.root_cause_stage}.{r.truth.sensor}"
                if r.naive_score
                else None
            )
        ),
        "naive_fully_correct_pct": acc(
            lambda r: r.naive_score.fully_correct if r.naive_score else None
        ),
        "agent_fully_correct_pct": acc(
            lambda r: r.agent_score.fully_correct if r.agent_score else None
        ),
        "agent_sensor_correct_pct": acc(
            lambda r: (
                (
                    r.agent_score.stage_correct
                    and r.agent.affected_sensor == f"{r.truth.root_cause_stage}.{r.truth.sensor}"
                )
                if r.agent_score
                else None
            )
        ),
        "agent_abstained_pct": acc(lambda r: r.agent_score.abstained if r.agent_score else None),
        "agent_confident_error_pct": acc(
            lambda r: r.agent_score.confident_error if r.agent_score else None
        ),
        "agent_mean_cost_usd": round(
            float(np.mean([r.agent_cost_usd for r in results if r.agent_cost_usd is not None])), 4
        )
        if any(r.agent_cost_usd is not None for r in results)
        else None,
    }


def write_results(results: list[RealResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "real_results.json"
    p.write_text(
        json.dumps(
            {
                "summary": summarise(results),
                "results": [json.loads(r.model_dump_json()) for r in results],
            },
            indent=2,
        )
    )
    return p


def new_run_dir(root: Path, tag: str) -> Path:
    d = root / f"{time.strftime('%Y%m%d-%H%M%S')}-real-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d
