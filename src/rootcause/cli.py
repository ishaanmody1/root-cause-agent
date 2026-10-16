"""Command-line entry point: simulate, diagnose, eval, dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rootcause.agent.fake import FakeRuleBasedBackend
from rootcause.agent.loop import AgentConfig, AnthropicBackend, DiagnosisAgent, DiagnosisRun
from rootcause.agent.tools import ToolContext
from rootcause.detect.detector import detect
from rootcause.sim.io import load_scenario, save_scenario
from rootcause.sim.scenarios import SCENARIO_KINDS, generate_suite, make_scenario
from rootcause.sim.simulator import Scenario, simulate
from rootcause.ui.dashboard import render_dashboard, render_eval_index

load_dotenv()
app = typer.Typer(
    help="Root-cause diagnosis agent for a multi-stage production line.", no_args_is_help=True
)
console = Console()


def _scenario(kind: str | None, seed: int, path: Path | None) -> Scenario:
    if path is not None:
        return load_scenario(path)
    if kind is None:
        raise typer.BadParameter("pass --kind or --path")
    if kind not in SCENARIO_KINDS:
        raise typer.BadParameter(f"kind must be one of {SCENARIO_KINDS}")
    return simulate(make_scenario(kind, seed))


@app.command()
def simulate_cmd(
    kind: Annotated[str, typer.Option("--kind", "-k", help=f"One of {SCENARIO_KINDS}")] = "clog",
    seed: Annotated[int, typer.Option("--seed", "-s")] = 7,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("data/scenarios"),
) -> None:
    """Generate one scenario (CSV + JSON) with ground truth."""
    sc = simulate(make_scenario(kind, seed))
    d = save_scenario(sc, out)
    det = detect(sc.data)
    console.print(f"[green]Saved[/] {d}")
    console.print(f"Truth: {sc.truth.description}")
    console.print(f"Detector: {det.first_alarm.describe() if det.first_alarm else 'no alarm'}")


@app.command()
def suite(
    n: Annotated[int, typer.Option("--n", "-n")] = 42,
    seed: Annotated[int, typer.Option("--seed", "-s")] = 0,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("data/scenarios"),
) -> None:
    """Generate the balanced scenario suite to disk."""
    for spec in generate_suite(n, seed):
        save_scenario(simulate(spec), out)
    console.print(f"[green]Wrote {n} scenarios to {out}[/]")


def _print_run(sc: Scenario, run: DiagnosisRun, reveal: bool) -> None:
    d = run.diagnosis
    assert d is not None
    color = {"fault_diagnosed": "red", "no_fault": "green", "insufficient_evidence": "yellow"}[
        d.status
    ]
    body = [
        f"[bold {color}]{d.status}[/]  root cause: [bold]{d.root_cause_stage or '-'}[/]  "
        f"type: [bold]{d.fault_type}[/]"
        + (f"  sensor: {d.affected_sensor}" if d.affected_sensor else ""),
        f"confidence: {d.confidence:.2f}   downtime: {'yes' if d.requires_downtime else 'no'}",
        "",
        f"[bold]Reasoning:[/] {d.reasoning_summary}",
        "",
        "[bold]Evidence chain:[/]",
    ]
    for i, e in enumerate(d.evidence, 1):
        body.append(f"  {i}. [cyan]{e.signal}[/]: {e.observation}  [dim]{e.why_it_matters}[/]")
    if d.alternative_hypotheses:
        body.append("")
        body.append("[bold]Alternatives considered:[/]")
        for h in d.alternative_hypotheses:
            body.append(
                f"  - {h.fault_type}@{h.root_cause_stage or '-'} ({h.confidence:.2f}): {h.why_less_likely}"
            )
    body.append("")
    body.append(f"[bold]Recommended action:[/] {d.recommended_action}")
    console.print(Panel("\n".join(body), title=f"Diagnosis: {sc.scenario_id}", expand=False))
    u = run.usage
    console.print(
        f"[dim]status={run.status} model={run.model} model_calls={u.model_calls} tool_calls={u.tool_calls} "
        f"tokens_in={u.input_tokens} (cache_read={u.cache_read_input_tokens}) tokens_out={u.output_tokens} "
        f"cost=${u.cost_usd:.4f} latency={u.latency_s}s[/]"
    )
    if run.error:
        console.print(f"[red]error:[/] {run.error}")
    if reveal:
        ok = d.root_cause_stage == sc.truth.root_cause_stage and d.fault_type == sc.truth.fault_type
        console.print(
            f"[bold]Ground truth:[/] {sc.truth.description}  -> {'[green]CORRECT[/]' if ok else '[red]WRONG[/]'}"
        )


@app.command()
def diagnose(
    kind: Annotated[str | None, typer.Option("--kind", "-k")] = "clog",
    seed: Annotated[int, typer.Option("--seed", "-s")] = 7,
    path: Annotated[Path | None, typer.Option("--path", "-p", help="Saved scenario dir")] = None,
    backend: Annotated[str, typer.Option("--backend", "-b", help="anthropic | rule")] = "anthropic",
    model: Annotated[
        str, typer.Option("--model", "-m", envvar="ROOTCAUSE_MODEL")
    ] = "claude-sonnet-5-5",
    effort: Annotated[str, typer.Option("--effort")] = "medium",
    max_steps: Annotated[int, typer.Option("--max-steps")] = 12,
    max_cost: Annotated[float, typer.Option("--max-cost")] = 0.5,
    grace: Annotated[
        int, typer.Option("--grace", help="Ticks after the alarm before diagnosis")
    ] = 40,
    html: Annotated[Path | None, typer.Option("--html", help="Write dashboard HTML here")] = None,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("runs/diagnose"),
    reveal: Annotated[bool, typer.Option("--reveal/--no-reveal", help="Show ground truth")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run detection + the diagnosis agent on one scenario."""
    sc = _scenario(kind, seed, path)
    det = detect(sc.data)
    if not det.triggered:
        console.print("[yellow]Detector did not trigger; nothing to diagnose.[/]")
        if reveal:
            console.print(f"Ground truth: {sc.truth.description}")
        raise typer.Exit(0)
    console.print(
        Panel(
            ToolContext.from_scenario(sc, det, grace).detection_summary(),
            title="Detector",
            expand=False,
        )
    )
    cfg = AgentConfig(
        model=model, effort=effort, max_steps=max_steps, max_cost_usd=max_cost, grace_ticks=grace
    )  # type: ignore[arg-type]
    if backend == "rule":
        agent = DiagnosisAgent(
            FakeRuleBasedBackend(), cfg.model_copy(update={"model": "fake-rule-based"})
        )
    else:
        agent = DiagnosisAgent(AnthropicBackend(model, cfg.effort), cfg)
    ctx = ToolContext.from_scenario(sc, det, grace)
    with console.status("Diagnosing..."):
        run = agent.run(ctx, sc.scenario_id)
    if verbose:
        for s in run.trace:
            if s.kind == "model_call":
                console.print(
                    f"[dim]step {s.step} model_call stop={s.stop_reason} {s.latency_s}s[/]"
                    + (f"\n  {s.text}" if s.text else "")
                )
            elif s.kind == "tool_call":
                console.print(
                    f"[dim]step {s.step}[/] [cyan]{s.tool_name}[/]({json.dumps(s.tool_input)}){' [red]ERROR[/]' if s.is_error else ''}"
                )
            else:
                console.print(f"[dim]step {s.step} {s.kind}: {s.text}[/]")
    _print_run(sc, run, reveal)
    run_path = out / f"{sc.scenario_id}.json"
    run.write(run_path)
    run.write_trace_jsonl(out / f"{sc.scenario_id}.trace.jsonl")
    html = html or out / f"{sc.scenario_id}.html"
    render_dashboard(sc, det, run, html)
    console.print(f"[dim]run: {run_path}   dashboard: {html}[/]")


@app.command("eval")
def eval_cmd(
    n: Annotated[int, typer.Option("--n", "-n")] = 42,
    seed: Annotated[int, typer.Option("--seed", "-s")] = 0,
    backend: Annotated[
        str, typer.Option("--backend", "-b", help="anthropic | offline")
    ] = "offline",
    model: Annotated[
        str, typer.Option("--model", "-m", envvar="ROOTCAUSE_MODEL")
    ] = "claude-sonnet-5-5",
    effort: Annotated[str, typer.Option("--effort")] = "medium",
    max_cost: Annotated[float, typer.Option("--max-cost", help="Per-diagnosis budget, USD")] = 0.5,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 4,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("runs"),
    dashboards: Annotated[bool, typer.Option("--dashboards/--no-dashboards")] = True,
) -> None:
    """Run the scenario suite against the naive baseline, the rule tracer, and (optionally) the LLM agent."""
    from rootcause.evals.runner import EvalConfig, run_eval

    cfg = EvalConfig(
        n_scenarios=n,
        seed=seed,
        llm_backend="anthropic" if backend == "anthropic" else "none",
        agent=AgentConfig(model=model, effort=effort, max_cost_usd=max_cost),  # type: ignore[arg-type]
        concurrency=concurrency,
        out_dir=out,
        write_dashboards=dashboards,
    )
    summary, results, run_dir = run_eval(cfg, console)
    render_eval_index(summary, results, run_dir / "index.html")
    table = Table(title="Diagnosis accuracy (alarm-triggered scenarios)")
    table.add_column("System")
    for col in [
        "stage acc",
        "type acc",
        "both",
        "both (misleading)",
        "abstain",
        "confident err",
        "FP no-fault",
        "Brier",
        "cost/diag",
    ]:
        table.add_column(col, justify="right")
    for name, s in summary.systems.items():
        table.add_row(
            name,
            f"{s['root_cause_stage_acc_pct']}%",
            f"{s['fault_type_acc_pct']}%",
            f"{s['fully_correct_pct']}%",
            f"{s['fully_correct_misleading_pct']}% (n={s['n_misleading']})",
            f"{s['abstention_pct']}%",
            f"{s['confident_error_pct']}%",
            f"{s['false_positive_pct_on_no_fault']}%",
            f"{s['calibration']['brier']}",
            f"${s['mean_cost_usd']}",
        )
    console.print(table)
    console.print(f"Detector: {summary.detector}")
    console.print(
        f"[green]Report:[/] {run_dir / 'REPORT.md'}   [green]Index:[/] {run_dir / 'index.html'}"
    )


@app.command()
def dashboard(
    run_dir: Annotated[Path, typer.Argument(help="Eval run directory containing results.json")],
) -> None:
    """Rebuild the HTML index for a saved eval run."""
    from rootcause.evals.runner import load_results

    summary, results = load_results(run_dir)
    p = render_eval_index(summary, results, run_dir / "index.html")
    console.print(f"[green]Wrote[/] {p}")


if __name__ == "__main__":
    app()
