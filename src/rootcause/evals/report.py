"""Markdown rendering of eval results."""

from __future__ import annotations

from rootcause.evals.types import EvalSummary, ScenarioResult

SYSTEM_LABELS = {
    "naive": "Naive (blame first alarm)",
    "rule_tracer": "Rule-based tracer",
}


def label(name: str) -> str:
    if name in SYSTEM_LABELS:
        return SYSTEM_LABELS[name]
    if name.startswith("agent:"):
        return f"LLM agent ({name.split(':', 1)[1]})"
    return name


def _f(v, suffix: str = "") -> str:
    return "–" if v is None else f"{v}{suffix}"


def render_markdown(summary: EvalSummary, results: list[ScenarioResult]) -> str:
    names = list(summary.systems.keys())
    lines: list[str] = []
    lines.append(
        f"## Eval results ({summary.n_scenarios} scenarios, {summary.n_triggered} alarms)\n"
    )
    d = summary.detector
    lines.append("### Detector\n")
    lines.append("| Metric | Value |\n|---|---|")
    lines.append(
        f"| False alarms on normal runs | {_f(d['false_alarm_pct_on_normal'], '%')} of {d['n_normal']} |"
    )
    lines.append(
        f"| Alarms on benign setpoint changes | {_f(d['alarm_pct_on_benign_event'], '%')} of {d['n_benign_event']} |"
    )
    lines.append(f"| Fault detection rate | {_f(d['detection_rate_pct'], '%')} of {d['n_fault']} |")
    lines.append(f"| Median detection delay | {_f(d['median_detection_delay_ticks'], ' ticks')} |")
    lines.append(
        f"| First alarm at the wrong stage (misleading) | {_f(d['misleading_pct_of_detected_faults'], '%')} |"
    )
    lines.append("")
    lines.append("### Diagnosis (on scenarios where the detector fired)\n")
    header = "| Metric | " + " | ".join(label(n) for n in names) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(names))
    rows = [
        ("Root-cause stage accuracy", "root_cause_stage_acc_pct", "%"),
        ("Fault-type accuracy", "fault_type_acc_pct", "%"),
        ("Both correct", "fully_correct_pct", "%"),
        ("Both correct, misleading cases only", "fully_correct_misleading_pct", "%"),
        ("Both correct when answered (excl. abstentions)", "fully_correct_answered_pct", "%"),
        ("Abstained (insufficient evidence)", "abstention_pct", "%"),
        ("Confident errors (wrong at conf >= 0.75)", "confident_error_pct", "%"),
        ("False positives on no-fault alarms", "false_positive_pct_on_no_fault", "%"),
        ("Brier score (lower is better)", ("calibration", "brier"), ""),
        ("Expected calibration error", ("calibration", "ece"), ""),
        ("Mean tool calls", "mean_tool_calls", ""),
        ("Mean model calls", "mean_model_calls", ""),
        ("Mean tokens (in+out)", "mean_tokens", ""),
        ("Mean cost per diagnosis", "mean_cost_usd", " USD"),
        ("p50 / p90 latency", None, ""),
    ]
    for title, key, suffix in rows:
        cells = []
        for n in names:
            s = summary.systems[n]
            if key is None:
                cells.append(f"{_f(s['p50_latency_s'])} / {_f(s['p90_latency_s'])} s")
            elif isinstance(key, tuple):
                cells.append(_f(s[key[0]][key[1]], suffix))
            else:
                cells.append(_f(s[key], suffix))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### Both-correct rate by scenario kind\n")
    kinds = sorted({r.kind for r in results if r.detector_triggered})
    lines.append("| Kind | n | " + " | ".join(label(n) for n in names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for k in kinds:
        n_k = summary.systems[names[0]]["by_kind"].get(k, {}).get("n", 0)
        cells = [
            _f(summary.systems[n]["by_kind"].get(k, {}).get("fully_correct_pct"), "%")
            for n in names
        ]
        lines.append(f"| {k} | {n_k} | " + " | ".join(cells) + " |")
    lines.append("")
    for n in names:
        cal = summary.systems[n]["calibration"]
        if cal["bins"]:
            lines.append(f"### Calibration: {label(n)}\n")
            lines.append("| Confidence | n | Mean conf | Accuracy |\n|---|---|---|---|")
            for b in cal["bins"]:
                lines.append(f"| {b['range']} | {b['n']} | {b['mean_conf']} | {b['accuracy']} |")
            lines.append("")
    return "\n".join(lines)


def render_scenario_table(results: list[ScenarioResult]) -> str:
    names = sorted({n for r in results for n in r.systems})
    lines = ["| Scenario | Truth | First alarm | " + " | ".join(label(n) for n in names) + " |"]
    lines.append("|---|---|---|" + "---|" * len(names))
    for r in results:
        if not r.detector_triggered:
            continue
        truth = f"{r.truth.fault_type}@{r.truth.root_cause_stage or '-'}"
        cells = []
        for n in names:
            s = r.systems.get(n)
            if not s:
                cells.append("–")
                continue
            p = s.prediction
            mark = "✅" if s.score.fully_correct else ("⚠️" if s.score.abstained else "❌")
            what = (
                p.fault_type + "@" + (p.root_cause_stage or "-")
                if p.status == "fault_diagnosed"
                else p.status
            )
            cells.append(f"{mark} {what} ({p.confidence:.2f})")
        flag = " ⚡" if r.misleading else ""
        lines.append(
            f"| {r.scenario_id} | {truth} | {r.first_alarm_sensor}{flag} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)
