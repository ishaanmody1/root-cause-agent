"""Aggregate metrics: accuracy, calibration, abstention, cost."""

from __future__ import annotations

from typing import Any

import numpy as np

from rootcause.evals.types import ScenarioResult


def _pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def calibration(confs: list[float], correct: list[bool], bins: int = 5) -> dict[str, Any]:
    if not confs:
        return {"brier": None, "ece": None, "bins": []}
    c = np.asarray(confs)
    y = np.asarray(correct, dtype=float)
    brier = float(np.mean((c - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (c >= lo) & (c < hi) if i < bins - 1 else (c >= lo) & (c <= hi)
        if m.sum() == 0:
            continue
        acc = float(y[m].mean())
        conf = float(c[m].mean())
        ece += (m.sum() / len(c)) * abs(acc - conf)
        rows.append(
            {
                "range": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "mean_conf": round(conf, 2),
                "accuracy": round(acc, 2),
            }
        )
    return {"brier": round(brier, 3), "ece": round(float(ece), 3), "bins": rows}


def summarise_system(name: str, results: list[ScenarioResult]) -> dict[str, Any]:
    rows = [(r, r.systems[name]) for r in results if r.detector_triggered and name in r.systems]
    faults = [(r, s) for r, s in rows if r.truth.fault_type != "none"]
    normals = [(r, s) for r, s in rows if r.truth.fault_type == "none"]
    misleading = [(r, s) for r, s in faults if r.misleading]
    answered = [(r, s) for r, s in rows if not s.score.abstained]

    def acc(pairs, attr):
        return _pct(sum(getattr(s.score, attr) for _, s in pairs), len(pairs))

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r, _ in rows}):
        pairs = [(r, s) for r, s in rows if r.kind == kind]
        by_kind[kind] = {"n": len(pairs), "fully_correct_pct": acc(pairs, "fully_correct")}

    costs = [s.cost_usd for _, s in rows]
    lat = [s.latency_s for _, s in rows]
    tools = [s.tool_calls for _, s in rows]
    calls = [s.model_calls for _, s in rows]
    toks = [s.input_tokens + s.output_tokens for _, s in rows]
    cache = [s.cache_read_input_tokens for _, s in rows]
    return {
        "n": len(rows),
        "root_cause_stage_acc_pct": acc(faults, "stage_correct"),
        "fault_type_acc_pct": acc(faults, "type_correct"),
        "fully_correct_pct": acc(faults, "fully_correct"),
        "fully_correct_misleading_pct": acc(misleading, "fully_correct"),
        "n_misleading": len(misleading),
        "fully_correct_answered_pct": acc(
            [p for p in answered if p[0].truth.fault_type != "none"], "fully_correct"
        ),
        "abstention_pct": _pct(sum(s.score.abstained for _, s in rows), len(rows)),
        "confident_error_pct": _pct(sum(s.score.confident_error for _, s in rows), len(rows)),
        "false_positive_pct_on_no_fault": _pct(
            sum(s.score.false_positive for _, s in normals), len(normals)
        ),
        "n_no_fault_alarms": len(normals),
        "calibration": calibration(
            [s.prediction.confidence for _, s in rows], [s.score.fully_correct for _, s in rows]
        ),
        "errors": sum(1 for _, s in rows if s.status not in ("ok",)),
        "mean_tool_calls": round(float(np.mean(tools)), 1) if tools else None,
        "mean_model_calls": round(float(np.mean(calls)), 1) if calls else None,
        "mean_tokens": int(np.mean(toks)) if toks else None,
        "mean_cache_read_tokens": int(np.mean(cache)) if cache else None,
        "mean_cost_usd": round(float(np.mean(costs)), 4) if costs else None,
        "total_cost_usd": round(float(np.sum(costs)), 3) if costs else None,
        "p50_latency_s": round(float(np.percentile(lat, 50)), 1) if lat else None,
        "p90_latency_s": round(float(np.percentile(lat, 90)), 1) if lat else None,
        "by_kind": by_kind,
    }


def summarise_detector(results: list[ScenarioResult]) -> dict[str, Any]:
    normal = [r for r in results if r.kind == "normal"]
    benign = [r for r in results if r.kind == "normal_setpoint_change"]
    faults = [r for r in results if r.truth.fault_type != "none"]
    delays = [r.detection_delay for r in faults if r.detection_delay is not None]
    return {
        "n_normal": len(normal),
        "false_alarm_pct_on_normal": _pct(sum(r.detector_triggered for r in normal), len(normal)),
        "n_benign_event": len(benign),
        "alarm_pct_on_benign_event": _pct(sum(r.detector_triggered for r in benign), len(benign)),
        "n_fault": len(faults),
        "detection_rate_pct": _pct(sum(r.detector_triggered for r in faults), len(faults)),
        "median_detection_delay_ticks": int(np.median(delays)) if delays else None,
        "misleading_pct_of_detected_faults": _pct(
            sum(bool(r.misleading) for r in faults if r.detector_triggered),
            sum(r.detector_triggered for r in faults),
        ),
    }
