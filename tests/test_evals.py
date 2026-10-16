from pathlib import Path

from rootcause.evals import EvalConfig, run_eval
from rootcause.evals.types import Prediction, score
from rootcause.sim.faults import GroundTruth


def test_scoring_rules():
    truth = GroundTruth(fault_type="clog", root_cause_stage="feed", description="")
    good = Prediction(
        status="fault_diagnosed", root_cause_stage="feed", fault_type="clog", confidence=0.9
    )
    assert score(good, truth).fully_correct
    wrong = Prediction(
        status="fault_diagnosed", root_cause_stage="press", fault_type="clog", confidence=0.9
    )
    s = score(wrong, truth)
    assert s.type_correct and not s.stage_correct and s.confident_error
    abst = Prediction(
        status="insufficient_evidence", root_cause_stage=None, fault_type="none", confidence=0.2
    )
    assert score(abst, truth).abstained and not score(abst, truth).confident_error
    none = GroundTruth(fault_type="none", root_cause_stage=None, description="")
    fp = Prediction(
        status="fault_diagnosed",
        root_cause_stage="heater",
        fault_type="overheating",
        confidence=0.8,
    )
    assert score(fp, none).false_positive
    ok = Prediction(status="no_fault", root_cause_stage=None, fault_type="none", confidence=0.8)
    assert score(ok, none).fully_correct


def test_offline_eval_runs_end_to_end(tmp_path: Path):
    cfg = EvalConfig(n_scenarios=7, seed=1, out_dir=tmp_path, write_dashboards=True)
    summary, results, run_dir = run_eval(cfg)
    assert summary.n_scenarios == 7
    assert set(summary.systems) == {"naive", "rule_tracer"}
    assert (run_dir / "REPORT.md").exists() and (run_dir / "summary.json").exists()
    dashboards = list((run_dir / "dashboards").glob("*.html"))
    assert len(dashboards) == summary.n_triggered
    assert summary.systems["rule_tracer"]["fully_correct_pct"] is not None
