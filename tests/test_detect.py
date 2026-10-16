import numpy as np
import pandas as pd

from rootcause.detect import DetectorConfig, detect
from rootcause.sim import make_scenario, simulate


def test_no_false_alarms_on_normal():
    for seed in range(20):
        det = detect(simulate(make_scenario("normal", seed)).data)
        assert not det.triggered, f"seed {seed}: {det.first_alarm}"


def test_detects_every_fault_kind():
    for kind in ["clog", "overheating", "supply_drop", "sensor_drift", "sensor_failure"]:
        for seed in range(5):
            sc = simulate(make_scenario(kind, seed))
            det = detect(sc.data)
            assert det.triggered, f"{kind} seed {seed}"
            assert det.alarm_tick >= sc.truth.onset_tick


def test_misleading_cases_exist(clog_scenario):
    sc, det = clog_scenario
    assert sc.truth.root_cause_stage == "heater"
    assert det.first_alarm.stage != "heater"


def test_missing_flatline_zero_kinds():
    base = simulate(make_scenario("normal", 3)).data.copy()
    base.loc[300:, "feed.flow"] = np.nan
    base.loc[300:, "press.pressure"] = 0.0
    base.loc[300:, "heater.temperature"] = base.loc[299, "heater.temperature"]
    det = detect(base)
    kinds = {a.sensor: a.kind for a in det.alarms}
    assert kinds["feed.flow"] == "missing"
    assert kinds["press.pressure"] == "stuck_zero"
    assert kinds["heater.temperature"] == "flatline"


def test_baseline_only_uses_baseline_window():
    df = pd.DataFrame(
        {"t": range(400), "feed.flow": np.r_[np.full(200, 100.0), np.full(200, 50.0)]}
    )
    df["feed.flow"] += np.random.default_rng(0).normal(0, 1, 400)
    det = detect(df, DetectorConfig(baseline_ticks=200))
    assert det.triggered and det.first_alarm.kind == "low"
    assert abs(det.baselines["feed.flow"].mean - 100) < 1
