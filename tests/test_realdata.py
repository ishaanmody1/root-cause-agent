"""Adapter tests use a tiny synthetic file in the Kaggle column format (the real CSV is not committed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rootcause.agent.tools import ToolContext, build_tool_specs
from rootcause.detect.detector import DetectorConfig, detect
from rootcause.realdata.adapter import column_map, load_real
from rootcause.realdata.inject import inject_sensor_fault

RAW_COLS = [
    "time_stamp",
    "AmbientConditions.AmbientHumidity.U.Actual",
    "Machine1.RawMaterial.Property1",
    "Machine1.Zone1Temperature.C.Actual",
    "Machine1.MaterialPressure.U.Actual",
    "Machine2.MaterialPressure.U.Actual",
    "Machine3.MaterialPressure.U.Actual",
    "FirstStage.CombinerOperation.Temperature1.U.Actual",
    "Stage1.Output.Measurement0.U.Actual",
    "Stage1.Output.Measurement0.U.Setpoint",
    "Machine4.Pressure.C.Actual",
    "Machine5.ExitTemperature.U.Actual",
    "Stage2.Output.Measurement0.U.Actual",
]


@pytest.fixture
def fake_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 6000  # seconds -> 600 ticks at 10 s
    ts = pd.date_range("2019-03-06 10:00:00", periods=n, freq="s")
    df = pd.DataFrame({"time_stamp": ts.strftime("%Y-%m-%d %H:%M:%S")})
    for c in RAW_COLS[1:]:
        if "Property" in c or "Zone1Temperature" in c or "Setpoint" in c:
            df[c] = 72.0  # exactly constant, like the real controlled / lab channels
        else:
            df[c] = 100 + rng.normal(0, 1, n)
    p = tmp_path / "continuous_factory_process.csv"
    df.to_csv(p, index=False)
    return p


def test_column_map_and_topology(fake_csv: Path):
    cmap = column_map(RAW_COLS)
    assert cmap["Machine1.MaterialPressure.U.Actual"] == "Machine1.MaterialPressure"
    assert cmap["FirstStage.CombinerOperation.Temperature1.U.Actual"] == "Combiner.Temperature1"
    assert cmap["Stage2.Output.Measurement0.U.Actual"] == "Stage2Output.Measurement0"
    assert "Stage1.Output.Measurement0.U.Setpoint" not in cmap
    df, topo = load_real(fake_csv, tick_seconds=10)
    assert len(df) == 600
    assert topo.upstream["Combiner"] == ["Machine1", "Machine2", "Machine3"]
    assert topo.downstream("Machine4") == ["Machine5"]
    # Constant columns are dropped, so the agent is not offered dead channels.
    assert "RawMaterialProperty1" not in topo.sensors["Machine1"]


def test_injected_fault_is_detected_and_tools_work(fake_csv: Path):
    df, topo = load_real(fake_csv, tick_seconds=10)
    rng = np.random.default_rng(1)
    sc = inject_sensor_fault(
        df,
        topo,
        rng,
        baseline_ticks=180,
        kind="sensor_failure",
        stage="Machine2",
        sensor="MaterialPressure",
        onset=300,
    )
    det = detect(sc.data, DetectorConfig(baseline_ticks=180))
    assert det.triggered and det.first_alarm.sensor == "Machine2.MaterialPressure"
    ctx = ToolContext(sc.data, det, events=[], t_now=330, topology=topo, baseline_ticks=180)
    specs = build_tool_specs(topo)
    assert "Machine2" in specs[2]["input_schema"]["properties"]["stage"]["enum"]
    r, err = ctx.call_tool("find_first_deviation", {})
    assert not err and r["deviations_in_order_of_onset"][0]["sensor"] == "Machine2.MaterialPressure"
    h, err = ctx.call_tool(
        "check_sensor_health", {"stage": "Machine2", "sensor": "MaterialPressure"}
    )
    assert not err and any(
        c["sensor"] == "Combiner.Temperature1" for c in h["physically_coupled_signals"]
    )


def test_missing_csv_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_real(tmp_path / "nope.csv")


def test_dashboard_renders_for_real_scenario(fake_csv: Path, tmp_path: Path):
    """Regression: the renderer used to read the simulator-only ``spec.events`` field."""
    from rootcause.agent.loop import DiagnosisRun, RunUsage
    from rootcause.agent.schema import Diagnosis
    from rootcause.ui.dashboard import render_dashboard

    df, topo = load_real(fake_csv, tick_seconds=10)
    sc = inject_sensor_fault(
        df,
        topo,
        np.random.default_rng(2),
        baseline_ticks=180,
        kind="sensor_failure",
        stage="Machine2",
        sensor="MaterialPressure",
        onset=300,
    )
    det = detect(sc.data, DetectorConfig(baseline_ticks=180))
    run = DiagnosisRun(
        scenario_id=sc.scenario_id,
        model="test",
        status="ok",
        t_now=330,
        usage=RunUsage(),
        diagnosis=Diagnosis(
            status="fault_diagnosed",
            root_cause_stage="Machine2",
            fault_type="sensor_failure",
            affected_sensor="Machine2.MaterialPressure",
            confidence=0.9,
            evidence=[],
            alternative_hypotheses=[],
            recommended_action="Replace the transmitter.",
            requires_downtime=False,
            reasoning_summary="Single flatlined channel.",
        ),
    )
    out = render_dashboard(sc, det, run, tmp_path / "real.html")
    text = out.read_text()
    assert "No events logged" in text and "Machine2.MaterialPressure" in text


def test_injected_scenario_is_scored_on_its_own_alarm(fake_csv: Path):
    """The injected sensor's alarm after onset is used, not an earlier alarm elsewhere."""
    from rootcause.realdata.evaluate import injected_alarm

    df, topo = load_real(fake_csv, tick_seconds=10)
    sc = inject_sensor_fault(
        df,
        topo,
        np.random.default_rng(3),
        baseline_ticks=180,
        kind="sensor_failure",
        stage="Machine4",
        sensor="Pressure",
        onset=400,
    )
    # A natural-looking alarm on another sensor, well before the injection.
    sc.data.loc[250:300, "Machine2.MaterialPressure"] += 30.0
    det = detect(sc.data, DetectorConfig(baseline_ticks=180))
    assert det.first_alarm is not None and det.first_alarm.sensor == "Machine2.MaterialPressure"
    chosen = injected_alarm(det, sc)
    assert chosen is not None
    assert chosen.sensor == "Machine4.Pressure" and chosen.alarm_tick >= 400


def test_submit_diagnosis_stage_enum_follows_topology(fake_csv: Path):
    """The diagnosis stage fields are restricted to the current line's stages."""
    _, topo = load_real(fake_csv, tick_seconds=10)
    spec = [t for t in build_tool_specs(topo) if t["name"] == "submit_diagnosis"][0]
    props = spec["input_schema"]["properties"]
    stage_enum = props["root_cause_stage"]["anyOf"][0]["enum"]
    assert "Machine2" in stage_enum and "feed" not in stage_enum
    hyp_enum = props["alternative_hypotheses"]["items"]["properties"]["root_cause_stage"]
    assert hyp_enum["anyOf"][0]["enum"] == stage_enum
    assert "feed" not in props["root_cause_stage"]["description"]
