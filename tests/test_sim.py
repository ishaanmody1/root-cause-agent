from rootcause.sim import SENSORS, STAGES, generate_suite, make_scenario, nominal_values, simulate
from rootcause.sim.faults import Fault, ScenarioSpec


def test_deterministic():
    a = simulate(make_scenario("clog", 3)).data
    b = simulate(make_scenario("clog", 3)).data
    assert a.equals(b)


def test_nominal_operating_point():
    nom = nominal_values()
    assert abs(nom["feed.flow"] - 100) < 0.5
    assert abs(nom["feed.pressure"] - 200) < 1
    assert abs(nom["heater.temperature"] - 90) < 0.5
    assert abs(nom["finish.quality"] - 100) < 0.5


def test_all_sensors_present():
    sc = simulate(make_scenario("normal", 1))
    cols = set(sc.data.columns) - {"t"}
    assert cols == {f"{s}.{m}" for s in STAGES for m in SENSORS[s]}
    assert len(sc.data) == 600


def test_clog_propagates_downstream_not_upstream():
    spec = ScenarioSpec(
        scenario_id="t",
        seed=1,
        fault=Fault(fault_type="clog", stage="heater", onset_tick=300, severity=0.4, ramp_ticks=20),
    )
    sc = simulate(spec)
    before = sc.true_data.iloc[200:290].mean()
    after = sc.true_data.iloc[400:].mean()
    # Inlet flow at the clogged stage is unchanged; the next stage starves.
    assert abs(after["heater.flow"] - before["heater.flow"]) < 2.5
    assert after["press.flow"] < before["press.flow"] * 0.8
    assert after["finish.flow"] < before["finish.flow"] * 0.8
    # Clogged stage runs hot; upstream untouched.
    assert after["heater.temperature"] > before["heater.temperature"] + 10
    assert abs(after["feed.pressure"] - before["feed.pressure"]) < 5
    assert after["press.pressure"] < before["press.pressure"] * 0.85
    assert after["finish.quality"] < 90


def test_supply_drop_lowers_feed_pressure_but_clog_raises_it():
    sup = simulate(
        ScenarioSpec(
            scenario_id="s",
            seed=2,
            fault=Fault(
                fault_type="supply_drop", stage="feed", onset_tick=300, severity=0.3, ramp_ticks=10
            ),
        )
    )
    clog = simulate(
        ScenarioSpec(
            scenario_id="c",
            seed=2,
            fault=Fault(
                fault_type="clog", stage="feed", onset_tick=300, severity=0.3, ramp_ticks=10
            ),
        )
    )
    assert sup.true_data["feed.pressure"].iloc[-1] < 160
    assert clog.true_data["feed.pressure"].iloc[-1] > 240


def test_sensor_faults_have_no_physical_effect():
    spec = ScenarioSpec(
        scenario_id="d",
        seed=4,
        fault=Fault(
            fault_type="sensor_drift",
            stage="press",
            sensor="flow",
            onset_tick=300,
            severity=-0.2,
            ramp_ticks=50,
        ),
    )
    sc = simulate(spec)
    assert sc.data["press.flow"].iloc[-1] < 85  # measured drifts
    assert abs(sc.true_data["press.flow"].iloc[-1] - 100) < 3  # physics unchanged
    assert abs(sc.data["finish.flow"].iloc[-1] - 100) < 4

    nan_spec = ScenarioSpec(
        scenario_id="n",
        seed=4,
        fault=Fault(
            fault_type="sensor_failure",
            stage="finish",
            sensor="temperature",
            onset_tick=300,
            failure_mode="nan",
        ),
    )
    sc2 = simulate(nan_spec)
    assert sc2.data["finish.temperature"].iloc[300:].isna().all()
    assert sc2.data["finish.temperature"].iloc[:300].notna().all()


def test_suite_is_balanced_and_reproducible():
    a = generate_suite(16, seed=0)
    b = generate_suite(16, seed=0)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    kinds = [s.tags[0] for s in a]
    assert kinds.count("clog") == 4 and kinds.count("normal") == 2


def test_clog_at_finish_not_generated():
    for s in generate_suite(70, seed=1):
        if s.fault and s.fault.fault_type == "clog":
            assert s.fault.stage != "finish"
