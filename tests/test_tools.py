import json

from rootcause.agent.schema import Diagnosis, strict_schema
from rootcause.agent.tools import TOOL_SPECS, ToolContext


def test_tools_respect_t_now(clog_scenario):
    sc, det = clog_scenario
    ctx = ToolContext.from_scenario(sc, det, grace_ticks=20)
    assert ctx.t_now == det.alarm_tick + 20
    r, err = ctx.call_tool(
        "query_sensor", {"stage": "press", "sensor": "flow", "t_start": 0, "t_end": 10_000}
    )
    assert not err and r["window"][1] == ctx.t_now
    r, err = ctx.call_tool("list_recent_events", {})
    assert not err and all(e["tick"] <= ctx.t_now for e in r["events"])


def test_tool_errors_are_reported_not_raised(clog_scenario):
    sc, det = clog_scenario
    ctx = ToolContext.from_scenario(sc, det)
    r, err = ctx.call_tool("query_sensor", {"stage": "heater", "sensor": "pressure"})
    assert err and "no sensor" in r["error"]
    r, err = ctx.call_tool("find_first_deviation", {"z_threshold": 0.1})
    assert err
    r, err = ctx.call_tool("nope", {})
    assert err


def test_find_first_deviation_orders_and_lists_normal(clog_scenario):
    sc, det = clog_scenario
    ctx = ToolContext.from_scenario(sc, det)
    r, _ = ctx.call_tool("find_first_deviation", {})
    ticks = [d["first_tick"] for d in r["deviations_in_order_of_onset"]]
    assert ticks == sorted(ticks)
    assert "feed.flow" in r["sensors_still_normal"]
    assert "heater.flow" in r["sensors_still_normal"]
    assert any(
        d["sensor"] == "press.flow" and d["direction"] == "low"
        for d in r["deviations_in_order_of_onset"]
    )


def test_compare_and_health(clog_scenario):
    sc, det = clog_scenario
    ctx = ToolContext.from_scenario(sc, det)
    r, _ = ctx.call_tool("compare_to_baseline", {"stage": "heater", "sensor": "temperature"})
    assert r["verdict"] == "elevated"
    h, _ = ctx.call_tool("check_sensor_health", {"stage": "press", "sensor": "flow"})
    assert h["own_verdict"] == "depressed"
    coupled = {c["sensor"]: c["verdict"] for c in h["physically_coupled_signals"]}
    assert coupled["heater.flow"] == "normal"


def test_tool_specs_are_strict_and_serialisable():
    s = json.dumps(TOOL_SPECS)
    assert "submit_diagnosis" in s
    for t in TOOL_SPECS:
        assert t["strict"] is True
        schema = t["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())


def test_strict_schema_inlines_refs_and_strips_bounds():
    schema = strict_schema(Diagnosis)
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)
    assert "minimum" not in json.dumps(schema)
    ev = schema["properties"]["evidence"]["items"]
    assert ev["additionalProperties"] is False
