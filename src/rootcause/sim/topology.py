"""Line topology: stages, sensors, and the causal links the simulator implements.

This module is the single source of truth for what the line looks like. The simulator
uses it to generate data and the agent's ``get_line_topology`` tool returns a
plain-language rendering of it, so the agent reasons about the same causal model that
produced the data (with no access to the ground-truth fault).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

STAGES: list[str] = ["feed", "heater", "press", "finish"]

SENSORS: dict[str, list[str]] = {
    "feed": ["flow", "pressure", "temperature"],
    "heater": ["flow", "temperature", "power"],
    "press": ["flow", "pressure", "temperature"],
    "finish": ["flow", "temperature", "quality"],
}

SENSOR_UNITS: dict[str, str] = {
    "flow": "units/min",
    "pressure": "kPa",
    "temperature": "degC",
    "power": "kW",
    "quality": "index (0-100)",
}


def sensor_id(stage: str, sensor: str) -> str:
    return f"{stage}.{sensor}"


def all_sensor_ids() -> list[str]:
    return [sensor_id(s, m) for s in STAGES for m in SENSORS[s]]


def split_sensor_id(sid: str) -> tuple[str, str]:
    stage, sensor = sid.split(".", 1)
    return stage, sensor


class LineParams(BaseModel):
    """Abstract physical parameters. Believable causality matters more than realism."""

    supply_nominal: float = 100.0
    ambient_temp: float = 20.0
    # Transport lag (ticks) from the previous stage's outlet into this stage.
    lag: dict[str, int] = Field(
        default_factory=lambda: {"feed": 2, "heater": 4, "press": 5, "finish": 4}
    )
    # Pressure = p_coef * inlet_flow * resistance. Only stages with a pressure sensor matter.
    p_coef: dict[str, float] = Field(
        default_factory=lambda: {"feed": 2.0, "heater": 0.0, "press": 4.0, "finish": 0.0}
    )
    # Heat input per stage. Temperature rise above inherited temperature = heat / flow.
    heat: dict[str, float] = Field(
        default_factory=lambda: {"feed": 2000.0, "heater": 6000.0, "press": 3000.0, "finish": 500.0}
    )
    # Fraction of upstream temperature excess (over ambient) carried into this stage.
    alpha: dict[str, float] = Field(
        default_factory=lambda: {"feed": 0.0, "heater": 0.5, "press": 0.7, "finish": 0.6}
    )
    # Thermal inertia: first-order time constant in ticks.
    thermal_tau: dict[str, float] = Field(
        default_factory=lambda: {"feed": 4.0, "heater": 8.0, "press": 6.0, "finish": 6.0}
    )
    # Quality penalty coefficients at the finish stage.
    quality_temp_coef: float = 1.2
    quality_flow_coef: float = 0.8
    # Slow process wander (AR(1)) on supply and heater heat.
    supply_wander_sigma: float = 0.15
    supply_wander_phi: float = 0.95
    heat_wander_rel_sigma: float = 0.002
    heat_wander_phi: float = 0.95
    # Measurement noise. Deliberately asymmetric so that downstream symptoms often cross a
    # detection threshold before the upstream root cause does.
    noise_sigma: dict[str, float] = Field(
        default_factory=lambda: {
            "feed.flow": 0.9,
            "feed.pressure": 7.0,
            "feed.temperature": 0.6,
            "heater.flow": 0.9,
            "heater.temperature": 0.8,
            "heater.power": 60.0,
            "press.flow": 0.7,
            "press.pressure": 5.0,
            "press.temperature": 0.9,
            "finish.flow": 0.7,
            "finish.temperature": 0.7,
            "finish.quality": 0.9,
        }
    )


CAUSAL_LINKS: list[dict[str, str]] = [
    {
        "from": "supply",
        "to": "feed.flow",
        "relation": "feed.flow follows the raw-material supply rate (lag ~2 ticks).",
    },
    {
        "from": "feed.flow",
        "to": "heater.flow",
        "relation": "Material flows feed -> heater -> press -> finish. Every 'flow' sensor measures "
        "the INLET flow into its stage. A stage's inlet flow equals the previous stage's "
        "outlet flow after a transport lag of 4-5 ticks. A stage's outlet flow is its inlet "
        "flow divided by its internal resistance (1.0 normally, higher when clogged).",
    },
    {
        "from": "flow restriction (clog) inside a stage",
        "to": "that stage's pressure and the NEXT stage's flow",
        "relation": "pressure = coefficient * inlet_flow * resistance. A clog INSIDE stage k raises "
        "k's pressure and leaves k's own (inlet) flow reading unchanged, while the flow reading "
        "at stage k+1 drops after the lag. By contrast, a flow drop arriving FROM upstream "
        "lowers a stage's pressure (inlet flow fell, resistance unchanged). So: pressure UP "
        "with normal inlet flow = restriction here; pressure DOWN with low inlet flow = problem "
        "upstream. The heater stage has no pressure sensor; a heater clog is seen as "
        "press.flow dropping while feed.flow and heater.flow stay normal, with heater.temperature "
        "rising (less material through the same heat input).",
    },
    {
        "from": "flow",
        "to": "temperature",
        "relation": "temperature = ambient + alpha * (upstream temperature excess) + heat / flow. "
        "Less flow through a heated stage means hotter material, so a flow drop anywhere "
        "upstream raises temperatures at every heated stage downstream of it (with lag "
        "and thermal inertia of 4-8 ticks).",
    },
    {
        "from": "heater.power",
        "to": "heater.temperature",
        "relation": "heater.power measures the heat input at the heater stage. An overheating fault "
        "(runaway heat input) raises heater.power and heater.temperature first, then "
        "downstream temperatures, with flows and pressures unchanged.",
    },
    {
        "from": "finish.temperature, finish.flow",
        "to": "finish.quality",
        "relation": "quality falls as finish temperature or finish flow move away from nominal. "
        "Quality is a downstream symptom, never a root cause.",
    },
    {
        "from": "sensor faults",
        "to": "one signal only",
        "relation": "Sensor drift or sensor failure affects the measured value of ONE sensor and has "
        "no physical effect: physically coupled signals (e.g. the same stage's pressure, "
        "or downstream flows) stay normal. A failed sensor reads NaN, a constant, or zero.",
    },
]


class Topology(BaseModel):
    """Line description shared by the simulator, the tools and the agent.

    ``upstream`` maps each stage to the stages that feed it directly, so parallel
    branches and merges can be expressed, not just a chain.
    """

    name: str = "simulated 4-stage line"
    stages: list[str]
    sensors: dict[str, list[str]]
    units: dict[str, str]
    lags_ticks: dict[str, int]
    causal_links: list[dict[str, str]]
    notes: list[str]
    upstream: dict[str, list[str]] = Field(default_factory=dict)
    tick_seconds: float = 60.0
    fault_vocabulary: str = (
        "clog = flow restriction inside a stage; overheating = runaway heat input; supply_drop = "
        "reduced raw-material input to the first stage; sensor_drift / sensor_failure = one "
        "instrument misreporting with no physical effect."
    )

    def downstream(self, stage: str) -> list[str]:
        return [s for s, ups in self.upstream.items() if stage in ups]

    def sensor_ids(self) -> list[str]:
        return [f"{s}.{m}" for s in self.stages for m in self.sensors[s]]

    def all_sensor_names(self) -> list[str]:
        seen: list[str] = []
        for s in self.stages:
            for m in self.sensors[s]:
                if m not in seen:
                    seen.append(m)
        return seen


def get_topology(params: LineParams | None = None) -> Topology:
    params = params or LineParams()
    return Topology(
        stages=STAGES,
        sensors=SENSORS,
        units=SENSOR_UNITS,
        lags_ticks=params.lag,
        causal_links=CAUSAL_LINKS,
        upstream={"feed": [], "heater": ["feed"], "press": ["heater"], "finish": ["press"]},
        tick_seconds=60.0,
        notes=[
            "Stages are strictly sequential: feed -> heater -> press -> finish.",
            "Faults propagate downstream only. Nothing downstream can change an upstream sensor.",
            "To localise a flow problem, find the FIRST stage (walking downstream from feed) whose "
            "inlet flow dropped: the cause is inside the stage immediately upstream of it, or in "
            "the raw-material supply if feed.flow itself dropped.",
            "One tick = one minute. Transport lags are a few ticks; thermal responses take ~4-8 "
            "ticks to settle, so temperature deviations lag flow deviations.",
            "Nominal operating point: supply ~100 units/min, feed ~40 degC, heater ~90 degC, "
            "press ~99 degC, finish ~72 degC, feed pressure ~200 kPa, press pressure ~400 kPa.",
        ],
    )
