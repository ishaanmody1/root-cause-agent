"""Tick-based simulator for a 4-stage sequential line with fault injection.

The model is deliberately abstract. Each tick:

    supply(t)        = nominal * wander * supply_drop_multiplier
    inlet_k(t)       = outlet_{k-1}(t - lag_k)
    resistance_k(t)  = 1 + clog_k(t)
    outlet_k(t)      = inlet_k / resistance_k
    pressure_k(t)    = p_coef_k * inlet_k * resistance_k
    temp_k(t)        -> first-order approach to
                        ambient + alpha_k * (temp_{k-1}(t - lag_k) - ambient) + heat_k / outlet_k
    quality(t)       = 100 - a*|temp_finish - nominal| - b*|flow_finish - nominal|

Flow sensors measure INLET flow into each stage (where flow meters usually sit), so a clog
inside stage k leaves k's own flow reading unchanged, raises k's pressure and temperature,
and lowers the flow reading at stage k+1 after a transport lag.

Measurements add Gaussian noise, then sensor faults (drift / failure) are applied to the
measured value only. Everything is seeded for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rootcause.sim.faults import Event, Fault, GroundTruth, ScenarioSpec
from rootcause.sim.topology import SENSORS, STAGES, LineParams, sensor_id

WARMUP_TICKS = 120


@dataclass
class Scenario:
    spec: ScenarioSpec
    data: pd.DataFrame  # columns: t, <stage>.<sensor> ... (measured values, may contain NaN)
    truth: GroundTruth
    true_data: pd.DataFrame  # noise-free physical values (for debugging / plots, not for the agent)
    params: LineParams

    @property
    def scenario_id(self) -> str:
        return self.spec.scenario_id


def _ground_truth(fault: Fault | None) -> GroundTruth:
    if fault is None:
        return GroundTruth(
            fault_type="none", root_cause_stage=None, description="No fault injected."
        )
    return GroundTruth(
        fault_type=fault.fault_type,
        root_cause_stage=fault.stage,
        sensor=fault.sensor,
        onset_tick=fault.onset_tick,
        severity=fault.severity,
        description=fault.describe(),
    )


def simulate(spec: ScenarioSpec, params: LineParams | None = None) -> Scenario:
    params = params or LineParams()
    rng = np.random.default_rng(spec.seed)
    fault = spec.fault
    total = spec.n_ticks + WARMUP_TICKS

    def wall(t: int) -> int:
        """Convert internal tick (with warm-up) to scenario tick."""
        return t - WARMUP_TICKS

    # --- physical state -------------------------------------------------------------
    supply = np.zeros(total)
    inlet = {s: np.zeros(total) for s in STAGES}
    outlet = {s: np.zeros(total) for s in STAGES}
    pressure = {s: np.zeros(total) for s in STAGES}
    temp = {s: np.zeros(total) for s in STAGES}
    heat_in = {s: np.zeros(total) for s in STAGES}
    quality = np.zeros(total)

    # Steady-state initialisation so the warm-up starts near equilibrium.
    nominal_flow = params.supply_nominal
    nominal_temp: dict[str, float] = {}
    prev_t = params.ambient_temp
    for s in STAGES:
        prev_t = (
            params.ambient_temp
            + params.alpha[s] * (prev_t - params.ambient_temp)
            + params.heat[s] / nominal_flow
        )
        nominal_temp[s] = prev_t
    nominal_quality = 100.0

    supply_wander = 0.0
    heat_wander = 0.0

    def event_heat_multiplier(stage: str, t: int) -> float:
        m = 1.0
        for ev in spec.events:
            if ev.kind == "setpoint_change" and ev.stage == stage and wall(t) >= ev.tick:
                m *= ev.heat_multiplier or 1.0
        return m

    for t in range(total):
        wt = wall(t)
        # Supply with slow wander and optional supply-drop fault.
        supply_wander = params.supply_wander_phi * supply_wander + rng.normal(
            0, params.supply_wander_sigma
        )
        sup = params.supply_nominal + supply_wander
        if fault and fault.fault_type == "supply_drop":
            sup *= 1.0 - fault.severity * fault.ramp(wt)
        supply[t] = sup

        heat_wander = params.heat_wander_phi * heat_wander + rng.normal(
            0, params.heat_wander_rel_sigma
        )

        prev_outlet_series = supply
        prev_stage: str | None = None
        for s in STAGES:
            lag = params.lag[s]
            src_t = t - lag
            fin = prev_outlet_series[src_t] if src_t >= 0 else nominal_flow
            resistance = 1.0
            if fault and fault.fault_type == "clog" and fault.stage == s:
                resistance += fault.severity * fault.ramp(wt)
            inlet[s][t] = fin
            outlet[s][t] = fin / resistance
            pressure[s][t] = params.p_coef[s] * fin * resistance

            heat = params.heat[s] * event_heat_multiplier(s, t)
            if s == "heater":
                heat *= 1.0 + heat_wander
            if fault and fault.fault_type == "overheating" and fault.stage == s:
                heat *= 1.0 + fault.severity * fault.ramp(wt)
            heat_in[s][t] = heat

            if prev_stage is None:
                up_temp = params.ambient_temp
            else:
                up_temp = temp[prev_stage][src_t] if src_t >= 0 else nominal_temp[prev_stage]
            target = (
                params.ambient_temp
                + params.alpha[s] * (up_temp - params.ambient_temp)
                + heat / max(outlet[s][t], 1e-3)
            )
            prev_temp = temp[s][t - 1] if t > 0 else nominal_temp[s]
            temp[s][t] = prev_temp + (target - prev_temp) / params.thermal_tau[s]

            prev_outlet_series = outlet[s]
            prev_stage = s

        quality[t] = (
            nominal_quality
            - params.quality_temp_coef * abs(temp["finish"][t] - nominal_temp["finish"])
            - params.quality_flow_coef * abs(outlet["finish"][t] - nominal_flow)
        )

    # --- measurement ----------------------------------------------------------------
    sl = slice(WARMUP_TICKS, total)
    ticks = np.arange(spec.n_ticks)
    true_cols: dict[str, np.ndarray] = {"t": ticks}
    meas_cols: dict[str, np.ndarray] = {"t": ticks}

    for s in STAGES:
        for m in SENSORS[s]:
            sid = sensor_id(s, m)
            match m:
                case "flow":
                    # Flow meters sit at each stage's INLET. A clog inside a stage therefore
                    # shows up first as a flow drop at the NEXT stage, not at its own meter.
                    true_v = inlet[s][sl]
                case "pressure":
                    true_v = pressure[s][sl]
                case "temperature":
                    true_v = temp[s][sl]
                case "power":
                    true_v = heat_in[s][sl]
                case "quality":
                    true_v = quality[sl]
                case _:
                    raise ValueError(m)
            true_cols[sid] = true_v.copy()
            meas = true_v + rng.normal(0, params.noise_sigma[sid], size=spec.n_ticks)

            if fault and fault.stage == s and fault.sensor == m:
                if fault.fault_type == "sensor_drift":
                    nominal = float(np.nanmean(true_v[: spec.baseline_ticks]))
                    ramp = np.array([fault.ramp(int(k)) for k in ticks])
                    meas = meas + fault.severity * nominal * ramp
                elif fault.fault_type == "sensor_failure":
                    on = fault.onset_tick
                    if fault.failure_mode == "nan":
                        meas[on:] = np.nan
                    elif fault.failure_mode == "zero":
                        meas[on:] = 0.0
                    elif fault.failure_mode == "flatline":
                        meas[on:] = meas[on - 1] if on > 0 else meas[0]
            meas_cols[sid] = meas

    data = pd.DataFrame(meas_cols)
    true_data = pd.DataFrame(true_cols)
    return Scenario(
        spec=spec, data=data, truth=_ground_truth(fault), true_data=true_data, params=params
    )


def nominal_values(params: LineParams | None = None) -> dict[str, float]:
    """Noise-free nominal operating point per sensor, useful for documentation and tests."""
    params = params or LineParams()
    spec = ScenarioSpec(scenario_id="nominal", seed=0, n_ticks=50, baseline_ticks=50)
    quiet = params.model_copy(
        update={
            "supply_wander_sigma": 0.0,
            "heat_wander_rel_sigma": 0.0,
            "noise_sigma": dict.fromkeys(params.noise_sigma, 0.0),
        }
    )
    sc = simulate(spec, quiet)
    return {c: float(sc.data[c].iloc[-1]) for c in sc.data.columns if c != "t"}


def is_finite(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


__all__ = ["Scenario", "simulate", "nominal_values", "Event", "Fault", "GroundTruth"]
