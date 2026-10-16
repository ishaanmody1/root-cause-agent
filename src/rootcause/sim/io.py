"""Persist scenarios as CSV + JSON so runs are inspectable without Python."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rootcause.sim.faults import GroundTruth, ScenarioSpec
from rootcause.sim.simulator import Scenario
from rootcause.sim.topology import LineParams


def save_scenario(sc: Scenario, root: Path) -> Path:
    d = root / sc.scenario_id
    d.mkdir(parents=True, exist_ok=True)
    sc.data.to_csv(d / "data.csv", index=False)
    sc.true_data.to_csv(d / "true_data.csv", index=False)
    (d / "spec.json").write_text(sc.spec.model_dump_json(indent=2))
    (d / "truth.json").write_text(sc.truth.model_dump_json(indent=2))
    (d / "params.json").write_text(sc.params.model_dump_json(indent=2))
    return d


def load_scenario(d: Path) -> Scenario:
    spec = ScenarioSpec.model_validate_json((d / "spec.json").read_text())
    truth = GroundTruth.model_validate_json((d / "truth.json").read_text())
    params = LineParams.model_validate_json((d / "params.json").read_text())
    data = pd.read_csv(d / "data.csv")
    true_path = d / "true_data.csv"
    true_data = pd.read_csv(true_path) if true_path.exists() else data.copy()
    return Scenario(spec=spec, data=data, truth=truth, true_data=true_data, params=params)


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())
