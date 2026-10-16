"""Adapter for the public "Multi-stage continuous-flow manufacturing process" dataset.

Source: Liveline Technologies via Kaggle (supergus/multistage-continuousflow-manufacturing-process),
one ~4 h production run sampled at 1 Hz, 116 columns. Not redistributed here: download the CSV
to ``data/real/continuous_factory_process.csv`` (see README).

Line structure (from the dataset description):

    Ambient ─┐
    Machine1 ─┬─> Combiner ─> Stage1Output ─> Machine4 ─> Machine5 ─> Stage2Output
    Machine2 ─┤
    Machine3 ─┘

The adapter resamples to fixed ticks (default 10 s bucket means) and renames columns to
``<stage>.<sensor>`` so the detector, tools and agent work unchanged. There are no fault
labels in this dataset, so evaluation on it is (a) qualitative on natural alarms and
(b) quantitative on *injected* sensor faults, which can be labelled without knowing the
plant physics (see ``inject.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rootcause.sim.topology import Topology

DEFAULT_CSV = Path("data/real/continuous_factory_process.csv")

STAGE_ORDER = [
    "Ambient",
    "Machine1",
    "Machine2",
    "Machine3",
    "Combiner",
    "Stage1Output",
    "Machine4",
    "Machine5",
    "Stage2Output",
]

UPSTREAM = {
    "Ambient": [],
    "Machine1": ["Ambient"],
    "Machine2": ["Ambient"],
    "Machine3": ["Ambient"],
    "Combiner": ["Machine1", "Machine2", "Machine3"],
    "Stage1Output": ["Combiner"],
    "Machine4": ["Stage1Output"],
    "Machine5": ["Machine4"],
    "Stage2Output": ["Machine5"],
}

CAUSAL_LINKS = [
    {
        "from": "Machine1/2/3 (parallel)",
        "to": "Combiner, Stage1Output",
        "relation": "Three machines run in parallel, each with its own raw material feed, heated "
        "zones, motor and material pressure/temperature. Their outputs merge in the combiner. "
        "A change in any one machine's feed or material shows up at the combiner and in the "
        "15 Stage1 output measurements, but NOT in the other two machines.",
    },
    {
        "from": "Stage1Output",
        "to": "Machine4 -> Machine5 -> Stage2Output",
        "relation": "The combined product passes through Machine4 then Machine5 in series "
        "(mostly temperature control), and is measured again at 15 locations (Stage2Output). "
        "Stage1 output deviations propagate to Stage2 output after the transport lag.",
    },
    {
        "from": "RawMaterialProperty1-4",
        "to": "machine behaviour",
        "relation": "Raw material properties are lab values for the material lot in use; they "
        "change in steps when a new lot is loaded. A step there is a supply-side cause, not a "
        "machine fault.",
    },
    {
        "from": "Ambient",
        "to": "everything, weakly",
        "relation": "Ambient humidity and temperature drift slowly over the run and affect all "
        "machines a little. They are rarely a root cause of a sharp deviation.",
    },
    {
        "from": "Output measurements",
        "to": "nothing",
        "relation": "Stage1Output and Stage2Output measurements are product properties: pure "
        "symptoms, never causes. Individual output sensors in this dataset sometimes drop to "
        "zero or flatline (instrument artefacts).",
    },
]


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name)


def column_map(columns: list[str]) -> dict[str, str]:
    """Raw column -> '<stage>.<sensor>' (Actual values only; setpoints dropped)."""
    out: dict[str, str] = {}
    for c in columns:
        if c == "time_stamp" or c.endswith(".Setpoint"):
            continue
        parts = c.split(".")
        if parts[0] == "AmbientConditions":
            out[c] = f"Ambient.{_clean(parts[1])}"
        elif parts[0].startswith("Machine"):
            sensor = (
                _clean("".join(parts[1:-2]))
                if parts[-1] == "Actual"
                else _clean("".join(parts[1:]))
            )
            out[c] = f"{parts[0]}.{sensor}"
        elif parts[0] == "FirstStage":
            out[c] = f"Combiner.{_clean(parts[2])}"
        elif parts[0] in ("Stage1", "Stage2"):
            out[c] = f"{parts[0]}Output.{_clean(parts[2])}"
    return out


def build_topology(tick_seconds: float, sensors: dict[str, list[str]]) -> Topology:
    stages = [s for s in STAGE_ORDER if s in sensors]
    return Topology(
        name="Liveline continuous-flow line (real data, 2 stages / 5 machines)",
        stages=stages,
        sensors={s: sensors[s] for s in stages},
        units={},
        lags_ticks={s: 1 for s in stages},
        causal_links=CAUSAL_LINKS,
        upstream={s: [u for u in UPSTREAM[s] if u in sensors] for s in stages},
        tick_seconds=tick_seconds,
        notes=[
            f"One tick = {tick_seconds:.0f} s (bucket means of 1 Hz data).",
            "Transport lags are not documented; assume seconds to a couple of minutes.",
            "No operator event log exists for this run; list_recent_events returns nothing.",
            "Real plant data: expect slow drifts, lot changes and instrument artefacts. Prefer "
            "insufficient_evidence over a forced story when coupled signals disagree.",
        ],
    )


def load_real(csv: Path = DEFAULT_CSV, tick_seconds: float = 10.0) -> tuple[pd.DataFrame, Topology]:
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Download continuous_factory_process.csv from the Kaggle dataset "
            "'supergus/multistage-continuousflow-manufacturing-process' and place it there."
        )
    raw = pd.read_csv(csv)
    cmap = column_map(list(raw.columns))
    df = raw[["time_stamp", *cmap.keys()]].rename(columns=cmap)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.set_index("time_stamp")
    bucket = f"{int(tick_seconds)}s"
    df = df.resample(bucket).mean(numeric_only=True)
    df = df.reset_index(drop=True)
    df.insert(0, "t", np.arange(len(df)))
    sensors: dict[str, list[str]] = {}
    for sid in df.columns:
        if sid == "t":
            continue
        stage, sensor = sid.split(".", 1)
        sensors.setdefault(stage, []).append(sensor)
    # Drop constant columns (a few setpoint-like 'Actual' channels never move).
    for sid in list(df.columns):
        if sid != "t" and float(np.nanstd(df[sid].to_numpy(dtype=float))) == 0.0:
            stage, sensor = sid.split(".", 1)
            sensors[stage].remove(sensor)
            df = df.drop(columns=[sid])
    sensors = {s: v for s, v in sensors.items() if v}
    return df, build_topology(tick_seconds, sensors)
