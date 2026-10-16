# root-cause-agent

An agent that diagnoses faults on a multi-stage production line. A cheap statistical
detector watches the sensors; when it fires, the agent investigates with a handful of tools
and returns a structured diagnosis: which stage caused it, what kind of fault, how sure it
is, and the evidence.

The problem it exists for: in a sequential line the alarm fires where the symptom lands,
not where the fault is. A partial clog in the feed stage shows up as a starving press stage
two hops downstream, and a threshold alert blames the press.

![Dashboard for a heater clog: alarm fired on press flow, the agent traced it to the heater](docs/dashboard.jpg)

## Quickstart

```bash
make setup        # installs uv if needed, then dependencies
make demo         # offline: a misleading clog, rule-based backend
```

With `ANTHROPIC_API_KEY` in `.env`:

```bash
uv run rootcause diagnose --kind clog --seed 6 -v   # one scenario, LLM agent
uv run rootcause eval --n 48 --backend anthropic    # scenario suite with baselines
uv run rootcause real-eval --n 20                   # injected faults on the real dataset
```

Runs write JSON results, a trace per diagnosis and an HTML dashboard under `runs/`.

## How it works

- `sim/` – a four-stage line (feed, heater, press, finish) with simple causal physics:
  transport lags, thermal inertia, pressure that rises when a stage restricts. Five fault
  types can be injected with known ground truth, plus operator events as distractors.
- `detect/` – robust z-score with a persistence rule, and separate rules for missing,
  flatlined and stuck-at-zero sensors. No LLM. Runs on every tick.
- `agent/` – a loop over the Anthropic Messages API with seven strict tools (topology,
  first deviation, compare to baseline, query sensor, sensor health, event log, submit).
  The loop owns step and dollar budgets, cost accounting, a JSONL trace, prompt caching and
  the failure paths (refusal, truncation, model never submitting). The final answer is a
  validated tool call, not prose.
- `evals/` – scores root-cause stage, fault type, calibration, abstentions, confident errors
  and cost against two non-LLM baselines.
- `realdata/` – adapter for the Liveline continuous-flow dataset (two stages, five
  machines, parallel branches into a combiner), with labelled sensor faults injected into
  the real series since the dataset has no labels of its own.
- `ui/` – one self-contained HTML page per diagnosis, laid out like a plant HMI.

The agent only sees data up to a few ticks after the alarm, so it is diagnosing a fault
that is still developing. The diagnosis schema is in `agent/schema.py`.

## Results

48 simulated scenarios, 40 alarms, seed 0. About a third of alarms fire at a stage other
than the root cause. Full reports: [docs/eval-llm-report.md](docs/eval-llm-report.md),
[docs/eval-offline-report.md](docs/eval-offline-report.md).

| | Naive (blame first alarm) | Rule-based tracer | LLM agent (Sonnet 5.5) |
|---|---|---|---|
| Stage and type both correct | 63.9% | 91.7% | 100% |
| Same, misleading alarms only (n=11) | 0% | 100% | 100% |
| Confident errors (wrong at ≥ 0.75) | 0% | 5.0% | 0% |
| False positives on no-fault alarms | 100% | 0% | 0% |
| Brier score | 0.26 | 0.089 | 0.032 |
| Cost per diagnosis | – | – | $0.026 |

The rule-based tracer is a hand-written policy that walks the topology upstream using the
same tools. It knows the simulator's physics, so it does well here and cannot run on the
real line at all. The LLM agent ran on the real line with no code changes. Seed 1 gives the
same LLM numbers; the suite is close to saturated and the next step is harder scenarios.

On the real dataset (20 injected sensor faults, 85 channels, 49 natural alarms in a four-hour
run) the detector caught 14; the agent got 5 right, abstained on 6, and was wrong on 3, never
above 0.5 confidence. On the first natural alarm it abstained and pointed to a material lot
change, which for an unlabelled run is the right answer. Details and the bugs found along
the way are in [docs/notes.md](docs/notes.md).

## Data

The real dataset is `supergus/multistage-continuousflow-manufacturing-process` on Kaggle
(Liveline Technologies). It is not redistributed here; put the CSV at
`data/real/continuous_factory_process.csv`.

## Layout

```
src/rootcause/   sim/ detect/ agent/ evals/ realdata/ ui/ cli.py
tests/           32 tests, all offline (the rule-based backend stands in for the API)
docs/            eval reports, design notes, screenshot
```

MIT licensed.
