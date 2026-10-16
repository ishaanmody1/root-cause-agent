# Root-cause diagnosis agent for a multi-stage production line

An AI agent that watches sensor data from a sequential production line, gets woken up by a
cheap statistical detector when something looks off, and then traces the symptom back to the
stage that actually caused it, explaining the evidence chain in plain language for a
reliability engineer.

The hard part is not spotting the anomaly. It is that **the stage where the alarm fires is
usually not where the problem is.** A partial clog in the feed stage shows up first as a
starving press stage three hops downstream, and a threshold alert blames the press.

![Dashboard: sensor traces with the alarm window shaded, the root-cause stage highlighted, and the structured diagnosis beside it](docs/dashboard.jpg)

*A heater clog. The alarm fired on `press.flow` (two stages downstream). The agent traced it
back: the press inlet flow fell while the heater inlet flow stayed normal, so the restriction is
inside the heater, confirmed by the heater running hot with unchanged heat input.*

## What is in the box

```mermaid
flowchart LR
    subgraph Data
        SIM[Simulator<br/>4 stages, 12 sensors<br/>5 labelled fault types<br/>operator events]
        REAL[Real dataset adapter<br/>Liveline 2-stage line<br/>86 sensors, 1 Hz]
    end
    DET[Detector<br/>robust z-score + persistence<br/>NaN / flatline / stuck-zero<br/>no LLM]
    subgraph Agent
        LOOP[Hand-rolled loop<br/>step + cost budget<br/>trace, retries, refusal handling]
        TOOLS[Tools<br/>get_line_topology<br/>find_first_deviation<br/>compare_to_baseline<br/>query_sensor<br/>check_sensor_health<br/>list_recent_events<br/>submit_diagnosis]
        LLM[(Claude<br/>strict tool use)]
    end
    OUT[Structured diagnosis<br/>root cause, type, confidence,<br/>evidence chain, alternatives,<br/>action]
    EVAL[Eval suite<br/>accuracy, calibration,<br/>abstention, cost, baselines]
    UI[HTML dashboard<br/>self-contained SVG]
    SIM --> DET
    REAL --> DET
    DET -- alarm --> LOOP
    LOOP <--> LLM
    LOOP <--> TOOLS
    TOOLS --> SIM
    TOOLS --> REAL
    LOOP --> OUT
    OUT --> EVAL
    OUT --> UI
```

| Layer | What it does | Where |
|---|---|---|
| Simulator | Abstract but causal physics for feed → heater → press → finish, with transport lags and thermal inertia. Injects clog, overheating, supply drop, sensor drift, sensor failure, plus benign and distracting operator events. Ground truth is known. | `src/rootcause/sim` |
| Detector | Deterministic and cheap. Robust baseline (median/MAD) from a known-good window, EWMA-smoothed z-score with a persistence rule, and separate rules for missing, flatline and stuck-at-zero signals. Guards against channels that are legitimately constant. | `src/rootcause/detect` |
| Agent | A hand-rolled loop over the Anthropic Messages API with strict tools. Owns budgets (steps and dollars), token/cost accounting, a JSONL trace of every model and tool call, prompt caching, and the failure paths (refusal, truncation, model never submitting). | `src/rootcause/agent` |
| Evals | Balanced scenario suite with deliberately misleading cases. Scores root-cause accuracy, fault-type accuracy, calibration (Brier, ECE), abstention rate, confident-error rate, false positives on no-fault alarms, tool calls, tokens, cost and latency. Two non-LLM baselines for comparison. | `src/rootcause/evals` |
| Real data | Adapter for the public Liveline continuous-flow dataset (parallel machines feeding a combiner, then two machines in series). Labelled sensor faults are injected into the real series for a quantitative check. | `src/rootcause/realdata` |
| UI | One self-contained HTML file per diagnosis: all sensor traces with baseline band, alarm window and true onset, next to the diagnosis, evidence chain and the full agent trace. No server, no CDN. | `src/rootcause/ui` |

## Quickstart

```bash
make setup                      # installs uv if needed, then all dependencies
make demo                       # offline: misleading clog scenario, rule-based backend
```

With an Anthropic key (copy `.env.example` to `.env`, or export `ANTHROPIC_API_KEY`):

```bash
uv run rootcause diagnose --kind clog --seed 6 -v        # LLM agent, verbose trace
uv run rootcause eval --n 48 --backend anthropic         # full eval with the LLM column
uv run rootcause real-eval --n 20                        # injected faults on the real dataset
```

Every `diagnose` and `eval` run writes JSON results, a JSONL trace per diagnosis, and HTML
dashboards under `runs/`. Open `runs/<run>/index.html` for an eval, or the per-scenario
`.html` for a single diagnosis.

Other commands: `rootcause simulate` (one scenario to CSV), `rootcause suite` (the whole
suite to CSV), `rootcause real-diagnose` (a natural alarm in the real data), `rootcause
dashboard <run_dir>` (rebuild an index). `--help` on any of them.

## How the agent works

The detector hands the agent a report: which sensor alarmed first, every alarm confirmed so
far, and the current tick. The agent only sees data up to that tick (by default 25 ticks after
the alarm), so it is diagnosing a fault that is still developing, like a real engineer would.

The system prompt gives it a method rather than answers: fetch the topology, find which
signals moved and in what order, form two or three hypotheses that each explain the whole
pattern, then confirm or kill them with the physically coupled signals. The topology tool
describes the causal rules of the specific line (for the simulator: a clog raises its own
stage's pressure and starves the next stage; a supply drop lowers feed pressure; a sensor
fault moves exactly one signal), so the same prompt and tools work on the real line, which
has a different shape.

The diagnosis is itself a strict tool call (`submit_diagnosis`), validated against a pydantic
schema:

```
status                  fault_diagnosed | no_fault | insufficient_evidence
root_cause_stage        stage name or null
fault_type              clog | overheating | sensor_drift | sensor_failure | supply_drop | none
affected_sensor         for sensor faults
confidence              P(stage AND type are both right)
evidence[]              signal, observation with ticks and magnitudes, why it matters
alternative_hypotheses[] stage, type, confidence, why less likely
recommended_action      what to inspect, where
requires_downtime       bool
reasoning_summary       the causal chain in plain language
```

Production concerns the loop handles explicitly:

- **Budgets.** Max model calls and max dollars per diagnosis. When either is hit the agent gets
  one final, submit-only turn; if it still does not submit, the run is recorded as
  `budget_exhausted` with a synthesized `insufficient_evidence` result.
- **Observability.** Every model call (stop reason, latency, token usage, cost) and every tool
  call (input, output, error flag) lands in the trace. Cost is computed from a pricing table,
  including cache reads and writes.
- **Bad data.** Tools never raise into the model. Unknown stages, out-of-range windows and
  missing columns come back as `is_error` tool results the model can recover from. NaN, zero
  and flatlined sensors are first-class concepts in the detector and the tools.
- **Honest outcomes.** `insufficient_evidence` is a legitimate answer and is scored separately
  from wrong answers. The eval tracks confident errors (wrong at confidence ≥ 0.75) because a
  confidently wrong diagnosis costs more than an abstention.
- **API failure modes.** Refusals (`stop_reason: refusal`), truncation (`max_tokens`), a model
  that answers in prose instead of calling the tool, rate limits and connection errors all
  have explicit paths and end in a well-formed run record.
- **Prompt caching.** The system prompt and tool definitions are stable and marked cacheable;
  the volatile detector report comes after them.

## Evals

Three systems are scored on the same scenarios:

- **Naive**: blame the stage where the first alarm fired, guess the type from the signal.
  This is what a threshold-alert dashboard gives you.
- **Rule-based tracer**: a hand-written policy that drives the same tools through the same
  loop. It walks the topology upstream from the first flow deviation and checks coupled
  signals. It knows the simulator's physics, so it is a strong baseline there and a useless
  one anywhere else (it cannot run on the real line).
- **LLM agent**: Claude with the tools, no knowledge of the simulator beyond the topology
  tool.

Offline results, 48 scenarios (40 alarms), seed 0, from
[docs/eval-offline-report.md](docs/eval-offline-report.md):

| Metric | Naive | Rule-based tracer | LLM agent |
|---|---|---|---|
| Root-cause stage accuracy | 69.4% | 94.4% | run `make eval-llm` |
| Fault-type accuracy | 94.4% | 91.7% | |
| Both correct | 63.9% | 91.7% | |
| Both correct, **misleading cases only** (n=11) | **0.0%** | 100.0% | |
| Abstained | 0.0% | 0.0% | |
| Confident errors | 0.0% | 5.0% | |
| False positives on no-fault alarms | 100.0% | 0.0% | |
| Brier score | 0.26 | 0.089 | |

Detector on the same suite: 0% false alarms on 6 normal runs, 100% detection on 36 faults,
median 17 ticks from fault onset to alarm, and 30.6% of alarms fire at a stage other than the
root cause.

Two things to read from this table. The naive approach is right about *what kind* of fault it
is most of the time and wrong about *where* in every misleading case, which is the failure
the project targets. And the rule-based tracer shows that on a simulator whose physics you
wrote, a few rules go a long way; its weak spots are the ambiguous cases (overheating with a
distracting setpoint change logged at the same stage: 50% correct) where judgment about
magnitude is needed. The LLM column is where the agent has to earn its keep: match the
tracer on the simulator, and generalize to the real line where there are no rules to write.

The LLM column is empty in this README because the machine this was built on had no API
credentials at the time. `make eval-llm` runs it (about 40 diagnoses at roughly $0.05 to
$0.10 each with the default `claude-sonnet-5-5` at medium effort) and writes the full report
with calibration bins, per-kind breakdown, tokens, cost and latency.

## Real-data validation

The public **Multi-stage continuous-flow manufacturing process** dataset (Liveline
Technologies, via Kaggle, listed in awesome-industrial-datasets) is one ~4 hour run of a real
line at 1 Hz: three parallel machines feed a combiner, the product is measured at 15
locations, then passes through two more machines in series and is measured again.

The adapter resamples to 10-second ticks, renames the 85 non-constant channels to
`<stage>.<sensor>`, and declares the topology as a DAG with the causal notes the agent needs
(parallel machines do not affect each other, output measurements are symptoms not causes, raw
material properties change in steps at lot changes). The detector, tools, prompt and
dashboard run unchanged.

The dataset has no fault labels, so validation has two parts:

1. **Injected sensor faults** (`rootcause real-eval`): drift, NaN, flatline or zero injected
   into one real channel at a random time. Ground truth is exact, and the real background
   (drifts, lot changes, instrument artefacts: 49 natural alarms in this run) makes it
   genuinely hard to say "this one sensor is lying" rather than "the process changed".
2. **Natural alarms** (`rootcause real-diagnose --alarm N`): no labels, judged
   qualitatively on whether the evidence chain is consistent with the topology and whether the
   agent abstains when it should.

Building the adapter changed the detector: real controlled and lab-value channels sit exactly
constant for long stretches, which the first flatline rule read as 66 dead sensors. Flatline
and stuck-zero rules now apply only to sensors that vary during their own baseline.

The CSV is not redistributed (license unstated on Kaggle). Download it from the Kaggle dataset
`supergus/multistage-continuousflow-manufacturing-process` to
`data/real/continuous_factory_process.csv`.

## Design trade-offs

- **Hand-rolled loop instead of a framework.** The loop is ~350 lines and every production
  concern above lives in it, visibly. A framework would hide exactly the parts worth
  reviewing. The cost is that the SDK's tool runner would have saved the tool dispatch code.
- **Final answer as a strict tool, not structured output.** `submit_diagnosis` keeps the
  diagnosis in the same trace as the evidence gathering, lets the model keep investigating if
  the submission fails validation, and gives the loop a clean terminal condition. Structured
  output on the final message would have been simpler but harder to nudge.
- **Deterministic detector, LLM only on alarm.** Detection runs on every tick and must be
  cheap and explainable; the LLM runs tens of times a day, not thousands.
- **Flow meters at stage inlets.** The first simulator version put them at outlets, which made
  the clogged stage alarm first and the problem trivial. Inlet metering is what real lines
  usually have and is what produces the misleading downstream symptom.
- **Deliberately asymmetric sensor noise.** Pressure and power sensors are noisier than flow
  sensors, so a root-cause signal often crosses the alarm threshold *after* a quieter
  downstream signal. Ordering alone does not solve the problem; physics does.
- **Sonnet at medium effort by default.** The task is bounded and tool-heavy; the eval reports
  cost per diagnosis so the model and effort can be chosen on evidence. Both are flags.
- **Inline SVG dashboards, no Plotly.** A CDN script failed to load in a sandboxed viewer
  during development; a self-contained file that opens anywhere, including from `file://`,
  is worth more than hover tooltips.
- **Rule-based backend speaks the SDK's `Message` type.** The loop cannot tell it from the
  real API, so the whole pipeline, including budget exhaustion and error paths, is tested in
  CI without a key.

## Limitations and next steps

- The LLM eval column needs to be populated and the prompt tuned against it; the calibration
  guidance in the prompt is untested until then.
- Single-fault scenarios only. Compound faults and a fault during a setpoint transient are
  the obvious next scenario kinds.
- The simulator's physics are abstract by design. Real lines have control loops that mask
  symptoms; the real-data adapter is where that shows up, and only sensor faults can be
  labelled there.
- Out of scope, per the brief: alert routing, manuals/RAG, plant integrations.

## Project layout

```
src/rootcause/
  sim/        topology.py (line model + causal rules), simulator.py, faults.py, scenarios.py, io.py
  detect/     detector.py
  agent/      loop.py, tools.py, schema.py, prompts.py, pricing.py, fake.py (rule-based backend)
  evals/      runner.py, metrics.py, baselines.py, report.py, types.py
  realdata/   adapter.py, inject.py, evaluate.py
  ui/         dashboard.py
  cli.py
tests/        29 tests, all offline
docs/         screenshot, offline eval report
```
