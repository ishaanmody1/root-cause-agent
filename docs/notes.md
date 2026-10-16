# Design notes

Longer-form notes that used to live in the README: how the agent is set up, what the eval
numbers mean, what happened on the real data, and the trade-offs behind the main decisions.

## The agent loop

The detector hands the agent a report: which sensor alarmed first, every alarm confirmed so
far, and the current tick. The agent only sees data up to that tick (25 ticks after the
alarm by default), so it diagnoses a fault that is still developing.

The system prompt gives it a method rather than answers: fetch the topology, find which
signals moved and in what order, form two or three hypotheses that each explain the whole
pattern, then confirm or kill them with the physically coupled signals. The topology tool
carries the causal rules of the specific line (a clog raises its own stage's pressure and
starves the next stage; a supply drop lowers feed pressure; a sensor fault moves exactly one
signal), which is why the same prompt and tools work on the real line.

The diagnosis is a strict tool call, `submit_diagnosis`, validated against a pydantic schema:

```
status                   fault_diagnosed | no_fault | insufficient_evidence
root_cause_stage         stage name or null
fault_type               clog | overheating | sensor_drift | sensor_failure | supply_drop | none
affected_sensor          for sensor faults
confidence               P(stage AND type are both right)
evidence[]               signal, observation with ticks and magnitudes, why it matters
alternative_hypotheses[] stage, type, confidence, why less likely
recommended_action       what to inspect, where
requires_downtime        bool
reasoning_summary        the causal chain in plain language
```

Things the loop handles explicitly:

- Budgets. Max model calls and max dollars per diagnosis. When either is hit the agent gets
  one final submit-only turn; if it still does not submit, the run is recorded as
  `budget_exhausted` with a synthesized `insufficient_evidence` result.
- Tracing. Every model call (stop reason, latency, tokens, cost) and tool call (input,
  output, error flag) lands in a JSONL trace. Cost comes from a pricing table and includes
  cache reads and writes.
- Bad data. Tools never raise into the model; unknown stages, bad windows and missing
  columns come back as error results. NaN, zero and flatlined sensors are first-class in the
  detector and the tools.
- Abstention. `insufficient_evidence` is a legitimate answer and is scored separately from
  wrong answers. Confident errors (wrong at confidence ≥ 0.75) are tracked because a
  confidently wrong diagnosis costs more than an abstention.
- API failure modes. Refusals, truncation, a model that answers in prose, rate limits and
  connection errors all end in a well-formed run record.
- Prompt caching. The system prompt and tool definitions are stable and cacheable; the
  detector report comes after them.

## Reading the eval table

Three systems run on the same scenarios. Naive blames the stage where the first alarm fired
and guesses the type from the signal; it is what a threshold dashboard gives you. The
rule-based tracer drives the same tools through the same loop and walks the topology
upstream from the first flow deviation. The LLM agent knows nothing about the simulator
beyond the topology tool.

The naive approach is usually right about what kind of fault it is and wrong about where in
every misleading case. The tracer shows that a few rules go a long way on a simulator whose
physics you wrote; it loses on the ambiguous cases, such as overheating at a stage with no
power sensor, or with a distracting setpoint change logged at the same stage (50% correct),
where judgment about magnitude is needed. The LLM agent matches the tracer on the misleading
clogs and beats it on those ambiguous cases, with no confident errors and a Brier score a
third of the tracer's, at about 2.5 cents and 11 seconds per diagnosis.

How the LLM column was produced. The first run with the untuned prompt scored 97.2% on seed
0; the one miss was an overheating fault at the feed stage (no power sensor), called a sensor
drift at 0.65 because the downstream heater temperature was only "slightly elevated" and the
prompt said a sensor fault's coupled signals are "normal". The prompt was tuned on that one
failure and validated on seed 1, a suite it had never seen: 100% and zero confident errors
both before and after the change, so no regression, at one extra tool call per diagnosis.
One scenario separates 97.2% from 100% and sampling is not deterministic, so read the LLM
column as "at or near ceiling on this suite". Across both seeds the agent is mildly
under-confident (ECE 0.17), which is the safe side for a diagnosis a human acts on.

Detector on the same suite: 0% false alarms on 6 normal runs, 100% detection on 36 faults,
median 17 ticks from onset to alarm, 30.6% of alarms at the wrong stage.

## Real data

The Liveline dataset is one four-hour run at 1 Hz: three parallel machines feed a combiner,
the product is measured at 15 locations, then passes through two more machines in series and
is measured again. The adapter resamples to 10-second ticks, renames the 85 non-constant
channels to `<stage>.<sensor>`, and declares the topology as a DAG with the notes the agent
needs (parallel machines do not affect each other, output measurements are symptoms not
causes, raw material properties change in steps at lot changes).

There are no fault labels, so validation has two parts: sensor faults (drift, NaN, flatline,
zero) injected into one real channel at a random time, where ground truth is exact; and
natural alarms judged qualitatively.

Results with Sonnet 5.5 at medium effort, 20 injections, seed 0. The detector caught 14; the
misses are drifts on channels that had already alarmed naturally, or that stayed inside the
real noise. On the 14 alarms the agent was right on 5, abstained on 6 and wrong on 3, never
above 0.5 confidence; mean cost $0.052. The naive baseline scores 100% on this suite by
construction (every injection is a sensor fault on the alarming channel), so it is a ceiling,
not a competitor. The three errors were all "clog at the alarming machine" at 0.4 to 0.5: with
85 sensors and 49 natural alarms in four hours, "one abnormal signal with flat neighbours" is
rarely cleanly true. The abstentions were missing-data and stuck-zero injections on
raw-material lab-value channels, which change in steps at lot changes anyway. On the first
natural alarm the agent abstained at 0.30: four output measurements moved together and
propagated downstream, so a real product change, but nothing visible upstream, and it
suggested checking for a lot change.

Building the adapter changed the detector. Real controlled and lab-value channels sit exactly
constant for long stretches, which the first flatline rule read as 66 dead sensors. Flatline
and stuck-zero rules now apply only to sensors that vary during their own baseline.

## Trade-offs

- Hand-rolled loop instead of a framework. The loop is about 350 lines and every production
  concern above is visible in it. The SDK's tool runner would have saved the dispatch code.
- Final answer as a tool call rather than structured output. It keeps the diagnosis in the
  same trace as the evidence, lets the model keep investigating if validation fails, and
  gives the loop a clean terminal condition.
- Deterministic detector, LLM only on alarm. Detection must be cheap and explainable; the
  LLM runs tens of times a day, not thousands.
- Flow meters at stage inlets. The first simulator put them at outlets, which made the
  clogged stage alarm first and the problem trivial. Inlet metering is what real lines
  usually have and is what produces the misleading downstream symptom.
- Asymmetric sensor noise. Pressure and power sensors are noisier than flow sensors, so a
  root-cause signal often crosses the alarm threshold after a quieter downstream signal.
  Ordering alone does not solve the problem.
- Sonnet at medium effort. The task is bounded and tool-heavy; model and effort are flags
  and the eval reports cost per diagnosis.
- Inline SVG dashboards. A CDN script failed to load in a sandboxed viewer during
  development; a self-contained file that opens from `file://` is worth more than hover
  tooltips. The layout follows high-performance HMI conventions (grey panels, colour only
  for abnormal states, dense tables) because the reader is a plant engineer.
- The rule-based backend speaks the SDK's `Message` type, so the whole pipeline, including
  budget exhaustion and error paths, is tested in CI without a key.

## Bugs worth recording

- The diagnosis schema's field description listed the simulator's four stage names. On the
  real line the agent named the right sensor and fault type and then wrote `feed` or
  `finish` as the stage. Stage fields are now enums built from the line's topology.
- The injected-fault eval initially scored every injection against the run's first natural
  alarm, which fires before any injection, so the agent abstained on all 20 and the numbers
  were meaningless. It now scores each injection on the injected sensor's own alarm and
  counts injections the detector never catches as misses.
- The first flatline rule flagged 66 healthy but constant channels on the real line (above).

## Limitations

- The simulator suite is close to saturated for the LLM agent, so it no longer separates
  prompts. Harder cases are needed: compound faults, a fault during a setpoint transient,
  smaller severities, shorter grace windows, and scenarios where the honest answer is
  `insufficient_evidence`.
- Each eval number is one run of a sampled model; seeds 0 and 1 agree but there is no
  confidence interval.
- The simulator's physics are abstract. Real lines have control loops that mask symptoms,
  and on real data only sensor faults can be labelled.
- Out of scope by the brief: alert routing, manuals, plant integrations.
