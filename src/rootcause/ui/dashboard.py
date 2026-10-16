"""Self-contained HTML dashboards: sensor traces with the anomaly window, next to the diagnosis.

No server, no external scripts: charts are inline SVG and the data is embedded as JSON.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from rootcause.agent.loop import DiagnosisRun
from rootcause.detect.detector import Detection
from rootcause.sim.simulator import Scenario
from rootcause.sim.topology import SENSORS, STAGES, sensor_id

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1a1d21;--muted:#5f6b7a;--line:#e3e6ea;--accent:#2563eb;
--red:#c0392b;--green:#1e8e3e;--amber:#b7791f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1115;--card:#171a20;--fg:#e8eaed;
--muted:#9aa4b2;--line:#2a2f37;--accent:#60a5fa;--red:#f87171;--green:#4ade80;--amber:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(320px,2fr);gap:16px}
@media(max-width:1000px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid}
.b-red{color:var(--red);border-color:var(--red)}.b-green{color:var(--green);border-color:var(--green)}
.b-amber{color:var(--amber);border-color:var(--amber)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:8px 0}.kv b{color:var(--muted);font-weight:500}
.conf{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.conf>i{display:block;height:100%;background:var(--accent)}
ol.ev{padding-left:20px;margin:8px 0}ol.ev li{margin-bottom:8px}ol.ev code{color:var(--accent);font-size:13px}
.muted{color:var(--muted)}details{margin-top:8px}summary{cursor:pointer;color:var(--accent)}
.trace{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;
background:var(--bg);border-radius:6px;padding:8px;max-height:400px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:600}a{color:var(--accent)}
.legend span{display:inline-block;margin-right:12px;font-size:12px;color:var(--muted)}
.sw{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:4px;border-radius:2px}
"""


def _badge(status: str) -> str:
    cls = {"fault_diagnosed": "b-red", "no_fault": "b-green", "insufficient_evidence": "b-amber"}[
        status
    ]
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def render_dashboard(sc: Scenario, det: Detection, run: DiagnosisRun, path: Path) -> Path:
    d = run.diagnosis
    assert d is not None
    t_now = run.t_now
    first = det.first_alarm
    alarm_tick = first.alarm_tick if first else None
    truth = sc.truth

    series = {}
    for s in STAGES:
        for m in SENSORS[s]:
            sid = sensor_id(s, m)
            vals = sc.data[sid].tolist()
            series[sid] = [None if (v != v) else round(v, 3) for v in vals]
    payload = {
        "t": sc.data["t"].tolist(),
        "series": series,
        "stages": STAGES,
        "sensors": SENSORS,
        "baselines": {k: {"mean": v.mean, "std": v.std} for k, v in det.baselines.items()},
        "alarms": [a.model_dump() for a in det.alarms],
        "alarm_tick": alarm_tick,
        "t_now": t_now,
        "onset": truth.onset_tick,
        "baseline_ticks": sc.spec.baseline_ticks,
        "root_stage": d.root_cause_stage,
        "truth_stage": truth.root_cause_stage,
        "events": [e.model_dump() for e in sc.spec.events],
    }

    ev_html = (
        "".join(
            f"<li><code>{html.escape(e.signal)}</code> — {html.escape(e.observation)}"
            f"<div class='muted'>{html.escape(e.why_it_matters)}</div></li>"
            for e in d.evidence
        )
        or "<li class='muted'>No evidence recorded.</li>"
    )
    alt_html = (
        "".join(
            f"<li><b>{html.escape(h.fault_type)}</b> at {html.escape(h.root_cause_stage or '-')} "
            f"<span class='muted'>({h.confidence:.2f})</span> — {html.escape(h.why_less_likely)}</li>"
            for h in d.alternative_hypotheses
        )
        or "<li class='muted'>None recorded.</li>"
    )
    trace_lines = []
    for s in run.trace:
        if s.kind == "model_call":
            trace_lines.append(
                f"[{s.step}] model_call stop={s.stop_reason} {s.latency_s}s tokens={s.usage}"
            )
            if s.text:
                trace_lines.append("    " + s.text.replace("\n", "\n    "))
        elif s.kind == "tool_call":
            trace_lines.append(
                f"[{s.step}] {'ERROR ' if s.is_error else ''}{s.tool_name}({json.dumps(s.tool_input)})"
            )
            trace_lines.append("    -> " + (s.tool_output or "")[:600])
        else:
            trace_lines.append(f"[{s.step}] {s.kind}: {s.text}")
    trace_html = html.escape("\n".join(trace_lines))
    correct = (
        d.root_cause_stage == truth.root_cause_stage and d.fault_type == truth.fault_type
        if truth.fault_type != "none"
        else d.status == "no_fault"
    )
    u = run.usage
    events_rows = (
        "".join(
            f"<tr><td>{e.tick}</td><td>{html.escape(e.kind)}</td><td>{html.escape(e.stage or '-')}</td><td>{html.escape(e.description)}</td></tr>"
            for e in sc.spec.events
        )
        or "<tr><td colspan=4 class='muted'>No events logged.</td></tr>"
    )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diagnosis {html.escape(sc.scenario_id)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>Root-cause diagnosis — {html.escape(sc.scenario_id)}</h1>
<div class="muted">Alarm at t={alarm_tick} on <code>{html.escape(first.sensor if first else "-")}</code>; diagnosis requested at t={t_now}. Model: {html.escape(run.model)}.</div>
<div class="grid" style="margin-top:12px">
<div>
 <div class="card"><h2>Sensor traces</h2>
 <div class="legend"><span><i class="sw" style="background:rgba(37,99,235,.15)"></i>baseline ±3σ</span>
 <span><i class="sw" style="background:rgba(192,57,43,.18)"></i>alarm → diagnosis window</span>
 <span><i class="sw" style="border-top:2px dashed #888;height:0;vertical-align:3px"></i>true fault onset (hidden from the agent)</span></div>
 <div id="plots"></div></div>
 <div class="card"><h2>Operator event log</h2><table><tr><th>t</th><th>Kind</th><th>Stage</th><th>Description</th></tr>{events_rows}</table></div>
</div>
<div>
 <div class="card"><h2>Diagnosis</h2>
 {_badge(d.status)}
 <div class="kv"><b>Root cause</b><span><strong>{html.escape(d.root_cause_stage or "-")}</strong></span>
 <b>Fault type</b><span><strong>{html.escape(d.fault_type)}</strong>{(" · " + html.escape(d.affected_sensor)) if d.affected_sensor else ""}</span>
 <b>Confidence</b><span>{d.confidence:.2f}<div class="conf"><i style="width:{d.confidence * 100:.0f}%"></i></div></span>
 <b>Downtime</b><span>{"required" if d.requires_downtime else "not required"}</span></div>
 <p>{html.escape(d.reasoning_summary)}</p>
 <h2>Evidence chain</h2><ol class="ev">{ev_html}</ol>
 <h2>Alternatives considered</h2><ul>{alt_html}</ul>
 <h2>Recommended action</h2><p>{html.escape(d.recommended_action)}</p>
 </div>
 <div class="card"><h2>Ground truth</h2>
 <details open><summary>Reveal</summary><p>{html.escape(truth.description)}</p>
 <p>Verdict: <span class="badge {"b-green" if correct else "b-red"}">{"correct" if correct else "wrong"}</span>
 {'<span class="badge b-amber">misleading alarm</span>' if truth.misleading else ""}</p></details></div>
 <div class="card"><h2>Run</h2>
 <div class="kv"><b>Status</b><span>{html.escape(run.status)}</span><b>Model calls</b><span>{u.model_calls}</span>
 <b>Tool calls</b><span>{u.tool_calls}</span><b>Tokens</b><span>{u.input_tokens} in ({u.cache_read_input_tokens} cached) / {u.output_tokens} out</span>
 <b>Cost</b><span>${u.cost_usd:.4f}</span><b>Latency</b><span>{u.latency_s}s</span></div>
 {f'<p class="muted">Error: {html.escape(run.error)}</p>' if run.error else ""}
 <details><summary>Agent trace</summary><div class="trace">{trace_html}</div></details></div>
</div></div></div>
<script>
const D = {json.dumps(payload)};
const root = document.getElementById('plots');
const W = 320, H = 170, L = 44, R = 8, T = 22, B = 20;
const fmt = v => Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2);
function chart(sid, stage) {{
  const ys = D.series[sid], xs = D.t, n = xs.length;
  const b = D.baselines[sid] || {{mean: 0, std: 0}};
  const vals = ys.filter(v => v !== null);
  let lo = Math.min(...vals, b.mean - 3 * b.std), hi = Math.max(...vals, b.mean + 3 * b.std);
  if (hi - lo < 1e-9) {{ lo -= 1; hi += 1; }}
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const sx = x => L + (x / (n - 1)) * (W - L - R), sy = y => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
  const isRoot = stage === D.root_stage;
  const alarm = D.alarms.find(a => a.sensor === sid);
  const color = isRoot ? 'var(--red)' : 'var(--accent)';
  let path = '', pen = false;
  for (let i = 0; i < n; i++) {{
    if (ys[i] === null) {{ pen = false; continue; }}
    path += (pen ? 'L' : 'M') + sx(xs[i]).toFixed(1) + ' ' + sy(ys[i]).toFixed(1) + ' ';
    pen = true;
  }}
  const nanRuns = [];
  for (let i = 0, s = -1; i <= n; i++) {{
    const isNan = i < n && ys[i] === null;
    if (isNan && s < 0) s = i; if (!isNan && s >= 0) {{ nanRuns.push([s, i - 1]); s = -1; }}
  }}
  let g = `<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" preserveAspectRatio="none" style="display:block;max-height:${{H}}px">`;
  g += `<rect x="${{sx(0)}}" y="${{T}}" width="${{sx(D.baseline_ticks) - sx(0)}}" height="${{H - T - B}}" fill="rgba(37,99,235,.06)"/>`;
  g += `<rect x="${{sx(0)}}" y="${{sy(b.mean + 3 * b.std)}}" width="${{sx(n - 1) - sx(0)}}" height="${{Math.max(0, sy(b.mean - 3 * b.std) - sy(b.mean + 3 * b.std))}}" fill="rgba(37,99,235,.15)"/>`;
  if (D.alarm_tick != null) g += `<rect x="${{sx(D.alarm_tick)}}" y="${{T}}" width="${{sx(D.t_now) - sx(D.alarm_tick)}}" height="${{H - T - B}}" fill="rgba(192,57,43,.18)"/>`;
  for (const [a, z] of nanRuns) g += `<rect x="${{sx(xs[a])}}" y="${{T}}" width="${{Math.max(1, sx(xs[z]) - sx(xs[a]))}}" height="${{H - T - B}}" fill="rgba(183,121,31,.25)"/>`;
  if (D.onset != null) g += `<line x1="${{sx(D.onset)}}" x2="${{sx(D.onset)}}" y1="${{T}}" y2="${{H - B}}" stroke="#888" stroke-dasharray="4 3"/>`;
  g += `<line x1="${{sx(D.t_now)}}" x2="${{sx(D.t_now)}}" y1="${{T}}" y2="${{H - B}}" stroke="currentColor" stroke-width="1"/>`;
  g += `<path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="1.2" vector-effect="non-scaling-stroke"/>`;
  for (const [yv, lbl] of [[hi - pad, fmt(hi - pad)], [lo + pad, fmt(lo + pad)], [b.mean, fmt(b.mean)]])
    g += `<text x="${{L - 4}}" y="${{sy(yv) + 3}}" font-size="9" text-anchor="end" fill="var(--muted)">${{lbl}}</text>`;
  for (const xv of [0, Math.round(n / 2), n - 1])
    g += `<text x="${{sx(xv)}}" y="${{H - 6}}" font-size="9" text-anchor="middle" fill="var(--muted)">t=${{xv}}</text>`;
  const title = sid + (alarm ? '  ⚠ ' + alarm.kind + ' @' + alarm.alarm_tick : '') + (isRoot ? '  ◀ root cause' : '');
  g += `<text x="${{L}}" y="14" font-size="11" font-weight="600" fill="${{isRoot ? 'var(--red)' : 'currentColor'}}">${{title}}</text></svg>`;
  return g;
}}
for (const stage of D.stages) {{
  const row = document.createElement('div');
  row.style.display = 'grid'; row.style.gridTemplateColumns = innerWidth < 700 ? '1fr' : 'repeat(3,minmax(0,1fr))'; row.style.gap = '8px'; row.style.marginBottom = '8px';
  for (const m of D.sensors[stage]) {{
    const div = document.createElement('div'); div.innerHTML = chart(stage + '.' + m, stage); row.appendChild(div);
  }}
  root.appendChild(row);
}}
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path


def render_eval_index(summary, results, path: Path) -> Path:
    """Index page for an eval run: summary table plus links to each scenario dashboard."""
    from rootcause.evals.report import label

    names = list(summary.systems.keys())
    head = "".join(f"<th>{html.escape(label(n))}</th>" for n in names)
    rows = []
    for r in results:
        if not r.detector_triggered:
            continue
        cells = []
        link = None
        for n in names:
            s = r.systems.get(n)
            if not s:
                cells.append("<td>–</td>")
                continue
            p = s.prediction
            mark = "✅" if s.score.fully_correct else ("⚠️" if s.score.abstained else "❌")
            what = (
                f"{p.fault_type}@{p.root_cause_stage or '-'}"
                if p.status == "fault_diagnosed"
                else p.status
            )
            cells.append(
                f"<td>{mark} {html.escape(what)} <span class='muted'>({p.confidence:.2f})</span></td>"
            )
            if s.dashboard_path:
                link = Path(s.dashboard_path).name
        sid = html.escape(r.scenario_id)
        sid_cell = f"<a href='dashboards/{link}'>{sid}</a>" if link else sid
        truth = f"{r.truth.fault_type}@{r.truth.root_cause_stage or '-'}"
        flag = " <span class='badge b-amber'>misleading</span>" if r.misleading else ""
        rows.append(
            f"<tr><td>{sid_cell}</td><td>{html.escape(truth)}</td><td><code>{html.escape(r.first_alarm_sensor or '-')}</code>{flag}</td>{''.join(cells)}</tr>"
        )
    metric_rows = []
    for title, key, suffix in [
        ("Root-cause stage accuracy", "root_cause_stage_acc_pct", "%"),
        ("Fault-type accuracy", "fault_type_acc_pct", "%"),
        ("Both correct", "fully_correct_pct", "%"),
        ("Both correct, misleading only", "fully_correct_misleading_pct", "%"),
        ("Abstained", "abstention_pct", "%"),
        ("Confident errors", "confident_error_pct", "%"),
        ("False positives on no-fault alarms", "false_positive_pct_on_no_fault", "%"),
        ("Mean tool calls", "mean_tool_calls", ""),
        ("Mean cost (USD)", "mean_cost_usd", ""),
    ]:
        cells = "".join(
            f"<td>{summary.systems[n][key] if summary.systems[n][key] is not None else '–'}{suffix}</td>"
            for n in names
        )
        metric_rows.append(f"<tr><td>{title}</td>{cells}</tr>")
    brier = "".join(f"<td>{summary.systems[n]['calibration']['brier']}</td>" for n in names)
    metric_rows.append(f"<tr><td>Brier score</td>{brier}</tr>")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval run</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Eval run — {summary.n_scenarios} scenarios, {summary.n_triggered} alarms</h1>
<div class="card"><h2>Diagnosis metrics</h2><table><tr><th>Metric</th>{head}</tr>{"".join(metric_rows)}</table></div>
<div class="card"><h2>Detector</h2><table>{"".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in summary.detector.items())}</table></div>
<div class="card"><h2>Scenarios</h2><table><tr><th>Scenario</th><th>Truth</th><th>First alarm</th>{head}</tr>{"".join(rows)}</table></div>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path
