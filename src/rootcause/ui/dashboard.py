"""Self-contained HTML dashboards in an industrial HMI style.

Follows the high-performance HMI conventions (ISA-101) that Ignition Perspective and AVEVA
InTouch applications use: flat grey panels with title bars, colour reserved for abnormal
states, dense tabular data, no decoration. No server, no external scripts: charts are inline
SVG and the data is embedded as JSON.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from rootcause.agent.loop import DiagnosisRun
from rootcause.detect.detector import Detection
from rootcause.sim.simulator import Scenario
from rootcause.sim.topology import Topology, get_topology, sensor_id

CSS = """
:root{--bg:#d6d9dd;--panel:#eceef0;--panel2:#f6f7f8;--line:#b4bac1;--line2:#d0d4d9;--fg:#1c2127;
--muted:#5a6471;--accent:#2b5f9e;--alarm:#b86e00;--alarm-bg:#f3e6cf;--fault:#b0261c;--fault-bg:#f1d9d6;
--ok:#2e7d32;--normal:#4e5964;--band:rgba(0,0,0,.075);--window:rgba(184,110,0,.16)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1f2327;--panel:#2a2f35;--panel2:#30363d;
--line:#4a525b;--line2:#3a4149;--fg:#e3e6e9;--muted:#9aa4ae;--accent:#7fb0e6;--alarm:#e3a23a;--alarm-bg:#4a3a1c;
--fault:#e06b62;--fault-bg:#4c2724;--ok:#7ccb80;--normal:#aab3bc;--band:rgba(255,255,255,.08);--window:rgba(227,162,58,.2)}}
:root[data-theme=dark]{--bg:#1f2327;--panel:#2a2f35;--panel2:#30363d;--line:#4a525b;--line2:#3a4149;--fg:#e3e6e9;
--muted:#9aa4ae;--accent:#7fb0e6;--alarm:#e3a23a;--alarm-bg:#4a3a1c;--fault:#e06b62;--fault-bg:#4c2724;--ok:#7ccb80;
--normal:#aab3bc;--band:rgba(255,255,255,.08);--window:rgba(227,162,58,.2)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 "Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.mono{font-family:"Roboto Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1500px;margin:0 auto;padding:12px 16px 24px}
.bar{display:flex;align-items:center;gap:18px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);
padding:8px 12px;margin-bottom:10px}
.bar .title{font-size:14px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.bar .meta{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px}
.bar .meta b{color:var(--fg);font-weight:600}
.bar .state{margin-left:auto;padding:3px 10px;font-weight:700;letter-spacing:.06em;font-size:12px;border:1px solid}
.state-fault{color:var(--fault);background:var(--fault-bg);border-color:var(--fault)}
.state-ok{color:var(--ok);border-color:var(--ok)}
.state-warn{color:var(--alarm);background:var(--alarm-bg);border-color:var(--alarm)}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(360px,2fr);gap:10px}
@media(max-width:1050px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);margin-bottom:10px}
.panel-title{display:flex;align-items:center;gap:10px;background:var(--panel2);border-bottom:1px solid var(--line);
padding:5px 10px;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.panel-title .no{color:var(--muted);font-weight:600}
.panel-title .status{margin-left:auto;font-size:11px;letter-spacing:.06em}
.panel-title .status.alarm{color:var(--alarm)}
.panel-title .status.root{color:var(--fault)}
.panel.root .panel-title{border-left:4px solid var(--fault)}
.panel-body{padding:8px 10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:4px 8px;border-bottom:1px solid var(--line2);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.05em;text-transform:uppercase;background:var(--panel2)}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right}
.hi,.lo,.abn{color:var(--alarm);font-weight:600}
.root-txt{color:var(--fault);font-weight:600}
.kv{display:grid;grid-template-columns:120px 1fr;gap:3px 10px}
.kv b{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.05em;text-transform:uppercase;padding-top:2px}
.kv span{font-size:13px}
.conf{display:flex;align-items:center;gap:8px}
.conf .track{flex:1;height:8px;background:var(--band);border:1px solid var(--line2)}
.conf .fill{display:block;height:100%;background:var(--normal)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--muted);padding:6px 10px;border-bottom:1px solid var(--line2)}
.sw{display:inline-block;width:14px;height:10px;vertical-align:-1px;margin-right:5px;border:1px solid var(--line)}
.charts{display:grid;gap:8px;padding:8px 10px}
.chart{background:var(--panel2);border:1px solid var(--line2)}
.chart-head{display:flex;align-items:baseline;gap:8px;padding:4px 8px 0;flex-wrap:wrap}
.chart-name{font-size:13px;font-weight:700}
.chart-unit{font-size:11px;color:var(--muted)}
.chart-val{margin-left:auto;font-size:13px;font-weight:600;white-space:nowrap}
.chart-base{font-size:11px;color:var(--muted)}
.chart-state{font-size:11px;font-weight:700;letter-spacing:.05em;min-width:26px;text-align:right}
p{margin:6px 0}
.note{border-left:3px solid var(--normal);background:var(--panel2);padding:6px 10px;margin:6px 0}
details summary{cursor:pointer;color:var(--accent);font-size:12px}
.trace{font:11.5px/1.4 "Roboto Mono",Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word;
background:var(--panel2);border:1px solid var(--line2);padding:8px;max-height:420px;overflow:auto;margin-top:6px}
.muted{color:var(--muted)}a{color:var(--accent)}
h1{font-size:15px;margin:0 0 8px;letter-spacing:.06em;text-transform:uppercase}
"""

_STATE_LABEL = {
    "high": "HI",
    "low": "LO",
    "missing": "BAD",
    "flatline": "FLAT",
    "stuck_zero": "ZERO",
}


def _view(sc: Any) -> tuple[Topology, list, int]:
    """Duck-type simulated and real scenarios: (topology, events, baseline_ticks)."""
    if isinstance(sc, Scenario):
        return get_topology(sc.params), list(sc.spec.events), sc.spec.baseline_ticks
    return sc.topology, list(getattr(sc, "events", []) or []), sc.baseline_ticks


def _pretty(sensor: str) -> str:
    """'flow' -> 'Flow', 'MaterialPressure' -> 'Material Pressure', 'Measurement0' unchanged."""
    out = []
    for i, ch in enumerate(sensor):
        if i and ch.isupper() and sensor[i - 1].islower():
            out.append(" ")
        out.append(ch)
    s = "".join(out)
    return s[:1].upper() + s[1:]


def render_dashboard(sc: Any, det: Detection, run: DiagnosisRun, path: Path) -> Path:
    d = run.diagnosis
    assert d is not None
    t_now = run.t_now
    first = det.first_alarm
    alarm_tick = first.alarm_tick if first else None
    truth = sc.truth
    topo, events, baseline_ticks = _view(sc)
    e = html.escape

    series = {}
    for s in topo.stages:
        for m in topo.sensors[s]:
            sid = sensor_id(s, m)
            vals = sc.data[sid].tolist()
            series[sid] = [None if (v != v) else round(v, 3) for v in vals]
    payload = {
        "t": sc.data["t"].tolist(),
        "series": series,
        "stages": topo.stages,
        "sensors": topo.sensors,
        "units": topo.units,
        "pretty": {m: _pretty(m) for m in topo.all_sensor_names()},
        "baselines": {k: {"mean": v.mean, "std": v.std} for k, v in det.baselines.items()},
        "alarms": [a.model_dump() for a in det.alarms],
        "alarm_tick": alarm_tick,
        "t_now": t_now,
        "onset": truth.onset_tick,
        "baseline_ticks": baseline_ticks,
        "root_stage": d.root_cause_stage if d.status == "fault_diagnosed" else None,
        "state_label": _STATE_LABEL,
    }

    # --- header state ---------------------------------------------------------------
    state_cls, state_txt = {
        "fault_diagnosed": (
            "state-fault",
            f"FAULT: {d.fault_type.upper()} AT {(d.root_cause_stage or '-').upper()}",
        ),
        "no_fault": ("state-ok", "NO FAULT"),
        "insufficient_evidence": ("state-warn", "INSUFFICIENT EVIDENCE"),
    }[d.status]

    # --- alarm summary --------------------------------------------------------------
    confirmed = sorted(
        [a for a in det.alarms if a.alarm_tick <= t_now],
        key=lambda a: (a.alarm_tick, a.onset_tick),
    )
    alarm_rows = (
        "".join(
            f"<tr><td class='num mono'>{i}</td><td class='mono'>{e(a.sensor)}</td>"
            f"<td class='abn'>{_STATE_LABEL.get(a.kind, a.kind.upper())}</td>"
            f"<td class='num mono'>{a.onset_tick}</td><td class='num mono'>{a.alarm_tick}</td>"
            f"<td class='num mono'>{'' if a.z_at_alarm is None else f'{a.z_at_alarm:+.1f}'}</td></tr>"
            for i, a in enumerate(confirmed, 1)
        )
        or "<tr><td colspan=6 class='muted'>No alarms.</td></tr>"
    )

    # --- diagnosis tables -----------------------------------------------------------
    ev_rows = (
        "".join(
            f"<tr><td class='num mono'>{i}</td><td class='mono'>{e(x.signal)}</td>"
            f"<td>{e(x.observation)}</td><td class='muted'>{e(x.why_it_matters)}</td></tr>"
            for i, x in enumerate(d.evidence, 1)
        )
        or "<tr><td colspan=4 class='muted'>No evidence recorded.</td></tr>"
    )
    alt_rows = (
        "".join(
            f"<tr><td>{e(h.fault_type)} at {e(h.root_cause_stage or '-')}</td>"
            f"<td class='num mono'>{h.confidence:.2f}</td><td class='muted'>{e(h.why_less_likely)}</td></tr>"
            for h in d.alternative_hypotheses
        )
        or "<tr><td colspan=3 class='muted'>None recorded.</td></tr>"
    )
    event_rows = (
        "".join(
            f"<tr><td class='num mono'>{ev.tick}</td><td>{e(ev.kind)}</td><td>{e(ev.stage or '-')}</td>"
            f"<td>{e(ev.description)}</td></tr>"
            for ev in events
        )
        or "<tr><td colspan=4 class='muted'>No events logged.</td></tr>"
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
    trace_html = e("\n".join(trace_lines))

    correct = (
        d.root_cause_stage == truth.root_cause_stage and d.fault_type == truth.fault_type
        if truth.fault_type != "none"
        else d.status == "no_fault"
    )
    u = run.usage
    verdict = (
        f"<span class='{'state-ok' if correct else 'root-txt'}'>{'CORRECT' if correct else 'WRONG'}</span>"
        + ("  <span class='abn'>MISLEADING ALARM</span>" if truth.misleading else "")
    )
    first_txt = (
        f"<b class='mono'>{e(first.sensor)}</b> {_STATE_LABEL.get(first.kind, '')} at t={alarm_tick}"
        if first
        else "-"
    )
    status_cls = "root-txt" if d.status == "fault_diagnosed" else ""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diagnosis {e(sc.scenario_id)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<div class="bar">
  <div class="title">Root-cause diagnosis</div>
  <div class="meta">
    <span>Line <b>{e(topo.name)}</b></span>
    <span>Scenario <b class="mono">{e(sc.scenario_id)}</b></span>
    <span>First alarm {first_txt}</span>
    <span>Diagnosed at <b class="mono">t={t_now}</b></span>
    <span>Model <b class="mono">{e(run.model)}</b></span>
  </div>
  <div class="state {state_cls}">{e(state_txt)}</div>
</div>
<div class="grid">
<div>
  <div class="panel"><div class="panel-title">Alarm summary <span class="status">{len(confirmed)} active</span></div>
    <table><tr><th class="num">#</th><th>Tag</th><th>State</th><th class="num">Onset</th><th class="num">Confirmed</th><th class="num">z</th></tr>{alarm_rows}</table></div>
  <div class="legend"><span><i class="sw" style="background:var(--band)"></i>Baseline band (mean ± 3σ)</span>
    <span><i class="sw" style="background:var(--window)"></i>Alarm to diagnosis window</span>
    <span><i class="sw" style="border:0;border-top:2px dashed var(--muted);height:0;vertical-align:3px"></i>True fault onset (simulation only)</span>
    <span>Red trace: diagnosed root-cause stage</span></div>
  <div id="stages"></div>
  <div class="panel"><div class="panel-title">Event log</div>
    <table><tr><th class="num">t</th><th>Kind</th><th>Stage</th><th>Description</th></tr>{event_rows}</table></div>
</div>
<div>
  <div class="panel"><div class="panel-title">Diagnosis</div><div class="panel-body">
    <div class="kv">
      <b>Status</b><span class="{status_cls}">{e(d.status.replace("_", " ").upper())}</span>
      <b>Root cause</b><span><strong>{e(d.root_cause_stage or "-")}</strong></span>
      <b>Fault type</b><span><strong>{e(d.fault_type)}</strong></span>
      <b>Sensor</b><span class="mono">{e(d.affected_sensor or "-")}</span>
      <b>Confidence</b><span class="conf"><span class="mono">{d.confidence:.2f}</span><span class="track"><i class="fill" style="width:{d.confidence * 100:.0f}%"></i></span></span>
      <b>Line stop</b><span>{"REQUIRED" if d.requires_downtime else "Not required"}</span>
    </div>
    <p style="margin-top:10px">{e(d.reasoning_summary)}</p>
  </div></div>
  <div class="panel"><div class="panel-title">Evidence</div>
    <table><tr><th class="num">#</th><th>Tag</th><th>Observation</th><th>Interpretation</th></tr>{ev_rows}</table></div>
  <div class="panel"><div class="panel-title">Alternatives considered</div>
    <table><tr><th>Hypothesis</th><th class="num">P</th><th>Why less likely</th></tr>{alt_rows}</table></div>
  <div class="panel"><div class="panel-title">Recommended action</div><div class="panel-body">
    <div class="note">{e(d.recommended_action)}</div></div></div>
  <div class="panel"><div class="panel-title">Simulation reference</div><div class="panel-body">
    <details open><summary>Ground truth (hidden from the agent)</summary>
    <p>{e(truth.description)}</p><p>Verdict: {verdict}</p></details></div></div>
  <div class="panel"><div class="panel-title">Agent run</div>
    <table>
      <tr><th>Status</th><td>{e(run.status)}</td><th>Model calls</th><td class="num mono">{u.model_calls}</td></tr>
      <tr><th>Tool calls</th><td class="num mono">{u.tool_calls}</td><th>Latency</th><td class="num mono">{u.latency_s} s</td></tr>
      <tr><th>Tokens in</th><td class="num mono">{u.input_tokens} (+{u.cache_read_input_tokens} cached)</td><th>Tokens out</th><td class="num mono">{u.output_tokens}</td></tr>
      <tr><th>Cost</th><td class="num mono">${u.cost_usd:.4f}</td><th></th><td></td></tr>
    </table>
    <div class="panel-body">{f'<p class="root-txt">Error: {e(run.error)}</p>' if run.error else ""}
    <details><summary>Agent trace</summary><div class="trace">{trace_html}</div></details></div></div>
</div></div></div>
<script>
const D = {json.dumps(payload)};
const root = document.getElementById('stages');
const W = 320, H = 130, L = 46, R = 8, T = 6, B = 18;
const fmt = v => Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2);
const esc = t => String(t).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function chart(sid, isRoot) {{
  const ys = D.series[sid], xs = D.t, n = xs.length;
  const b = D.baselines[sid] || {{mean: 0, std: 0}};
  const vals = ys.filter(v => v !== null);
  let lo = Math.min(...vals, b.mean - 3 * b.std), hi = Math.max(...vals, b.mean + 3 * b.std);
  if (hi - lo < 1e-9) {{ lo -= 1; hi += 1; }}
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const sx = x => L + (x / (n - 1)) * (W - L - R), sy = y => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
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
  g += `<rect x="${{sx(0)}}" y="${{sy(b.mean + 3 * b.std)}}" width="${{sx(n - 1) - sx(0)}}" height="${{Math.max(0, sy(b.mean - 3 * b.std) - sy(b.mean + 3 * b.std))}}" fill="var(--band)"/>`;
  if (D.alarm_tick != null) g += `<rect x="${{sx(D.alarm_tick)}}" y="${{T}}" width="${{sx(D.t_now) - sx(D.alarm_tick)}}" height="${{H - T - B}}" fill="var(--window)"/>`;
  for (const [a, z] of nanRuns) g += `<rect x="${{sx(xs[a])}}" y="${{T}}" width="${{Math.max(1, sx(xs[z]) - sx(xs[a]))}}" height="${{H - T - B}}" fill="var(--alarm-bg)"/>`;
  g += `<line x1="${{sx(0)}}" x2="${{sx(n - 1)}}" y1="${{sy(b.mean)}}" y2="${{sy(b.mean)}}" stroke="var(--line)" stroke-width="0.6"/>`;
  if (D.onset != null) g += `<line x1="${{sx(D.onset)}}" x2="${{sx(D.onset)}}" y1="${{T}}" y2="${{H - B}}" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="0.8"/>`;
  g += `<line x1="${{sx(D.t_now)}}" x2="${{sx(D.t_now)}}" y1="${{T}}" y2="${{H - B}}" stroke="currentColor" stroke-width="0.8"/>`;
  g += `<path d="${{path}}" fill="none" stroke="${{isRoot ? 'var(--fault)' : 'var(--normal)'}}" stroke-width="1.1" vector-effect="non-scaling-stroke"/>`;
  for (const [yv, lbl] of [[hi - pad, fmt(hi - pad)], [lo + pad, fmt(lo + pad)], [b.mean, fmt(b.mean)]])
    g += `<text x="${{L - 4}}" y="${{sy(yv) + 3}}" font-size="8.5" font-family="Menlo,Consolas,monospace" text-anchor="end" fill="var(--muted)">${{lbl}}</text>`;
  for (const xv of [0, Math.round(n / 2), n - 1])
    g += `<text x="${{sx(xv)}}" y="${{H - 5}}" font-size="8.5" font-family="Menlo,Consolas,monospace" text-anchor="middle" fill="var(--muted)">${{xv}}</text>`;
  return g + '</svg>';
}}
function card(stage, m, isRoot) {{
  const sid = stage + '.' + m, alarm = D.alarms.find(a => a.sensor === sid && a.alarm_tick <= D.t_now);
  const b = D.baselines[sid] || {{mean: NaN}};
  let cur = null; for (let i = D.t_now; i >= 0; i--) {{ if (D.series[sid][i] !== null) {{ cur = D.series[sid][i]; break; }} }}
  const unit = D.units[m] ? `<span class="chart-unit">${{esc(D.units[m])}}</span>` : '';
  const state = alarm ? `<span class="chart-state abn">${{D.state_label[alarm.kind] || alarm.kind}}</span>` : '<span class="chart-state"></span>';
  return `<div class="chart"><div class="chart-head"><span class="chart-name">${{esc(D.pretty[m] || m)}}</span>${{unit}}
    <span class="chart-val mono">${{cur === null ? 'BAD' : fmt(cur)}} <span class="chart-base">/ ${{isNaN(b.mean) ? '-' : fmt(b.mean)}}</span></span>${{state}}</div>${{chart(sid, isRoot)}}</div>`;
}}
D.stages.forEach((stage, i) => {{
  const isRoot = stage === D.root_stage;
  const nAlarms = D.alarms.filter(a => a.stage === stage && a.alarm_tick <= D.t_now).length;
  const panel = document.createElement('div'); panel.className = 'panel' + (isRoot ? ' root' : '');
  const status = isRoot ? `<span class="status root">Root cause${{nAlarms ? ' · ' + nAlarms + ' alarm' + (nAlarms > 1 ? 's' : '') : ''}}</span>`
    : (nAlarms ? `<span class="status alarm">${{nAlarms}} alarm${{nAlarms > 1 ? 's' : ''}}</span>` : '<span class="status muted">Normal</span>');
  panel.innerHTML = `<div class="panel-title"><span class="no">Stage ${{i + 1}}</span>${{esc(stage)}}${{status}}</div>`;
  const grid = document.createElement('div'); grid.className = 'charts';
  const nS = D.sensors[stage].length;
  grid.style.gridTemplateColumns = innerWidth < 720 ? '1fr' : (nS <= 3 ? 'repeat(3,minmax(0,1fr))' : 'repeat(auto-fill,minmax(230px,1fr))');
  for (const m of D.sensors[stage]) {{ const d = document.createElement('div'); d.innerHTML = card(stage, m, isRoot); grid.appendChild(d.firstChild); }}
  panel.appendChild(grid); root.appendChild(panel);
}});
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path


def render_eval_index(summary, results, path: Path) -> Path:
    """Index page for an eval run: summary table plus links to each scenario dashboard."""
    from rootcause.evals.report import label

    e = html.escape
    names = list(summary.systems.keys())
    head = "".join(f"<th>{e(label(n))}</th>" for n in names)
    rows = []
    for r in results:
        if not r.detector_triggered:
            continue
        cells = []
        link = None
        for n in names:
            s = r.systems.get(n)
            if not s:
                cells.append("<td>-</td>")
                continue
            p = s.prediction
            if s.score.fully_correct:
                mark = "<span class='state-ok'>OK</span>"
            elif s.score.abstained:
                mark = "<span class='abn'>ABSTAIN</span>"
            else:
                mark = "<span class='root-txt'>WRONG</span>"
            what = (
                f"{p.fault_type} at {p.root_cause_stage or '-'}"
                if p.status == "fault_diagnosed"
                else p.status.replace("_", " ")
            )
            cells.append(
                f"<td>{mark} {e(what)} <span class='muted mono'>{p.confidence:.2f}</span></td>"
            )
            if s.dashboard_path:
                link = Path(s.dashboard_path).name
        sid = e(r.scenario_id)
        sid_cell = f"<a href='dashboards/{link}'>{sid}</a>" if link else sid
        truth = f"{r.truth.fault_type} at {r.truth.root_cause_stage or '-'}"
        flag = " <span class='abn'>MISLEADING</span>" if r.misleading else ""
        rows.append(
            f"<tr><td class='mono'>{sid_cell}</td><td>{e(truth)}</td>"
            f"<td><span class='mono'>{e(r.first_alarm_sensor or '-')}</span>{flag}</td>{''.join(cells)}</tr>"
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
            f"<td class='num mono'>{summary.systems[n][key] if summary.systems[n][key] is not None else '-'}{suffix}</td>"
            for n in names
        )
        metric_rows.append(f"<tr><td>{title}</td>{cells}</tr>")
    brier = "".join(
        f"<td class='num mono'>{summary.systems[n]['calibration']['brier']}</td>" for n in names
    )
    metric_rows.append(f"<tr><td>Brier score</td>{brier}</tr>")
    det_rows = "".join(
        f"<tr><td>{e(k.replace('_', ' '))}</td><td class='num mono'>{v}</td></tr>"
        for k, v in summary.detector.items()
    )
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval run</title><style>{CSS}</style></head><body><div class="wrap">
<div class="bar"><div class="title">Eval run</div><div class="meta"><span><b>{summary.n_scenarios}</b> scenarios</span><span><b>{summary.n_triggered}</b> alarms</span></div></div>
<div class="panel"><div class="panel-title">Diagnosis metrics</div><table><tr><th>Metric</th>{head}</tr>{"".join(metric_rows)}</table></div>
<div class="panel"><div class="panel-title">Detector</div><table>{det_rows}</table></div>
<div class="panel"><div class="panel-title">Scenarios</div><table><tr><th>Scenario</th><th>Truth</th><th>First alarm</th>{head}</tr>{"".join(rows)}</table></div>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path
