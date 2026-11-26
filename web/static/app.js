/* Aqua-Swachh dashboard (Stage 3 prototype)
 * Plain JS, no build step. Talks to the FastAPI backend in web/app.py.
 */
'use strict';

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const STATUS = {
  'Normal': { c: 'emerald', icon: 'fa-circle-check', headline: 'Your purifier is working well.' },
  'Monitor': { c: 'amber', icon: 'fa-eye', headline: 'Still working, but showing early signs of wear.' },
  'Service Required': { c: 'rose', icon: 'fa-triangle-exclamation', headline: 'Your purifier needs service now.' },
};
const STATUS_HEX = { 'Normal': '#10b981', 'Monitor': '#f59e0b', 'Service Required': '#f43f5e' };
const CLASSES = ['Normal', 'Monitor', 'Service Required'];

let toastTimer = null;
function toast(msg, ms = 3500) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
}

function setApiStatus(ok) {
  const el = $('apiStatus');
  el.innerHTML = ok
    ? '<span class="w-2 h-2 rounded-full bg-emerald-500"></span>API online'
    : '<span class="w-2 h-2 rounded-full bg-rose-500"></span>API offline';
  el.className = `hidden xl:inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1.5 rounded-full ${ok ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'}`;
}

async function api(path, options = {}) {
  let resp;
  try {
    resp = await fetch(path, options);
  } catch (e) {
    setApiStatus(false);
    throw new Error('Cannot reach the server. Is uvicorn running?');
  }
  setApiStatus(true);
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : body.detail.map((d) => d.msg).join('; ');
    } catch (_) { /* not JSON */ }
    throw new Error(detail);
  }
  return resp.json();
}

// ---------------------------------------------------------------- tabs
const TABS = ['live', 'batch', 'hardware', 'ml'];
function switchTab(tab) {
  if (!TABS.includes(tab)) tab = 'live';
  TABS.forEach((t) => $(`tab-${t}`).classList.toggle('hidden', t !== tab));
  if (location.hash !== `#${tab}`) history.replaceState(null, '', `#${tab}`);
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  if (tab === 'hardware') loadTelemetry();
  if (tab === 'ml') loadMetrics();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------------------------------------------------------------- input handling
const state = {
  mode: 'voltage',
  source: { kind: 'sensor', label: 'Your sensor reading' },
  seq: 0,
  realRows: [],
  realAverages: [],
  examples: [],
};
let autoTimer = null;

function setInputMode(mode) {
  state.mode = mode;
  ['voltage', 'ppm', 'examples'].forEach((m) => {
    $(`mode-${m}`).classList.toggle('active', m === mode);
    $(`panel-${m}`).classList.toggle('hidden', m !== mode);
  });
  const isExamples = mode === 'examples';
  $('flowBlock').classList.toggle('hidden', isExamples);
  $('checkRow').classList.toggle('hidden', isExamples);
  $('inputError').classList.add('hidden');
}

// Number inputs are the source of truth; range sliders only mirror them.
function syncFromRange(key) {
  $(`p${key}`).value = $(`r${key}`).value;
  onInputEdited();
}
function syncFromNumber(key) {
  const v = parseFloat($(`p${key}`).value);
  if (!Number.isNaN(v)) $(`r${key}`).value = v;
  onInputEdited();
}

function onFlowToggle() {
  const on = $('flowOn').checked;
  $('flowInputs').classList.toggle('hidden', !on);
  $('flowHelp').textContent = on
    ? 'Using the flow value below: the full model (TDS + flow) will be used.'
    : 'No flow meter fitted: the TDS-only model will be used.';
  onInputEdited();
}

function onInputEdited() {
  if (sim.running) pauseSimulation('Simulation paused because you changed a value.');
  $('kSummary').textContent = `K1=${fmt(parseFloat($('kIn').value), 2)} · K2=${fmt(parseFloat($('kOut').value), 2)}`;
  state.source = state.mode === 'voltage'
    ? { kind: 'sensor', label: 'Your sensor reading' }
    : { kind: 'manual', label: 'Entered manually' };
  if ($('autoCheck').checked) {
    clearTimeout(autoTimer);
    autoTimer = setTimeout(() => runCheck(true), 350);
  }
}

function readNum(id, lo, hi, name, errors) {
  const el = $(id);
  const v = parseFloat(el.value);
  const bad = Number.isNaN(v) || v < lo || v > hi;
  el.classList.toggle('invalid', bad);
  if (bad) errors.push(`${name} must be between ${lo} and ${hi}.`);
  return v;
}

function collectInputs() {
  const errors = [];
  const flow = $('flowOn').checked ? readNum('pFlow', 0, 10, 'Flow', errors) : null;
  if (state.mode === 'voltage') {
    const payload = {
      tds_in_voltage: readNum('vIn', 0, 3.3, 'Sensor 1 voltage', errors),
      tds_out_voltage: readNum('vOut', 0, 3.3, 'Sensor 2 voltage', errors),
      temperature: readNum('vTemp', 0, 60, 'Temperature', errors),
      flow_rate: flow,
      k_in: readNum('kIn', 0.1, 5, 'K1', errors),
      k_out: readNum('kOut', 0.1, 5, 'K2', errors),
    };
    return { errors, endpoint: '/api/predict-voltage', payload };
  }
  const payload = {
    tds_in: readNum('pTdsIn', 0, 2500, 'Feed TDS', errors),
    tds_out: readNum('pTdsOut', 0, 2500, 'Output TDS', errors),
    temperature: readNum('pTemp', 0, 60, 'Temperature', errors),
    flow_rate: flow,
  };
  return { errors, endpoint: '/api/predict', payload };
}

async function runCheck(silent = false) {
  const { errors, endpoint, payload } = collectInputs();
  const errBox = $('inputError');
  if (errors.length) {
    errBox.textContent = errors.join(' ');
    errBox.classList.remove('hidden');
    return null;
  }
  errBox.classList.add('hidden');
  if (!silent) $('checkBtn').disabled = true;
  try {
    return await diagnose(endpoint, payload, state.source);
  } finally {
    $('checkBtn').disabled = false;
  }
}

/** Calls the API and renders the result. Stale responses (an older request
 *  finishing after a newer one) are dropped, so the screen never goes backwards. */
async function diagnose(endpoint, payload, source) {
  const seq = ++state.seq;
  let res;
  try {
    res = await api(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  } catch (e) {
    toast(`Check failed: ${e.message}`);
    return null;
  }
  if (seq !== state.seq) return res; // a newer check has already been sent
  const conv = res.sensor_conversion || null;
  const reading = {
    tdsIn: conv ? conv.tds_in_ppm : payload.tds_in,
    tdsOut: conv ? conv.tds_out_ppm : payload.tds_out,
    temp: payload.temperature,
    flow: res.physics_indicators.flow_rate_lpm,
    conv,
  };
  renderResult(res, reading, source);
  renderJourney(res, reading);
  renderKpis(res, reading);
  addHistory(res, reading, source);
  return res;
}

// ---------------------------------------------------------------- result rendering
function chip(cls, icon, text, title = '') {
  return `<span class="inline-flex items-center gap-1.5 text-[11px] font-bold px-2.5 py-1 rounded-full border ${cls}" title="${esc(title)}"><i class="fa-solid ${icon} text-[10px]"></i>${esc(text)}</span>`;
}

function outputQuality(tdsOut) {
  if (tdsOut < 150) return { c: 'emerald', text: 'Output TDS looks good', level: 'ok' };
  if (tdsOut < 250) return { c: 'amber', text: 'Output TDS is high-ish', level: 'warn' };
  return { c: 'rose', text: 'Output TDS too high', level: 'bad' };
}

function renderResult(res, r, source) {
  const st = STATUS[res.predicted_status] || STATUS['Monitor'];
  $('resultEmpty').classList.add('hidden');
  const body = $('resultBody');
  body.classList.remove('hidden');
  body.classList.remove('fade-in'); void body.offsetWidth; body.classList.add('fade-in');

  $('resultCard').className = `xl:col-span-3 card p-6 sm:p-7 border-2 transition-colors duration-300 border-${st.c}-200 bg-gradient-to-br from-white to-${st.c}-50/40`;

  const sourceStyles = {
    sensor: ['bg-sky-50 text-sky-700 border-sky-200', 'fa-bolt'],
    manual: ['bg-slate-50 text-slate-700 border-slate-200', 'fa-keyboard'],
    example: ['bg-indigo-50 text-indigo-700 border-indigo-200', 'fa-book-open'],
    simulation: ['bg-violet-50 text-violet-700 border-violet-200', 'fa-wand-magic-sparkles'],
    batch: ['bg-sky-50 text-sky-700 border-sky-200', 'fa-table-list'],
  };
  const [scls, sicon] = sourceStyles[source.kind] || sourceStyles.manual;
  const q = outputQuality(r.tdsOut);
  let chips = chip(scls, sicon, source.label);
  chips += res.flow_available
    ? chip('bg-white text-slate-600 border-slate-200', 'fa-gauge-high', 'Model: TDS + flow')
    : chip('bg-white text-slate-600 border-slate-200', 'fa-droplet', 'Model: TDS only (no flow)');
  chips += chip(`bg-${q.c}-50 text-${q.c}-700 border-${q.c}-200`, 'fa-glass-water', q.text, 'Based on output TDS only. TDS alone does not prove water is safe to drink.');
  if (r.conv && !r.conv.calibrated) chips += chip('bg-amber-50 text-amber-700 border-amber-200', 'fa-scale-unbalanced', 'Uncalibrated probes (K = 1.0)');
  $('resultChips').innerHTML = chips;

  const statusText = $('statusText');
  statusText.textContent = res.predicted_status;
  statusText.className = `text-4xl sm:text-5xl font-extrabold tracking-tight text-${st.c}-600`;
  $('statusIcon').className = `w-12 h-12 rounded-2xl flex items-center justify-center text-xl mb-1 bg-${st.c}-100 text-${st.c}-600`;
  $('statusIcon').innerHTML = `<i class="fa-solid ${st.icon}"></i>`;
  $('statusHeadline').textContent = st.headline;
  $('actionText').textContent = res.action_recommendation;
  $('explanationText').textContent = res.explanation || '';

  const notes = [...(res.notes || [])];
  if (r.conv && !r.conv.calibrated) notes.push('Probe calibration (K) is 1.0, so ppm values are approximate. A small K change can move a borderline result.');
  $('notesBox').innerHTML = notes.map((n) => `<div><i class="fa-solid fa-circle-info mr-1.5"></i>${esc(n)}</div>`).join('');
  $('notesBox').classList.toggle('hidden', notes.length === 0);

  const g = $('guardrailBox');
  if (res.guardrail_triggered) {
    const modelP = res.probabilities[res.model_status];
    g.innerHTML = `<strong><i class="fa-solid fa-shield-halved mr-1.5"></i>Safety rule applied:</strong> ${esc(res.guardrail_reason)}.`
      + (res.model_status && res.model_status !== res.predicted_status
        ? ` The model alone said <strong>${esc(res.model_status)}</strong> (${fmt(modelP * 100)}%), but the safety rule always wins.`
        : ' The model agreed.');
    g.classList.remove('hidden');
    $('confidenceText').textContent = 'Decided by safety rule';
  } else {
    g.classList.add('hidden');
    $('confidenceText').textContent = `${fmt(res.confidence * 100)}% sure`;
  }

  $('probBars').innerHTML = CLASSES.map((cls) => {
    const p = (res.probabilities[cls] || 0) * 100;
    const c = STATUS[cls].c;
    return `<div class="flex items-center gap-3 text-xs">
      <span class="w-28 shrink-0 font-semibold text-slate-600">${cls}</span>
      <div class="flex-1 h-2.5 bg-slate-100 rounded-full overflow-hidden"><div class="h-full bg-${c}-500 rounded-full transition-all duration-500" style="width:${p}%"></div></div>
      <span class="w-12 text-right font-mono font-semibold text-${c}-700">${fmt(p)}%</span>
    </div>`;
  }).join('');
}

// ---------------------------------------------------------------- purifier journey
const LEVEL = {
  ok: { c: 'emerald', label: 'Healthy' },
  warn: { c: 'amber', label: 'Watch' },
  bad: { c: 'rose', label: 'Problem' },
  unknown: { c: 'slate', label: 'Not measured' },
  info: { c: 'sky', label: 'Input' },
};
const SEVERITY = { bad: 2, warn: 1, ok: 0, unknown: -1, info: -1 };

function renderJourney(res, r) {
  const rej = res.physics_indicators.salt_rejection_pct;
  const flow = r.flow;
  const roLevel = rej >= 88 ? 'ok' : rej >= 74 ? 'warn' : 'bad';
  const preLevel = flow === null ? 'unknown' : flow >= 1.4 ? 'ok' : flow >= 1.0 ? 'warn' : 'bad';
  const outLevel = outputQuality(r.tdsOut).level;
  const meterLevel = flow === null ? 'unknown' : preLevel;

  // Which stage most likely explains a non-normal result?
  let culprit = null;
  if (res.predicted_status !== 'Normal') {
    const cands = [['pre', preLevel], ['ro', roLevel]].filter(([, l]) => SEVERITY[l] > 0);
    if (cands.length) {
      cands.sort((a, b) => SEVERITY[b[1]] - SEVERITY[a[1]]);
      culprit = cands[0][0];
      if (cands.length === 2 && SEVERITY[cands[0][1]] === SEVERITY[cands[1][1]]) {
        culprit = /flow/i.test(res.guardrail_reason || '') ? 'pre' : 'ro';
      }
    }
  }

  const volt = (v) => (v !== undefined && v !== null ? `<span class="block text-[11px] font-semibold text-slate-400 mt-0.5">from ${fmt(v, 3)} V</span>` : '');
  const stages = [
    { id: 'feed', icon: 'fa-faucet-drip', title: 'Feed water', sub: 'TDS Sensor 1', value: `${fmt(r.tdsIn, 0)} ppm`, extra: r.conv ? volt(r.conv.tds_in_voltage) : '', level: 'info', hint: 'What goes in' },
    { id: 'pre', icon: 'fa-filter', title: 'Pre-filters', sub: 'Sediment + carbon', value: flow === null ? 'Unknown' : (preLevel === 'ok' ? 'Clear' : preLevel === 'warn' ? 'Slowing' : 'Clogged'), level: preLevel, hint: flow === null ? 'Needs a flow reading' : 'Judged from flow' },
    { id: 'ro', icon: 'fa-atom', title: 'RO membrane', sub: 'Salt rejection', value: `${fmt(rej, 1)}%`, level: roLevel, hint: 'Healthy above 88%' },
    { id: 'out', icon: 'fa-glass-water', title: 'Purified water', sub: 'TDS Sensor 2', value: `${fmt(r.tdsOut, 0)} ppm`, extra: r.conv ? volt(r.conv.tds_out_voltage) : '', level: outLevel, hint: 'Good below 150 ppm' },
    { id: 'meter', icon: 'fa-gauge-high', title: 'Flow', sub: 'YF-S201 meter', value: flow === null ? 'Not fitted' : `${fmt(flow, 2)} L/min`, level: meterLevel, hint: flow === null ? 'Optional sensor' : 'Healthy ≥ 1.4 L/min' },
  ];

  const speed = flow === null ? null : clamp(2.4 / Math.max(flow, 0.05), 0.5, 6);
  const pipe = () => {
    const cls = flow === null ? 'unknown' : flow > 0.05 ? 'flowing' : '';
    const style = speed ? `style="--flow-speed:${speed.toFixed(2)}s"` : '';
    return `<div class="hidden md:flex items-center"><div class="pipe ${cls} w-10" ${style} title="${flow === null ? 'Flow not measured' : `${fmt(flow, 2)} L/min`}"></div></div>`;
  };

  $('journey').innerHTML = stages.map((s, i) => {
    const L = LEVEL[s.level];
    const isCulprit = culprit === s.id;
    const ringHex = { emerald: '#10b981', amber: '#f59e0b', rose: '#f43f5e', slate: '#94a3b8', sky: '#0ea5e9' }[L.c];
    const card = `
      <div class="relative rounded-2xl border-2 p-4 flex flex-col gap-2 bg-white transition-all ${isCulprit ? `border-${L.c}-400 stage-culprit` : 'border-slate-100'}" ${isCulprit ? `style="--ring:${ringHex}55"` : ''}>
        ${isCulprit ? `<span class="absolute -top-2.5 left-3 text-[10px] font-bold px-2 py-0.5 rounded-full bg-${L.c}-600 text-white">Likely cause</span>` : ''}
        <div class="flex items-center justify-between">
          <span class="w-9 h-9 rounded-xl bg-${L.c}-50 text-${L.c}-600 flex items-center justify-center"><i class="fa-solid ${s.icon}"></i></span>
          <span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-${L.c}-50 text-${L.c}-700">${L.label}</span>
        </div>
        <div>
          <div class="text-sm font-bold text-slate-900">${s.title}</div>
          <div class="text-[11px] text-slate-500">${s.sub}</div>
        </div>
        <div class="text-xl font-extrabold font-mono text-slate-900 leading-tight">${s.value}${s.extra || ''}</div>
        <div class="text-[11px] text-slate-400">${s.hint}</div>
      </div>`;
    return card + (i < stages.length - 1 ? pipe() : '');
  }).join('');

  let note;
  if (res.predicted_status === 'Normal') {
    note = 'All measured stages look healthy. Nothing is highlighted because there is nothing to fix.';
  } else if (culprit === 'ro') {
    note = `The RO membrane is highlighted because salt rejection is ${fmt(rej, 1)}%, below the 88% healthy level.`;
  } else if (culprit === 'pre') {
    note = `The pre-filters are highlighted because flow has dropped to ${fmt(flow, 2)} L/min, a sign of clogging.`;
  } else {
    note = 'The model sees an early-warning pattern overall; no single measured stage stands out.';
  }
  if (flow === null) note += ' Pre-filter health can\'t be judged without a flow meter.';
  $('journeyNote').textContent = note;
}

// ---------------------------------------------------------------- KPIs
function renderKpis(res, r) {
  const p = res.physics_indicators;
  const rej = p.salt_rejection_pct;
  const rejL = rej >= 88 ? ['emerald', 'Healthy (88%+)'] : rej >= 74 ? ['amber', 'Below the 88% healthy level'] : ['rose', 'Low: membrane problem'];
  const q = outputQuality(r.tdsOut);
  const flow = r.flow;
  const flowL = flow === null ? ['slate', 'No flow meter fitted'] : flow >= 1.4 ? ['emerald', 'Normal flow'] : flow >= 1.0 ? ['amber', 'Reduced flow'] : ['rose', 'Flow is choking'];
  const card = (icon, iconC, title, value, unit, label, labelC, foot) => `
    <div class="card p-5 flex flex-col gap-3">
      <div class="flex items-center justify-between text-xs font-semibold text-slate-500"><span>${title}</span><i class="fa-solid ${icon} text-${iconC}-500"></i></div>
      <div><span class="text-3xl font-extrabold font-mono text-slate-900">${value}</span> <span class="text-xs text-slate-400 font-semibold">${unit}</span></div>
      <div class="text-xs font-semibold text-${labelC}-600">${label}</div>
      <div class="text-[11px] text-slate-500 pt-2 border-t border-slate-100">${foot}</div>
    </div>`;
  $('kpis').innerHTML =
    card('fa-filter', 'sky', 'Salt rejection', fmt(rej, 1), '%', rejL[1], rejL[0], `Share of dissolved solids removed. Temp-corrected: ${fmt(p.temp_compensated_rejection_pct, 1)}%`)
    + card('fa-glass-water', 'teal', 'Purified water TDS', fmt(r.tdsOut, 1), 'ppm', q.text, q.c, `Feed: ${fmt(r.tdsIn, 1)} ppm · removed ${fmt(p.delta_tds_ppm, 1)} ppm`)
    + card('fa-gauge-high', 'amber', 'Flow rate', flow === null ? 'N/A' : fmt(flow, 2), flow === null ? '' : 'L/min', flowL[1], flowL[0], flow === null ? 'Add a YF-S201 to detect clogged pre-filters' : `Flow × rejection index: ${fmt(p.flow_rejection_index, 2)}`)
    + card('fa-temperature-half', 'rose', 'Water temperature', fmt(r.temp, 1), '°C', 'Used for correction', 'slate', 'TDS readings are corrected to 25 °C');
}

// ---------------------------------------------------------------- history charts
const MAX_HISTORY = 30;
let tdsChart = null;
let rejChart = null;

function initCharts() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.font.family = '"Plus Jakarta Sans", sans-serif';
  Chart.defaults.color = '#64748b';
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { boxWidth: 10, usePointStyle: true, font: { weight: '600', size: 11 } } } },
    scales: { x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } } } },
  };
  tdsChart = new Chart($('tdsChart'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Feed TDS (ppm)', data: [], borderColor: '#0284c7', backgroundColor: 'rgba(2,132,199,.08)', fill: true, tension: .3, borderWidth: 2, pointRadius: 3 },
      { label: 'Purified TDS (ppm)', data: [], borderColor: '#0d9488', backgroundColor: 'rgba(13,148,136,.08)', fill: true, tension: .3, borderWidth: 2, pointRadius: 4, pointBackgroundColor: [] },
    ] },
    options: { ...common, scales: { ...common.scales, y: { beginAtZero: true, grid: { color: '#f1f5f9' } } } },
  });
  rejChart = new Chart($('rejChart'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Salt rejection (%)', data: [], borderColor: '#64748b', tension: .3, borderWidth: 2, pointRadius: 5, pointBackgroundColor: [], pointBorderColor: '#fff', yAxisID: 'y' },
      { label: 'Flow (L/min)', data: [], borderColor: '#d97706', borderDash: [5, 4], tension: .3, borderWidth: 2, pointRadius: 2, spanGaps: false, yAxisID: 'y1' },
    ] },
    options: { ...common, scales: { ...common.scales,
      y: { min: 40, max: 100, grid: { color: '#f1f5f9' }, title: { display: true, text: '%', font: { size: 10 } } },
      y1: { position: 'right', min: 0, max: 3, grid: { drawOnChartArea: false }, title: { display: true, text: 'L/min', font: { size: 10 } } },
    } },
  });
}

function addHistory(res, r, source) {
  if (!tdsChart) return;
  const label = source.kind === 'simulation' && source.step ? source.step : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const color = STATUS_HEX[res.predicted_status];
  const push = (chart, values, colorIdx) => {
    chart.data.labels.push(label);
    values.forEach((v, i) => chart.data.datasets[i].data.push(v));
    chart.data.datasets[colorIdx].pointBackgroundColor.push(color);
    if (chart.data.labels.length > MAX_HISTORY) {
      chart.data.labels.shift();
      chart.data.datasets.forEach((d) => { d.data.shift(); if (Array.isArray(d.pointBackgroundColor)) d.pointBackgroundColor.shift(); });
    }
    chart.update();
  };
  push(tdsChart, [r.tdsIn, r.tdsOut], 1);
  push(rejChart, [res.physics_indicators.salt_rejection_pct, r.flow], 0);
}

function clearHistory() {
  [tdsChart, rejChart].forEach((c) => {
    if (!c) return;
    c.data.labels = [];
    c.data.datasets.forEach((d) => { d.data = []; if (Array.isArray(d.pointBackgroundColor)) d.pointBackgroundColor = []; });
    c.update();
  });
}

// ---------------------------------------------------------------- bench readings & examples
function fillVoltage(vIn, vOut, temp, flow = null) {
  setInputMode('voltage');
  $('vIn').value = vIn;
  $('vOut').value = vOut;
  $('vTemp').value = temp;
  setFlow(flow);
}
function fillPpm(tdsIn, tdsOut, temp, flow = null) {
  $('pTdsIn').value = tdsIn; $('rTdsIn').value = tdsIn;
  $('pTdsOut').value = tdsOut; $('rTdsOut').value = tdsOut;
  $('pTemp').value = temp; $('rTemp').value = temp;
  setFlow(flow);
}
function setFlow(flow) {
  const on = flow !== null && flow !== undefined;
  $('flowOn').checked = on;
  $('flowInputs').classList.toggle('hidden', !on);
  $('flowHelp').textContent = on
    ? 'Using the flow value below: the full model (TDS + flow) will be used.'
    : 'No flow meter fitted: the TDS-only model will be used.';
  if (on) { $('pFlow').value = flow; $('rFlow').value = flow; }
}

function onRealPick() {
  const v = $('realPick').value;
  if (!v) return;
  const [kind, idx] = v.split(':');
  if (sim.running) pauseSimulation('Simulation paused.');
  if (kind === 'avg') {
    const a = state.realAverages[+idx];
    fillVoltage(a.tds_in_voltage, a.tds_out_voltage, a.temperature);
    state.source = { kind: 'sensor', label: `${a.sample_type} · average` };
  } else {
    const row = state.realRows[+idx];
    fillVoltage(row.tds_in_voltage, row.tds_out_voltage, row.temperature);
    state.source = { kind: 'sensor', label: `${row.sample_type} #${row.reading_no}` };
  }
  runCheck();
}

async function loadPresets() {
  let scenarios = [];
  let rows = [];
  try { scenarios = await api('/api/scenarios'); } catch (e) { toast(e.message); }
  try { rows = await api('/api/real-readings'); } catch (_) { /* optional */ }

  state.realAverages = scenarios.filter((s) => s.source === 'real').map((s) => ({ ...s, sample_type: s.name.replace(' (real bench data)', '') }));
  state.examples = scenarios.filter((s) => s.source !== 'real');
  state.realRows = rows;

  const sel = $('realPick');
  let html = '<option value="">— choose a reading —</option>';
  if (state.realAverages.length) {
    html += '<optgroup label="Averages">' + state.realAverages.map((a, i) => `<option value="avg:${i}">${esc(a.sample_type)} · average</option>`).join('') + '</optgroup>';
  }
  const groups = {};
  rows.forEach((r, i) => { (groups[r.sample_type] = groups[r.sample_type] || []).push([r, i]); });
  Object.entries(groups).forEach(([name, list]) => {
    html += `<optgroup label="${esc(name)}">` + list.map(([r, i]) => `<option value="row:${i}">#${esc(r.reading_no)} · ${fmt(r.tds_in_voltage, 3)} V / ${fmt(r.tds_out_voltage, 3)} V · ${fmt(r.temperature, 2)} °C</option>`).join('') + '</optgroup>';
  });
  sel.innerHTML = html;

  $('exampleList').innerHTML = state.examples.map((s, i) => {
    const c = (STATUS[s.category] || STATUS['Monitor']).c;
    return `<button onclick="applyExample(${i})" class="w-full text-left p-3.5 rounded-xl border border-slate-200 hover:border-${c}-300 hover:bg-${c}-50/40 transition flex items-start justify-between gap-3">
      <div><div class="text-sm font-bold text-slate-900">${esc(s.name)}</div><div class="text-[11px] text-slate-500 mt-0.5">${esc(s.description)}</div></div>
      <span class="shrink-0 text-[10px] font-bold px-2 py-0.5 rounded-full bg-${c}-100 text-${c}-800">${esc(s.category)}</span>
    </button>`;
  }).join('');
}

function applyExample(i) {
  const s = state.examples[i];
  if (sim.running) pauseSimulation('Simulation paused.');
  fillPpm(s.tds_in, s.tds_out, s.temperature, s.flow_rate);
  diagnose('/api/predict', { tds_in: s.tds_in, tds_out: s.tds_out, temperature: s.temperature, flow_rate: s.flow_rate },
    { kind: 'example', label: `Example: ${s.name}` });
}

// ---------------------------------------------------------------- simulation
function rng(seed) { // deterministic mulberry32, so each story is repeatable
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const noise = (r, amp) => (r() * 2 - 1) * amp;
const r2 = (v, d = 1) => Math.round(v * 10 ** d) / 10 ** d;

function ppmStep(label, feed, rej, temp, flow) {
  return { kind: 'ppm', label, payload: { tds_in: r2(feed), tds_out: r2(feed * (1 - rej)), temperature: r2(temp), flow_rate: flow === null ? null : r2(Math.max(flow, 0), 2) } };
}

const SIMS = [
  {
    id: 'real', title: 'Replay my bench readings', icon: 'fa-microscope',
    desc: 'Your 18 real Serial Monitor readings, one at a time (no flow).',
    build: () => state.realRows.map((r) => ({
      kind: 'voltage', label: `${r.sample_type} #${r.reading_no}`,
      payload: { tds_in_voltage: r.tds_in_voltage, tds_out_voltage: r.tds_out_voltage, temperature: r.temperature, flow_rate: null, k_in: 1, k_out: 1 },
    })),
  },
  {
    id: 'healthy', title: 'A healthy purifier', icon: 'fa-sun',
    desc: '24 hours of normal use. Small natural changes; it should stay Normal.',
    build: () => {
      const r = rng(7);
      return Array.from({ length: 24 }, (_, h) => ppmStep(`Hour ${h + 1}`,
        420 + 18 * Math.sin(h / 4) + noise(r, 6), 0.94 + noise(r, 0.008), 25 + 2.5 * Math.sin((h - 6) / 3.8) + noise(r, 0.3), 1.9 + noise(r, 0.06)));
    },
  },
  {
    id: 'ageing', title: 'Membrane wearing out', icon: 'fa-hourglass-half',
    desc: '26 weeks. Salt rejection slowly falls as the RO membrane ages.',
    build: () => {
      const r = rng(11);
      return Array.from({ length: 26 }, (_, w) => {
        const t = w / 25;
        return ppmStep(`Week ${w + 1}`, 450 + noise(r, 20), 0.955 - 0.34 * t ** 1.6 + noise(r, 0.004), 26 + noise(r, 2), 1.95 - 0.7 * t ** 1.3 + noise(r, 0.04));
      });
    },
  },
  {
    id: 'clogging', title: 'Pre-filter clogging', icon: 'fa-filter-circle-xmark',
    desc: '20 days. Water stays clean but flow keeps dropping as sediment builds up.',
    build: () => {
      const r = rng(23);
      return Array.from({ length: 20 }, (_, d) => {
        const t = d / 19;
        // Clogging cuts flow sharply; rejection also slips a little (less water pushed through the membrane).
        return ppmStep(`Day ${d + 1}`, 430 + noise(r, 15), 0.94 - 0.10 * t + noise(r, 0.004), 25.5 + noise(r, 1.5), 1.95 - 1.6 * t ** 1.7 + noise(r, 0.03));
      });
    },
  },
];

const sim = { def: SIMS[0], steps: [], i: 0, running: false, runId: 0, speed: 900, lastStatus: null };

function renderSimChoices() {
  $('simChoices').innerHTML = SIMS.map((s) => {
    const active = sim.def.id === s.id;
    const disabled = s.id === 'real' && !state.realRows.length;
    return `<button ${disabled ? 'disabled' : ''} onclick="selectSim('${s.id}')" class="w-full text-left p-3 rounded-xl border-2 transition flex items-start gap-3 ${active ? 'border-violet-400 bg-violet-50/60' : 'border-slate-100 hover:border-slate-200'} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}">
      <span class="w-8 h-8 shrink-0 rounded-lg ${active ? 'bg-violet-600 text-white' : 'bg-slate-100 text-slate-500'} flex items-center justify-center text-xs"><i class="fa-solid ${s.icon}"></i></span>
      <span><span class="block text-sm font-bold text-slate-900">${s.title}</span><span class="block text-[11px] text-slate-500">${s.desc}</span></span>
    </button>`;
  }).join('');
}

function selectSim(id) {
  pauseSimulation();
  sim.def = SIMS.find((s) => s.id === id) || SIMS[0];
  resetSimulation();
  renderSimChoices();
}

function setSimSpeed(ms) {
  sim.speed = ms;
  document.querySelectorAll('#simSpeed button').forEach((b) => b.classList.toggle('active', +b.dataset.ms === ms));
}

function updateSimUI() {
  const n = sim.steps.length || (sim.def.build().length);
  $('simCount').textContent = `${sim.i} / ${n}`;
  $('simProgress').style.width = `${n ? (sim.i / n) * 100 : 0}%`;
  const btn = $('simPlayBtn');
  const done = sim.steps.length && sim.i >= sim.steps.length;
  btn.innerHTML = sim.running
    ? '<i class="fa-solid fa-pause text-[11px]"></i><span>Pause</span>'
    : `<i class="fa-solid ${done ? 'fa-rotate-right' : 'fa-play'} text-[11px]"></i><span>${done ? 'Play again' : sim.i > 0 ? 'Resume' : 'Start'}</span>`;
  $('headerSimIcon').className = `fa-solid ${sim.running ? 'fa-pause' : 'fa-play'} text-[11px]`;
  $('headerSimText').textContent = sim.running ? 'Pause' : 'Simulate';
  $('simBanner').classList.toggle('hidden', !sim.running);
  $('simBannerTitle').textContent = `Simulation running: ${sim.def.title}`;
}

function logSim(html, cls = '') {
  const log = $('simLog');
  if (log.dataset.empty !== 'no') { log.innerHTML = ''; log.dataset.empty = 'no'; }
  const li = document.createElement('li');
  li.className = `flex gap-2 items-start ${cls}`;
  li.innerHTML = html;
  log.appendChild(li);
  log.scrollTop = log.scrollHeight;
}

function resetSimulation() {
  pauseSimulation();
  sim.steps = [];
  sim.i = 0;
  sim.lastStatus = null;
  $('simLog').innerHTML = '<li class="text-slate-400">Status changes will be listed here.</li>';
  $('simLog').dataset.empty = 'yes';
  $('simStepLabel').textContent = 'Not started';
  updateSimUI();
}

function toggleSimulation() {
  if (sim.running) pauseSimulation();
  else startSimulation();
}

async function startSimulation() {
  if (!sim.steps.length) sim.steps = sim.def.build();
  if (!sim.steps.length) { toast('Nothing to simulate yet.'); return; }
  if (sim.i >= sim.steps.length) { // finished: start over
    sim.i = 0; sim.lastStatus = null;
    $('simLog').dataset.empty = 'yes';
  }
  sim.running = true;
  const myRun = ++sim.runId;
  updateSimUI();
  while (sim.running && myRun === sim.runId && sim.i < sim.steps.length) {
    const step = sim.steps[sim.i];
    if (step.kind === 'voltage') fillVoltage(step.payload.tds_in_voltage, step.payload.tds_out_voltage, step.payload.temperature);
    else { setInputMode('ppm'); fillPpm(step.payload.tds_in, step.payload.tds_out, step.payload.temperature, step.payload.flow_rate); }
    $('simStepLabel').textContent = step.label;
    const endpoint = step.kind === 'voltage' ? '/api/predict-voltage' : '/api/predict';
    const res = await diagnose(endpoint, step.payload, { kind: 'simulation', label: `Simulation · ${step.label}`, step: step.label });
    if (myRun !== sim.runId) return; // paused / reset while waiting
    if (!res) { pauseSimulation(); return; }
    if (res.predicted_status !== sim.lastStatus) {
      const c = STATUS[res.predicted_status].c;
      const flowTxt = res.physics_indicators.flow_rate_lpm === null ? '' : `, flow ${fmt(res.physics_indicators.flow_rate_lpm, 2)} L/min`;
      logSim(`<span class="mt-1 w-2 h-2 shrink-0 rounded-full bg-${c}-500"></span><span><strong>${esc(step.label)}:</strong> ${sim.lastStatus ? 'changed to' : 'starts as'} <strong class="text-${c}-700">${esc(res.predicted_status)}</strong> (rejection ${fmt(res.physics_indicators.salt_rejection_pct, 1)}%${flowTxt})${res.guardrail_triggered ? ' · safety rule' : ''}</span>`);
      sim.lastStatus = res.predicted_status;
    }
    sim.i += 1;
    updateSimUI();
    if (sim.i < sim.steps.length) await sleep(sim.speed);
  }
  if (myRun === sim.runId && sim.i >= sim.steps.length) {
    sim.running = false;
    logSim(`<i class="fa-solid fa-flag-checkered text-slate-400 mt-0.5"></i><span>Finished after ${sim.steps.length} steps. Final result: <strong>${esc(sim.lastStatus)}</strong>.</span>`, 'pt-1');
    updateSimUI();
  }
}

function pauseSimulation(message) {
  if (!sim.running) return;
  sim.running = false;
  sim.runId += 1;
  updateSimUI();
  if (message) toast(message);
}
function stopSimulation() { pauseSimulation(); }

function headerSimulate() {
  switchTab('live');
  setTimeout(() => $('simCard').scrollIntoView({ behavior: 'smooth', block: 'center' }), 150);
  toggleSimulation();
}

// ---------------------------------------------------------------- batch lab
let batchRows = [];

function batchQuery() {
  const kIn = parseFloat($('batchKIn').value) || 1;
  const kOut = parseFloat($('batchKOut').value) || 1;
  return `k_in=${kIn}&k_out=${kOut}`;
}

async function loadDataset(kind) {
  const url = kind === 'real' ? `/api/load-real-data?${batchQuery()}`
    : kind === 'simflow' ? `/api/load-real-data?${batchQuery()}&with_simulated_flow=true`
      : '/api/load-sample-csv';
  await runBatch(() => api(url));
}

async function uploadCsv(file) {
  if (!file) return;
  if (!/\.csv$/i.test(file.name)) { toast('Please choose a .csv file.'); return; }
  const fd = new FormData();
  fd.append('file', file);
  await runBatch(() => api(`/api/upload-csv?${batchQuery()}`, { method: 'POST', body: fd }));
}

async function runBatch(fetcher) {
  $('batchLoading').classList.remove('hidden');
  $('batchResults').classList.add('hidden');
  try {
    renderBatch(await fetcher());
  } catch (e) {
    toast(`Could not analyse: ${e.message}`, 6000);
  } finally {
    $('batchLoading').classList.add('hidden');
  }
}

function renderBatch(d) {
  $('batchResults').classList.remove('hidden');
  $('resTotal').textContent = d.total_records.toLocaleString();
  $('resFile').textContent = d.filename;
  [['Normal', 'Normal'], ['Monitor', 'Monitor'], ['Service', 'Service Required']].forEach(([id, key]) => {
    $(`res${id}`).textContent = `${d.percentages[key]}%`;
    $(`res${id}N`).textContent = `${d.breakdown[key]} readings`;
  });

  const info = [];
  if (d.input_mode === 'voltage') {
    const cal = d.calibration || {};
    info.push(`Input was probe voltage, converted to ppm with K1 = ${cal.k_in}, K2 = ${cal.k_out}${cal.k_in === 1 && cal.k_out === 1 ? ' (uncalibrated, so ppm values are approximate)' : ''}.`);
  }
  if (d.simulated_flow_rows > 0) info.push(`${d.simulated_flow_rows} readings use simulated flow values. They show how the model reacts to flow, not a measured diagnosis.`);
  if (!d.flow_available) info.push('No flow values, so the TDS-only model was used. Clogged pre-filters can\'t be detected.');
  else if (d.flow_rows < d.total_records) info.push(`${d.total_records - d.flow_rows} readings had no flow and used the TDS-only model.`);
  $('batchInfo').innerHTML = info.map((t) => `<div><i class="fa-solid fa-circle-info mr-2"></i>${esc(t)}</div>`).join('');
  $('batchInfo').classList.toggle('hidden', !info.length);

  const groups = d.by_sample_type || [];
  $('groupCards').innerHTML = groups.map((g) => {
    const c = STATUS[g.majority_status].c;
    const counts = Object.entries(g.status_counts).map(([k, v]) => `${v} ${k}`).join(', ');
    return `<div class="card-soft p-5 space-y-2 text-sm">
      <div class="flex items-center justify-between"><strong class="text-base text-slate-900">${esc(g.sample_type)}</strong><span class="text-[11px] font-bold px-2.5 py-1 rounded-full bg-${c}-100 text-${c}-800">${esc(g.majority_status)}</span></div>
      <div class="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs text-slate-600">
        <span>Readings</span><strong class="font-mono text-right">${g.records}</strong>
        ${g.avg_tds_in_voltage !== undefined ? `<span>Avg voltage S1 / S2</span><strong class="font-mono text-right">${fmt(g.avg_tds_in_voltage, 3)} / ${fmt(g.avg_tds_out_voltage, 3)} V</strong>` : ''}
        <span>Avg feed / output</span><strong class="font-mono text-right">${fmt(g.avg_tds_in_ppm)} / ${fmt(g.avg_tds_out_ppm)} ppm</strong>
        <span>Avg salt rejection</span><strong class="font-mono text-right">${fmt(g.avg_salt_rejection_pct, 2)}%</strong>
        <span>Temperature</span><strong class="font-mono text-right">${fmt(g.temperature_range_c[0], 1)}–${fmt(g.temperature_range_c[1], 1)} °C</strong>
        <span>Results</span><strong class="text-right">${esc(counts)}</strong>
      </div>
    </div>`;
  }).join('');
  $('groupSection').classList.toggle('hidden', !groups.length);

  batchRows = d.rows_preview;
  filterTable('all');
  $('batchResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function filterTable(f) {
  document.querySelectorAll('#tableFilter button').forEach((b) => b.classList.toggle('active', b.dataset.f === f));
  const rows = f === 'all' ? batchRows : batchRows.filter((r) => r.predicted_status === f);
  $('batchTable').innerHTML = rows.length ? rows.map((r) => {
    const c = STATUS[r.predicted_status].c;
    const v = (x) => (x !== undefined ? `<span class="block text-[10px] text-slate-400">${x} V</span>` : '');
    return `<tr class="hover:bg-sky-50/50 cursor-pointer" onclick="openBatchRow(${r.row_index})" title="Open in Live Check">
      <td class="py-2.5 px-3 text-slate-400">${r.row_index}</td>
      <td class="py-2.5 px-3 font-medium text-slate-700">${esc(r.timestamp)}</td>
      <td class="py-2.5 px-3 font-mono text-sky-700">${r.tds_in} ppm${v(r.tds_in_voltage)}</td>
      <td class="py-2.5 px-3 font-mono text-teal-700">${r.tds_out} ppm${v(r.tds_out_voltage)}</td>
      <td class="py-2.5 px-3 font-mono">${r.salt_rejection}%</td>
      <td class="py-2.5 px-3 font-mono text-amber-700">${r.flow_rate === null ? '<span class="text-slate-400">—</span>' : `${r.flow_rate} L/min`}${r.flow_simulated ? '<span class="block text-[10px] text-violet-500 font-sans font-semibold">simulated</span>' : ''}</td>
      <td class="py-2.5 px-3 font-mono">${r.temperature} °C</td>
      <td class="py-2.5 px-3"><span class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-${c}-100 text-${c}-800">${esc(r.predicted_status)}</span>${r.guardrail_triggered ? ' <i class="fa-solid fa-shield-halved text-rose-500" title="Safety rule applied"></i>' : ''}</td>
      <td class="py-2.5 px-3 font-mono">${r.guardrail_triggered ? '<span class="text-[10px] text-rose-600 font-sans font-semibold">rule</span>' : `${r.confidence}%`}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="9" class="py-6 text-center text-slate-400">No readings with this result.</td></tr>';
}

function openBatchRow(rowIndex) {
  const r = batchRows.find((x) => x.row_index === rowIndex);
  if (!r) return;
  if (sim.running) pauseSimulation();
  switchTab('live');
  if (r.tds_in_voltage !== undefined) {
    fillVoltage(r.tds_in_voltage, r.tds_out_voltage, r.temperature, r.flow_rate);
    $('kIn').value = $('batchKIn').value; $('kOut').value = $('batchKOut').value;
  } else {
    setInputMode('ppm');
    fillPpm(r.tds_in, r.tds_out, r.temperature, r.flow_rate);
  }
  state.source = { kind: 'batch', label: `Batch row · ${r.timestamp}${r.flow_simulated ? ' (simulated flow)' : ''}` };
  runCheck();
}

function initDropZone() {
  const dz = $('dropZone');
  const input = $('csvFileInput');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { uploadCsv(input.files[0]); input.value = ''; });
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('border-sky-500', 'bg-sky-50'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('border-sky-500', 'bg-sky-50'); }));
  dz.addEventListener('drop', (e) => uploadCsv(e.dataTransfer.files[0]));
}

// ---------------------------------------------------------------- hardware tab
let telemetryLoaded = false;
async function loadTelemetry() {
  if (telemetryLoaded) return;
  try {
    const t = await api('/api/system-telemetry');
    const icons = { 'Feed TDS': 'fa-faucet-drip', 'Permeate TDS': 'fa-glass-water', 'Temperature': 'fa-temperature-half', 'Permeate Flow': 'fa-gauge-high' };
    $('sensorStatus').innerHTML = t.hardware.sensors.map((s) => {
      const online = s.status === 'ONLINE';
      return `<div class="card-soft p-4 space-y-2">
        <div class="flex items-center justify-between">
          <span class="w-9 h-9 rounded-xl ${online ? 'bg-emerald-50 text-emerald-600' : 'bg-amber-50 text-amber-600'} flex items-center justify-center"><i class="fa-solid ${icons[s.name] || 'fa-microchip'}"></i></span>
          <span class="text-[10px] font-bold px-2 py-0.5 rounded-full ${online ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'}">${online ? 'Connected' : 'Not fitted'}</span>
        </div>
        <div class="font-bold text-sm text-slate-900">${esc(s.name)}</div>
        <div class="text-[11px] text-slate-500">${esc(s.sensor)}</div>
        <div class="text-[11px] font-mono text-slate-600">${esc(s.channel)} · ${esc(s.output_unit || '')}</div>
      </div>`;
    }).join('');
    telemetryLoaded = true;
  } catch (e) {
    $('sensorStatus').innerHTML = `<div class="text-sm text-rose-600">Could not load node status: ${esc(e.message)}</div>`;
  }
}

// ---------------------------------------------------------------- ML tab
let metricsLoaded = false;
async function loadMetrics() {
  if (metricsLoaded) return;
  let m;
  try { m = await api('/api/metrics'); } catch (e) { $('modelCards').innerHTML = `<div class="text-sm text-rose-600">${esc(e.message)}</div>`; return; }
  if (!m.comparison) { $('modelCards').innerHTML = '<div class="text-sm text-slate-500">Metrics not found. Run <code class="inline">python -m model.train</code>.</div>'; return; }
  const variants = [[m, 'Full model', 'TDS + temperature + flow', 'Used when a flow reading exists', 'sky']];
  if (m.no_flow_model) variants.push([m.no_flow_model, 'TDS-only model', 'TDS + temperature', 'Used now (no flow meter fitted)', 'teal']);
  $('modelCards').innerHTML = variants.map(([v, name, feats, when, c]) => modelCard(v, name, feats, when, c)).join('');
  metricsLoaded = true;
}

function modelCard(v, name, feats, when, c) {
  const pct = (x) => `${fmt(x * 100, 1)}%`;
  const imps = Object.entries(v.feature_importances).slice(0, 6);
  const maxImp = Math.max(...imps.map(([, x]) => x)) || 1;
  const cm = v.confusion_matrix;
  const cmMax = Math.max(...cm.flat()) || 1;
  const shortCls = ['Normal', 'Monitor', 'Service'];
  const rows = Object.entries(v.comparison).map(([k, s]) => `
    <tr><td class="py-2 px-3">${esc(k.replace(/ \(.*\)/, ''))}</td><td class="py-2 px-3 font-mono text-right">${fmt(s.cv_f1_mean, 3)} ± ${fmt(s.cv_f1_std, 3)}</td><td class="py-2 px-3 font-mono text-right">${pct(s.test_accuracy)}</td></tr>`).join('');
  return `<div class="card-soft p-5 space-y-5">
    <div class="flex items-start justify-between gap-3">
      <div><div class="text-base font-bold text-slate-900">${name}</div><div class="text-xs text-slate-500">${feats}</div></div>
      <span class="text-[10px] font-bold px-2.5 py-1 rounded-full bg-${c}-100 text-${c}-800 text-right">${when}</span>
    </div>
    <div class="grid grid-cols-3 gap-3 text-center">
      <div class="bg-white rounded-xl border border-slate-200 p-3"><div class="text-xl font-extrabold font-mono text-${c}-700">${pct(v.test_accuracy)}</div><div class="text-[10px] text-slate-500 font-semibold">Test accuracy</div></div>
      <div class="bg-white rounded-xl border border-slate-200 p-3"><div class="text-xl font-extrabold font-mono text-${c}-700">${fmt(v.test_macro_f1, 3)}</div><div class="text-[10px] text-slate-500 font-semibold">Macro F1</div></div>
      <div class="bg-white rounded-xl border border-slate-200 p-3"><div class="text-xl font-extrabold font-mono text-${c}-700">${v.feature_columns.length}</div><div class="text-[10px] text-slate-500 font-semibold">Inputs used</div></div>
    </div>
    <div>
      <div class="eyebrow mb-2">What the model relies on most</div>
      <div class="space-y-1.5">${imps.map(([f, x]) => `<div class="flex items-center gap-2 text-xs"><span class="w-36 shrink-0 font-mono text-slate-600 truncate">${esc(f)}</span><div class="flex-1 h-2 bg-white rounded-full border border-slate-200 overflow-hidden"><div class="h-full bg-${c}-500" style="width:${(x / maxImp) * 100}%"></div></div><span class="w-10 text-right font-mono text-slate-500">${fmt(x * 100, 0)}%</span></div>`).join('')}</div>
    </div>
    <div>
      <div class="eyebrow mb-2">Test results (rows = truth, columns = predicted)</div>
      <div class="grid grid-cols-4 gap-1 text-[11px] text-center">
        <div></div>${shortCls.map((s) => `<div class="font-semibold text-slate-500 py-1">${s}</div>`).join('')}
        ${cm.map((row, i) => `<div class="font-semibold text-slate-500 py-2 text-right pr-2">${shortCls[i]}</div>` + row.map((n, j) => `<div class="rounded-lg py-2 font-mono font-bold ${i === j ? 'text-white' : (n ? 'text-rose-700' : 'text-slate-300')}" style="background:${i === j ? `rgba(13,148,136,${0.35 + 0.65 * n / cmMax})` : (n ? '#ffe4e6' : '#fff')}">${n}</div>`).join('')).join('')}
      </div>
    </div>
    <div>
      <div class="eyebrow mb-2">Compared with simpler baselines</div>
      <table class="w-full text-xs bg-white rounded-xl border border-slate-200 overflow-hidden"><thead class="bg-slate-50 text-slate-500"><tr><th class="py-2 px-3 text-left">Model</th><th class="py-2 px-3 text-right">5-fold CV F1</th><th class="py-2 px-3 text-right">Test acc.</th></tr></thead><tbody class="divide-y divide-slate-100">${rows}</tbody></table>
    </div>
  </div>`;
}

// ---------------------------------------------------------------- boot
window.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  initDropZone();
  setSimSpeed(900);
  await loadPresets();
  renderSimChoices();
  resetSimulation();
  window.addEventListener('hashchange', () => switchTab(location.hash.slice(1)));
  if (location.hash.length > 1) switchTab(location.hash.slice(1));
  if (state.realAverages.length) {
    $('realPick').value = 'avg:0';
    onRealPick();
  } else {
    runCheck(true);
  }
});
