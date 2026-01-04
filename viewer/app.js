const els = {
  file: document.getElementById('file'),
  tokenStream: document.getElementById('tokenStream'),
  meta: document.getElementById('meta'),
  unit: document.getElementById('unit'),
  deadzone: document.getElementById('deadzone'),
  deadzoneVal: document.getElementById('deadzoneVal'),
  gamma: document.getElementById('gamma'),
  gammaVal: document.getElementById('gammaVal'),
  minCount: document.getElementById('minCount'),
  topTokensFull: document.getElementById('topTokensFull'),
  topTokensSelection: document.getElementById('topTokensSelection'),
  selectionStats: document.getElementById('selectionStats'),
  clearSelection: document.getElementById('clearSelection'),
};

let logData = null;
let steps = [];
let selStart = null;
let selEnd = null;

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function escapeHtml(s) {
  return s.replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#039;');
}

function showTokenForTable(tok) {
  // Keep it one-line for the table.
  let s = tok;
  s = s.replaceAll('\n', '\\n');
  s = s.replaceAll('\t', '\\t');
  s = s.replaceAll('\r', '\\r');
  // Collapse long whitespace runs
  if (s.trim() === '') {
    s = '[whitespace] ' + JSON.stringify(tok);
  }
  return s;
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const k = (sorted.length - 1) * (p / 100.0);
  const f = Math.floor(k);
  const c = Math.ceil(k);
  if (f === c) return sorted[k];
  return sorted[f] * (c - k) + sorted[c] * (k - f);
}

function computeColorMapParams(values) {
  const v = [...values].sort((a,b)=>a-b);
  if (!v.length) return { mid: 0, halfRange: 1 };
  const mid = percentile(v, 50);
  const p10 = percentile(v, 10);
  const p90 = percentile(v, 90);
  const halfRange = Math.max(1e-9, (p90 - p10) / 2);
  return { mid, halfRange };
}

function entropyValue(step) {
  return els.unit.value === 'nats' ? step.entropy_nats : step.entropy_bits;
}

function colorFor(value, params) {
  if (!isFiniteNumber(value)) return 'transparent';
  const deadzone = parseFloat(els.deadzone.value);
  const gamma = parseFloat(els.gamma.value);

  let z = (value - params.mid) / params.halfRange; // roughly -1..1 (p10..p90)
  z = Math.max(-1, Math.min(1, z));

  const absz = Math.abs(z);
  if (absz <= deadzone) return 'transparent';

  const t = (absz - deadzone) / (1 - deadzone);
  const a = Math.pow(t, gamma);

  if (z > 0) {
    // red
    return `rgba(220, 50, 47, ${0.55 * a})`;
  } else {
    // green
    return `rgba(133, 153, 0, ${0.45 * a})`;
  }
}

function clearStream() {
  els.tokenStream.innerHTML = '';
  els.topTokensFull.innerHTML = '';
  els.topTokensSelection.innerHTML = '';
  els.meta.textContent = '';
  els.selectionStats.textContent = 'Click two tokens to select a range.';
  selStart = null;
  selEnd = null;
  els.clearSelection.disabled = true;
}

function renderMeta() {
  const m = [];
  m.push(`model: ${logData.model_id || 'unknown'}`);
  if (logData.backend) m.push(`backend: ${logData.backend}`);
  if (logData.created_at) m.push(`created_at: ${logData.created_at}`);
  if (logData.decode) {
    const d = logData.decode;
    m.push(`decode: T=${d.temperature}, top_k=${d.top_k}, top_p=${d.top_p}, min_p=${d.min_p}, rep_pen=${d.repetition_penalty}`);
    m.push(`max_new_tokens: ${d.max_new_tokens}, seed: ${d.seed}`);
  }
  if (logData.timing && logData.timing.tokens_per_second) {
    m.push(`tokens/sec: ${logData.timing.tokens_per_second.toFixed(2)}`);
  }
  els.meta.innerHTML = '<div class="pill">' + escapeHtml(m.join(' • ')) + '</div>';
}

function renderTable(container, rows) {
  let html = '';
  html += `<div class="row hdr"><div>token</div><div>count</div><div>mean</div></div>`;
  for (const r of rows) {
    html += `<div class="row">
      <div class="cell-token" title="${escapeHtml(r.token)}">${escapeHtml(showTokenForTable(r.token))}</div>
      <div>${r.count}</div>
      <div>${r.mean.toFixed(3)}</div>
    </div>`;
  }
  container.innerHTML = html;
}

function topTokens(stepsSlice, minCount, topN=30) {
  const map = new Map();
  for (const s of stepsSlice) {
    const tok = s.token;
    const v = entropyValue(s);
    if (!isFiniteNumber(v)) continue;
    let agg = map.get(tok);
    if (!agg) {
      agg = { token: tok, count: 0, sum: 0 };
      map.set(tok, agg);
    }
    agg.count += 1;
    agg.sum += v;
  }
  const arr = [];
  for (const agg of map.values()) {
    if (agg.count < minCount) continue;
    arr.push({ token: agg.token, count: agg.count, mean: agg.sum / agg.count });
  }
  arr.sort((a,b)=>b.mean-a.mean);
  return arr.slice(0, topN);
}

function selectionSlice() {
  if (selStart == null || selEnd == null) return null;
  const a = Math.min(selStart, selEnd);
  const b = Math.max(selStart, selEnd);
  return steps.slice(a, b + 1);
}

function renderSelectionStats() {
  const slice = selectionSlice();
  if (!slice) {
    els.selectionStats.textContent = 'Click two tokens to select a range.';
    els.topTokensSelection.innerHTML = '';
    return;
  }
  const vals = slice.map(entropyValue).filter(isFiniteNumber).sort((a,b)=>a-b);
  const unit = els.unit.value;
  if (!vals.length) {
    els.selectionStats.textContent =
      `range: [${Math.min(selStart, selEnd)}..${Math.max(selStart, selEnd)}] • tokens: ${slice.length} • entropy: unavailable`;
    els.topTokensSelection.innerHTML = '';
    return;
  }
  const mean = vals.reduce((x,y)=>x+y,0) / vals.length;
  const p50 = percentile(vals, 50);
  const p90 = percentile(vals, 90);
  els.selectionStats.textContent =
    `range: [${Math.min(selStart, selEnd)}..${Math.max(selStart, selEnd)}] • tokens: ${slice.length} • entropy samples: ${vals.length} • mean: ${mean.toFixed(3)} ${unit} • p50: ${p50.toFixed(3)} • p90: ${p90.toFixed(3)}`;

  const minCount = parseInt(els.minCount.value || '5', 10);
  const top = topTokens(slice, minCount, 30);
  renderTable(els.topTokensSelection, top);
}

function updateSelectionClasses() {
  const spans = els.tokenStream.querySelectorAll('.tok');
  spans.forEach(sp => {
    const i = parseInt(sp.dataset.idx, 10);
    sp.classList.remove('selected', 'in-range');
    if (selStart == null) return;
    if (selEnd == null) {
      if (i === selStart) sp.classList.add('selected');
      return;
    }
    const a = Math.min(selStart, selEnd);
    const b = Math.max(selStart, selEnd);
    if (i === selStart || i === selEnd) sp.classList.add('selected');
    if (i >= a && i <= b) sp.classList.add('in-range');
  });
  els.clearSelection.disabled = !(selStart != null);
}

function renderStream() {
  els.tokenStream.innerHTML = '';
  if (!steps.length) return;

  const vals = steps.map(entropyValue).filter(isFiniteNumber);
  const params = computeColorMapParams(vals);

  const frag = document.createDocumentFragment();
  for (const s of steps) {
    const span = document.createElement('span');
    span.className = 'tok';
    span.dataset.idx = String(s.index);
    span.textContent = s.token;
    const v = entropyValue(s);
    span.style.background = colorFor(v, params);
    const h = isFiniteNumber(v) ? v.toFixed(3) : 'NA';
    span.title = `t=${s.index} • id=${s.token_id} • H=${h} ${els.unit.value}`;

    span.addEventListener('click', () => {
      const idx = parseInt(span.dataset.idx, 10);
      if (selStart == null || (selStart != null && selEnd != null)) {
        selStart = idx;
        selEnd = null;
      } else {
        selEnd = idx;
      }
      updateSelectionClasses();
      renderSelectionStats();
    });

    frag.appendChild(span);
  }
  els.tokenStream.appendChild(frag);
  updateSelectionClasses();
}

function renderFullTopTokens() {
  if (!steps.length) return;
  const minCount = parseInt(els.minCount.value || '5', 10);
  const top = topTokens(steps, minCount, 40);
  renderTable(els.topTokensFull, top);
}

function rerenderAll() {
  if (!logData) return;
  renderMeta();
  renderStream();
  renderFullTopTokens();
  renderSelectionStats();
}

els.file.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  clearStream();

  const text = await file.text();
  logData = JSON.parse(text);
  steps = (logData.steps || []).map(s => ({
    index: s.index,
    token_id: s.token_id,
    token: s.token,
    entropy_bits: s.entropy_bits,
    entropy_nats: s.entropy_nats,
  }));

  rerenderAll();
});

els.unit.addEventListener('change', rerenderAll);
els.deadzone.addEventListener('input', () => { els.deadzoneVal.textContent = parseFloat(els.deadzone.value).toFixed(2); rerenderAll(); });
els.gamma.addEventListener('input', () => { els.gammaVal.textContent = parseFloat(els.gamma.value).toFixed(2); rerenderAll(); });
els.minCount.addEventListener('input', () => { renderFullTopTokens(); renderSelectionStats(); });

els.clearSelection.addEventListener('click', () => {
  selStart = null;
  selEnd = null;
  updateSelectionClasses();
  renderSelectionStats();
});

// Defaults
els.deadzoneVal.textContent = parseFloat(els.deadzone.value).toFixed(2);
els.gammaVal.textContent = parseFloat(els.gamma.value).toFixed(2);
