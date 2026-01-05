const els = {
  file: document.getElementById('file'),
  tokenStream: document.getElementById('tokenStream'),
  promptText: document.getElementById('promptText'),
  meta: document.getElementById('meta'),
  unit: document.getElementById('unit'),
  deadzone: document.getElementById('deadzone'),
  deadzoneVal: document.getElementById('deadzoneVal'),
  gamma: document.getElementById('gamma'),
  gammaVal: document.getElementById('gammaVal'),
  minCount: document.getElementById('minCount'),
  topN: document.getElementById('topN'),
  topTokensFull: document.getElementById('topTokensFull'),
  topTokensSelection: document.getElementById('topTokensSelection'),
  selectionStats: document.getElementById('selectionStats'),
  clearSelection: document.getElementById('clearSelection'),
  hoverShowTokenId: document.getElementById('hoverShowTokenId'),
  hoverShowEntropy: document.getElementById('hoverShowEntropy'),
  hoverShowMass: document.getElementById('hoverShowMass'),
  hoverShowAlternatives: document.getElementById('hoverShowAlternatives'),
};

let logData = null;
let steps = [];
let selStart = null;
let selEnd = null;
let hoverTooltip = null;

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function escapeHtml(s) {
  const text = String(s ?? '');
  return text.replaceAll('&', '&amp;')
             .replaceAll('<', '&lt;')
             .replaceAll('>', '&gt;')
             .replaceAll('"', '&quot;')
             .replaceAll("'", '&#039;');
}

function showTokenForTable(tok) {
  // Keep it one-line for the table.
  let s = String(tok ?? '');
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
  if (f === c) return sorted[f];
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
  const nats = Number(step.entropy_nats);
  const bits = Number(step.entropy_bits);
  if (els.unit.value === 'nats') {
    if (Number.isFinite(nats)) return nats;
    if (Number.isFinite(bits)) return bits * Math.log(2);
    return NaN;
  }
  // bits
  if (Number.isFinite(bits)) return bits;
  if (Number.isFinite(nats)) return nats / Math.log(2);
  return NaN;
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
  if (els.promptText) els.promptText.textContent = '';
  els.selectionStats.textContent = 'Click two tokens to select a range.';
  selStart = null;
  selEnd = null;
  els.clearSelection.disabled = true;
}

function ensureHoverTooltip() {
  if (hoverTooltip) return;
  hoverTooltip = document.createElement('div');
  hoverTooltip.className = 'hover-tooltip';
  hoverTooltip.style.display = 'none';
  document.body.appendChild(hoverTooltip);
}

function positionTooltip(x, y) {
  if (!hoverTooltip) return;
  const pad = 12;
  const rect = hoverTooltip.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + rect.width > window.innerWidth - pad) {
    left = x - rect.width - 14;
  }
  if (top + rect.height > window.innerHeight - pad) {
    top = y - rect.height - 14;
  }
  hoverTooltip.style.left = `${Math.max(pad, left)}px`;
  hoverTooltip.style.top = `${Math.max(pad, top)}px`;
}

function showTooltip(text, x, y) {
  if (!text) return;
  ensureHoverTooltip();
  hoverTooltip.textContent = text;
  hoverTooltip.style.display = 'block';
  positionTooltip(x, y);
}

function hideTooltip() {
  if (!hoverTooltip) return;
  hoverTooltip.style.display = 'none';
}

window.addEventListener('blur', hideTooltip);
window.addEventListener('scroll', hideTooltip, true);

function getTopN() {
  const v = parseInt((els.topN && els.topN.value) || '100', 10);
  if (!Number.isFinite(v) || v <= 0) return 100;
  return v;
}

function formatProb(p) {
  if (!isFiniteNumber(p)) return 'NA';
  return (p * 100).toFixed(2) + '%';
}

function safeExp(logp) {
  const lp = Number(logp);
  if (!Number.isFinite(lp)) return NaN;
  // exp(-inf) is 0; exp(1000) would overflow, but logprobs should be <= 0.
  if (lp < -1000) return 0;
  return Math.exp(lp);
}

function topAlternativesText(step, maxAlts = 5) {
  if (!step || !Array.isArray(step.top_logprobs) || step.top_logprobs.length < 2) return '';
  const alts = step.top_logprobs.slice(1, 1 + maxAlts);
  if (!alts.length) return '';
  const lines = [];
  lines.push('top alternatives:');
  for (const c of alts) {
    const tok = (c && typeof c.token === 'string') ? c.token : '';
    const p = safeExp(c && c.logprob);
    lines.push(`  ${showTokenForTable(tok)}  (${formatProb(p)})`);
  }
  return lines.join('\n');
}

function buildTokenHoverTitle(step, pos, highlighted) {
  const t = (step.index != null) ? step.index : pos;
  const parts = [];
  parts.push(`t=${t}`);

  // Always include a stable representation for whitespace tokens.
  parts.push(`token=${showTokenForTable(step.token || '')}`);

  if (highlighted) {
    if (els.hoverShowTokenId && els.hoverShowTokenId.checked) {
      parts.push(`token_id=${step.token_id}`);
    }
    if (els.hoverShowEntropy && els.hoverShowEntropy.checked) {
      const v = entropyValue(step);
      const h = isFiniteNumber(v) ? v.toFixed(3) : 'NA';
      parts.push(`H=${h} ${els.unit.value}`);
    }
    if (els.hoverShowMass && els.hoverShowMass.checked && step.mass_observed != null) {
      parts.push(`mass=${Number(step.mass_observed).toFixed(3)}`);
    }
    if (els.hoverShowAlternatives && els.hoverShowAlternatives.checked) {
      const altText = topAlternativesText(step, 5);
      if (altText) parts.push(altText);
    }
  }

  if (step.note) parts.push(String(step.note));
  return parts.join('\n');
}

function renderMeta() {
  const m = [];
  m.push(`model: ${logData.model_id || 'unknown'}`);
  if (logData.backend) m.push(`backend: ${logData.backend}`);
  if (logData.kmax != null) m.push(`kmax: ${logData.kmax}`);
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
    const title = r.hover || r.token;
    html += `<div class="row">
      <div class="cell-token" title="${escapeHtml(title)}">${escapeHtml(showTokenForTable(r.token))}</div>
      <div>${r.count}</div>
      <div>${r.mean.toFixed(3)}</div>
    </div>`;
  }
  container.innerHTML = html;
}

function topTokens(stepsSlice, minCount, topN=30) {
  const map = new Map();
  for (const s of stepsSlice) {
    if (s.is_prompt) continue;
    const tok = s.token;
    const v = entropyValue(s);
    if (!isFiniteNumber(v)) continue;
    let agg = map.get(tok);
    if (!agg) {
      agg = { token: tok, count: 0, sum: 0, max: -Infinity, example: s };
      map.set(tok, agg);
    }
    agg.count += 1;
    agg.sum += v;
    if (v > agg.max) {
      agg.max = v;
      agg.example = s;
    }
  }
  const arr = [];
  for (const agg of map.values()) {
    if (agg.count < minCount) continue;
    const mean = agg.sum / agg.count;
    const hover = (() => {
      const ex = agg.example;
      if (!ex || !Array.isArray(ex.top_logprobs) || ex.top_logprobs.length < 2) return agg.token;
      const t = (ex.index != null) ? ex.index : '?';
      const lines = [];
      lines.push(`example t=${t}`);
      const altText = topAlternativesText(ex, 5);
      if (altText) lines.push(altText);
      return lines.join('\n');
    })();
    arr.push({ token: agg.token, count: agg.count, mean, hover });
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
  const top = topTokens(slice, minCount, getTopN());
  renderTable(els.topTokensSelection, top);
}

function updateSelectionClasses() {
  const spans = els.tokenStream.querySelectorAll('.tok');
  spans.forEach(sp => {
    const i = parseInt(sp.dataset.pos, 10);
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
  ensureHoverTooltip();

  const vals = steps.map(entropyValue).filter(isFiniteNumber);
  const params = computeColorMapParams(vals);

  const frag = document.createDocumentFragment();
  for (let pos = 0; pos < steps.length; pos++) {
    const s = steps[pos];
    const span = document.createElement('span');
    span.className = 'tok';
    span.dataset.pos = String(pos);
    span.textContent = s.token;
    const v = entropyValue(s);
    const bg = colorFor(v, params);
    span.style.background = bg;
    const highlighted = bg !== 'transparent';
    const hoverText = buildTokenHoverTitle(s, pos, highlighted);
    span.title = hoverText;

    span.addEventListener('click', () => {
      const idx = parseInt(span.dataset.pos, 10);
      if (selStart == null || (selStart != null && selEnd != null)) {
        selStart = idx;
        selEnd = null;
      } else {
        selEnd = idx;
      }
      updateSelectionClasses();
      renderSelectionStats();
    });

    span.addEventListener('pointerenter', (e) => {
      showTooltip(hoverText, e.clientX, e.clientY);
    });
    span.addEventListener('pointermove', (e) => {
      positionTooltip(e.clientX, e.clientY);
    });
    span.addEventListener('pointerleave', () => {
      hideTooltip();
    });

    frag.appendChild(span);
  }
  els.tokenStream.appendChild(frag);
  updateSelectionClasses();
}

function renderFullTopTokens() {
  if (!steps.length) return;
  const minCount = parseInt(els.minCount.value || '5', 10);
  const top = topTokens(steps, minCount, getTopN());
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
  if (els.promptText) {
    els.promptText.textContent = (logData && logData.prompt) ? String(logData.prompt) : '';
  }
  steps = (logData.steps || []).map(s => ({
    index: s.index,
    token_id: s.token_id,
    token: s.token,
    entropy_bits: s.entropy_bits,
    entropy_nats: s.entropy_nats,
    mass_observed: s.mass_observed,
    note: s.note,
    is_prompt: Boolean(s.is_prompt),
    top_logprobs: s.top_logprobs,
  }));
  // Ensure order is stable even if a log writes steps out of order.
  steps.sort((a, b) => {
    const ai = Number(a.index);
    const bi = Number(b.index);
    const aKey = Number.isFinite(ai) ? ai : 0;
    const bKey = Number.isFinite(bi) ? bi : 0;
    return aKey - bKey;
  });

  rerenderAll();
});

els.unit.addEventListener('change', rerenderAll);
els.deadzone.addEventListener('input', () => { els.deadzoneVal.textContent = parseFloat(els.deadzone.value).toFixed(2); rerenderAll(); });
els.gamma.addEventListener('input', () => { els.gammaVal.textContent = parseFloat(els.gamma.value).toFixed(2); rerenderAll(); });
els.minCount.addEventListener('input', () => { renderFullTopTokens(); renderSelectionStats(); });
if (els.topN) {
  els.topN.addEventListener('input', () => { renderFullTopTokens(); renderSelectionStats(); });
}

// Hover tooltip options only affect the token stream.
for (const el of [els.hoverShowTokenId, els.hoverShowEntropy, els.hoverShowMass, els.hoverShowAlternatives]) {
  if (!el) continue;
  el.addEventListener('change', () => { renderStream(); });
}

els.clearSelection.addEventListener('click', () => {
  selStart = null;
  selEnd = null;
  updateSelectionClasses();
  renderSelectionStats();
});

// Defaults
els.deadzoneVal.textContent = parseFloat(els.deadzone.value).toFixed(2);
els.gammaVal.textContent = parseFloat(els.gamma.value).toFixed(2);
