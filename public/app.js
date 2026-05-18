'use strict';

// ── Theme ──────────────────────────────────────────────────────────────────
(function () {
  const saved = localStorage.getItem('ca-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = saved ? saved === 'dark' : prefersDark;
  if (dark) document.documentElement.setAttribute('data-theme', 'dark');

  function syncIcons(d) {
    document.querySelectorAll('.theme-toggle').forEach(btn => {
      btn.textContent = d ? '☀' : '☽';
      btn.title = d ? 'Switch to light mode' : 'Switch to dark mode';
    });
  }
  syncIcons(dark);

  document.addEventListener('click', e => {
    if (!e.target.classList.contains('theme-toggle')) return;
    const nowDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (nowDark) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', 'dark');
    localStorage.setItem('ca-theme', nowDark ? 'light' : 'dark');
    syncIcons(!nowDark);
  });
})();

// ── State ──────────────────────────────────────────────────────────────────
let _mode = 'demo';
let _runId = null;
let _repoUrl = null;
let _sse = null;
let _strictness = { demo: 'balanced', byok: 'balanced' };
const _gaps = {};
let _runStartTs = null;
let _runTimer = null;
let _lastReport = null;

// ── Helpers ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const GH_RE = /^https?:\/\/github\.com\/[^/\s]+\/[^/\s]+/;
function escHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function repoName(url) {
  return url.replace(/^https?:\/\/github\.com\//, '').replace(/\/$/, '');
}
function pluralize(n, w) { return `${n} ${w}${n === 1 ? '' : 's'}`; }

// ── View switching ─────────────────────────────────────────────────────────
function showView(name) {
  ['landing', 'progress', 'results'].forEach(v => {
    const el = $(`view-${v}`);
    if (el) el.style.display = v === name ? '' : 'none';
  });
  document.querySelectorAll('.ca-nav-link').forEach(a => {
    a.classList.toggle('active', a.dataset.view === name);
  });
  if (name === 'landing') window.scrollTo(0, 0);
}

// ── Mode tabs + strictness pills ───────────────────────────────────────────
const MODE_DESC = {
  demo: 'Server-provided keys · max 3 gaps · public repos only.',
  byok: 'Your LLM + E2B keys · up to 15 gaps · any Python repo.',
};
const STRICTNESS_HINT = {
  strict:   'Commit only with branch proof. Stricter draft checks. 3 retries per gap.',
  balanced: 'Commit only with branch proof. Default checks. 3 retries per gap.',
  loose:    'Commit only with branch proof. Easier draft checks. 1 retry per gap.',
};

function syncStrictnessHints() {
  ['demo', 'byok'].forEach(target => {
    const hint = document.getElementById(`${target}-strictness-hint`);
    if (hint) hint.textContent = STRICTNESS_HINT[_strictness[target]];
  });
}

document.addEventListener('DOMContentLoaded', syncStrictnessHints);

document.querySelectorAll('.mode-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    _mode = btn.dataset.mode;
    document.querySelectorAll('.mode-tab').forEach(b => {
      const active = b === btn;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    $('mode-desc').textContent = MODE_DESC[_mode];
    $('demo-fields').style.display = _mode === 'demo' ? '' : 'none';
    $('byok-fields').style.display = _mode === 'byok' ? '' : 'none';
    validate();
  });
});

document.querySelectorAll('.strictness-row').forEach(row => {
  const target = row.dataset.target;
  row.querySelectorAll('.strictness-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const value = pill.dataset.strictness;
      _strictness[target] = value;
      row.querySelectorAll('.strictness-pill').forEach(p => {
        p.classList.toggle('active', p === pill);
      });
      const hint = $(`${target}-strictness-hint`);
      if (hint) hint.textContent = STRICTNESS_HINT[value];
    });
  });
});

// ── Key / model coupling ───────────────────────────────────────────────────
const KEY_PREFIXES = [
  { prefix: 'gsk_',     provider: 'groq',      label: 'Groq' },
  { prefix: 'sk-ant-',  provider: 'anthropic', label: 'Anthropic' },
  { prefix: 'sk-proj-', provider: 'openai',    label: 'OpenAI' },
  { prefix: 'sk-',      provider: 'openai',    label: 'OpenAI' },
  { prefix: 'AIza',     provider: 'gemini',    label: 'Gemini' },
  { prefix: 'csk-',     provider: 'cerebras',  label: 'Cerebras' },
];

function detectProvider(key) {
  const k = (key || '').trim();
  if (!k) return null;
  for (const entry of KEY_PREFIXES) {
    if (k.startsWith(entry.prefix)) return entry;
  }
  return { provider: 'unknown', label: 'Unknown' };
}

function modelProvider(modelStr) {
  const m = (modelStr || '').trim().toLowerCase();
  return m.includes('/') ? m.split('/')[0] : 'unknown';
}

function applyProviderFilter() {
  const detected = detectProvider($('llm-key').value);
  const badge = $('provider-badge');
  const hint = $('llm-key-hint');
  const select = $('model-select');

  if (!detected) {
    badge.style.display = 'none';
    hint.textContent = 'Pasting a key auto-selects the matching model below.';
    Array.from(select.options).forEach(o => { o.hidden = false; o.disabled = false; });
    return;
  }

  if (detected.provider === 'unknown') {
    badge.style.display = 'inline-flex';
    badge.textContent = '? unknown provider';
    badge.dataset.provider = 'unknown';
    hint.textContent = 'Key prefix not recognised — request will be sent as-is to the selected model.';
    Array.from(select.options).forEach(o => { o.hidden = false; o.disabled = false; });
    return;
  }

  badge.style.display = 'inline-flex';
  badge.textContent = `Detected: ${detected.label}`;
  badge.dataset.provider = detected.provider;

  let firstMatch = null;
  Array.from(select.options).forEach(o => {
    const match = modelProvider(o.value) === detected.provider;
    o.hidden = !match;
    o.disabled = !match;
    if (match && firstMatch == null) firstMatch = o.value;
  });

  if (firstMatch && modelProvider(select.value) !== detected.provider) {
    select.value = firstMatch;
  }

  if (firstMatch) {
    hint.textContent = `Model dropdown filtered to ${detected.label} models.`;
  } else {
    hint.textContent = `No ${detected.label} model in the dropdown — request will fail unless you change the key.`;
  }
}

// ── Validation ─────────────────────────────────────────────────────────────
function validate() {
  const url = $('repo-url').value.trim();
  const urlOk = GH_RE.test(url);
  if (url.length > 0) {
    $('repo-url').classList.toggle('invalid', !urlOk);
    $('url-error').textContent = urlOk ? '' : 'Must be a valid github.com URL';
  } else {
    $('repo-url').classList.remove('invalid');
    $('url-error').textContent = '';
  }
  let ready = urlOk;
  if (_mode === 'byok') {
    ready = ready
      && $('llm-key').value.trim().length > 0
      && $('e2b-key').value.trim().length > 0;
  }
  $('submit-btn').disabled = !ready;
}
$('repo-url').addEventListener('input', validate);
$('llm-key').addEventListener('input', () => { applyProviderFilter(); validate(); });
$('e2b-key').addEventListener('input', validate);
applyProviderFilter();

// ── Preflight + Submit ─────────────────────────────────────────────────────
function buildRunBody() {
  const repoUrl = $('repo-url').value.trim();
  const maxGaps = _mode === 'demo'
    ? Math.min(parseInt($('demo-max-gaps').value, 10) || 2, 3)
    : Math.min(parseInt($('byok-max-gaps').value, 10) || 5, 15);

  const body = {
    mode: _mode,
    repo_url: repoUrl,
    max_gaps: maxGaps,
    eval_strictness: _strictness[_mode],
  };

  if (_mode === 'byok') {
    body.llm_api_key = $('llm-key').value.trim();
    body.e2b_api_key = $('e2b-key').value.trim();
    body.model = $('model-select').value;
    const bt = $('braintrust-key').value.trim();
    if (bt) body.braintrust_api_key = bt;
  }
  return body;
}

function renderPreflight(state, report) {
  const strip = $('preflight-strip');
  strip.style.display = '';
  strip.classList.toggle('busy', state === 'busy');

  if (state === 'busy') {
    strip.innerHTML = `
      <div class="preflight-line">
        <span class="pf-label">Preflight</span>
        <span class="pf-icon pending">…</span>
        <span class="pf-msg">Verifying repo, LLM key, and sandbox…</span>
      </div>`;
    return;
  }

  const row = (label, check) => `
    <div class="preflight-line">
      <span class="pf-label">${label}</span>
      <span class="pf-icon ${check.ok ? 'ok' : 'fail'}">${check.ok ? '✓' : '✗'}</span>
      <span class="pf-msg ${check.ok ? '' : 'fail'}" title="${escHtml(check.detail || '')}">${escHtml(check.message || '')}</span>
    </div>`;

  strip.innerHTML = [
    row('Repo', report.repo),
    row('LLM',  report.llm),
    row('E2B',  report.e2b),
  ].join('');
}

async function runPreflight(body) {
  renderPreflight('busy');
  try {
    const res = await fetch('/api/preflight', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      renderPreflight('error', {
        repo: { ok: false, message: data.error || 'Preflight rejected.' },
        llm:  { ok: false, message: '—' },
        e2b:  { ok: false, message: '—' },
      });
      return null;
    }
    renderPreflight('done', data);
    return data;
  } catch (err) {
    renderPreflight('error', {
      repo: { ok: false, message: 'Network error — is the server running?' },
      llm:  { ok: false, message: '—' },
      e2b:  { ok: false, message: '—' },
    });
    return null;
  }
}

$('submit-btn').addEventListener('click', async () => {
  $('submit-btn').disabled = true;
  $('submit-error').textContent = '';

  const body = buildRunBody();

  // Step 1: preflight. Saves the user from waiting on a doomed pipeline.
  const pf = await runPreflight(body);
  if (!pf || !pf.ready) {
    $('submit-btn').disabled = false;
    $('submit-error').textContent = 'Fix the red checks above before running.';
    return;
  }

  // Step 2: actually start the run.
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      $('submit-error').textContent = data.error || 'Failed to start run.';
      $('submit-btn').disabled = false;
      return;
    }
    _runId = data.run_id;
    _repoUrl = body.repo_url;
    startProgress(body.repo_url, _mode);
  } catch {
    $('submit-error').textContent = 'Network error — is the server running?';
    $('submit-btn').disabled = false;
  }
});

// ── Progress view ──────────────────────────────────────────────────────────
function startProgress(repoUrl, mode) {
  Object.keys(_gaps).forEach(k => delete _gaps[k]);
  $('gap-table').innerHTML = '';
  $('log-content').innerHTML = '';

  $('prog-repo').textContent = repoName(repoUrl);
  $('prog-mode-badge').textContent = mode === 'byok' ? 'BYOK' : 'Demo';
  $('prog-headline').textContent = `${repoName(repoUrl)}…`;
  $('prog-status').textContent = 'Setting up sandbox…';
  $('prog-gap-counter').textContent = '— / —';
  $('prog-current-target').textContent = '—';

  resetAgentPipeline();
  document.querySelectorAll('.ca-nav-link[data-view="progress"]').forEach(a => a.classList.remove('disabled'));
  showView('progress');

  _runStartTs = Date.now();
  startRunTimer();

  _sse = new EventSource(`/api/run/${_runId}/events`);
  _sse.onmessage = handleSSE;
  _sse.onerror = () => {
    $('prog-status').textContent = 'Connection lost — check server logs.';
    _sse.close();
    stopRunTimer();
  };
}

let _activeAgent = '';

function startRunTimer() {
  stopRunTimer();
  _runTimer = setInterval(renderHeaderMeta, 1000);
}
function stopRunTimer() {
  if (_runTimer) { clearInterval(_runTimer); _runTimer = null; }
}

function renderHeaderMeta() {
  if (!_runStartTs) { $('header-meta').textContent = ''; return; }
  const sec = Math.floor((Date.now() - _runStartTs) / 1000);
  const mm = String(Math.floor(sec / 60)).padStart(2, '0');
  const ss = String(sec % 60).padStart(2, '0');
  const active = _activeAgent ? `· Running ${AGENT_LABELS[_activeAgent] || _activeAgent}` : '';
  $('header-meta').textContent = `Elapsed ${mm}:${ss} ${active}`.trim();
}

const AGENT_NAMES = ['context_architect', 'test_writer', 'eval_agent', 'execution_runner', 'regression_guard', 'result_summarizer'];

function resetAgentPipeline() {
  AGENT_NAMES.forEach(name => {
    const el = document.querySelector(`.ap-stage[data-agent="${name}"]`);
    if (!el) return;
    el.classList.remove('running', 'done');
    el.querySelector('.ap-status').textContent = 'idle';
  });
}

function setAgentState(name, state) {
  const el = document.querySelector(`.ap-stage[data-agent="${name}"]`);
  if (!el) return;
  el.classList.remove('running', 'done');
  if (state === 'running') el.classList.add('running');
  if (state === 'done')    el.classList.add('done');
  const statusEl = el.querySelector('.ap-status');
  if (statusEl) statusEl.textContent = state;
}

function handleSSE(evt) {
  const e = JSON.parse(evt.data);

  if (e.type === 'gap_start') {
    const { gap_idx: idx, total_gaps: total } = e.data;
    _gaps[e.gap_id] = { idx, total, status: 'running', agents: {} };
    $('prog-gap-counter').textContent = `${idx} / ${total}`;
    $('prog-current-target').textContent = e.gap_id;
    $('prog-status').textContent = `Working gap ${idx} of ${total}`;
    appendGapRow(e.gap_id, idx);
    AGENT_NAMES.forEach(n => setAgentState(n, 'idle'));

  } else if (e.type === 'agent_start') {
    const g = _gaps[e.gap_id];
    if (g) { g.agents[e.agent] = 'running'; updateGapRow(e.gap_id); }
    setAgentState(e.agent, 'running');
    _activeAgent = e.agent;
    renderHeaderMeta();

  } else if (e.type === 'agent_end') {
    const g = _gaps[e.gap_id];
    if (g) { g.agents[e.agent] = 'done'; updateGapRow(e.gap_id); }
    setAgentState(e.agent, 'done');
    if (_activeAgent === e.agent) _activeAgent = '';
    renderHeaderMeta();

  } else if (e.type === 'gap_end') {
    const g = _gaps[e.gap_id];
    if (g) {
      g.status = e.data.committed ? 'committed' : 'skipped';
      updateGapRow(e.gap_id);
    }

  } else if (e.type === 'log') {
    const msg = (e.data.msg || '').trim();
    if (msg) appendLog(msg);

  } else if (e.type === 'done') {
    _sse.close();
    $('prog-status').textContent = 'Complete — loading results…';
    AGENT_NAMES.forEach(n => setAgentState(n, 'done'));
    _activeAgent = '';
    stopRunTimer();
    _runStartTs = null;
    $('header-meta').textContent = '';
    fetchAndRenderResults();

  } else if (e.type === 'error') {
    _sse.close();
    $('prog-status').textContent = `Error: ${e.data.msg || 'unknown'}`;
    _activeAgent = '';
    stopRunTimer();
  }
}

const AGENT_LABELS = {
  context_architect:  'Context',
  test_writer:        'Writer',
  eval_agent:         'Eval',
  execution_runner:   'Runner',
  regression_guard:   'Guard',
  result_summarizer:  'Summary',
};

function appendGapRow(gapId, idx) {
  const row = document.createElement('div');
  row.className = 'gap-row';
  row.id = `gap-${gapId}`;
  row.innerHTML = `
    <span class="gap-idx">#${idx}</span>
    <span class="gap-badge running" id="gbadge-${gapId}">running</span>
    <span class="gap-agents" id="gagents-${gapId}"></span>
  `;
  $('gap-table').appendChild(row);
}

function updateGapRow(gapId) {
  const g = _gaps[gapId];
  if (!g) return;
  const badge = $(`gbadge-${gapId}`);
  if (badge) {
    badge.className = `gap-badge ${g.status}`;
    badge.textContent = g.status;
  }
  const agentsEl = $(`gagents-${gapId}`);
  if (agentsEl) {
    agentsEl.innerHTML = Object.entries(g.agents)
      .map(([agent, status]) => `<span class="agent-chip ${status}" data-agent="${agent}">${AGENT_LABELS[agent] || agent}</span>`)
      .join('');
  }
}

function appendLog(msg) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.textContent = msg;
  $('log-content').appendChild(div);
  $('log-content').scrollTop = $('log-content').scrollHeight;
}

// ── Results view ───────────────────────────────────────────────────────────
async function fetchAndRenderResults() {
  try {
    const res = await fetch(`/api/run/${_runId}`);
    const data = await res.json();
    _lastReport = data;
    renderResults(data);
  } catch {
    $('prog-status').textContent = 'Failed to load results — try refreshing.';
  }
}

function statusChipFor(r) {
  if (r.test_committed)                      return { cls: 'committed', label: 'Committed' };
  if (r.status === 'executed_missed_branch') return { cls: 'warn',      label: 'Missed branch' };
  if (r.status === 'sandbox_failed')         return { cls: 'fail',      label: 'Sandbox failed' };
  if (r.status === 'retry_budget_exhausted') return { cls: 'fail',      label: 'Retries exhausted' };
  if (r.status === 'eval_loop_exhausted')    return { cls: 'fail',      label: 'Retries exhausted' };
  return { cls: 'muted', label: r.status || 'no result' };
}

function renderScorecard(sc) {
  if (!sc) return;
  const delta = parseFloat(sc.avg_coverage_delta ?? 0) || 0;
  const committed = sc.tests_committed ?? 0;
  const targeted = sc.gaps_targeted ?? 0;
  $('scorecard-strip').innerHTML = `
    <div class="sc-cell">
      <span class="sc-cell-label">Gaps targeted</span>
      <span class="sc-cell-value">${targeted}</span>
    </div>
    <div class="sc-cell">
      <span class="sc-cell-label">Tests committed</span>
      <span class="sc-cell-value ${committed > 0 ? 'positive' : 'muted'}">${committed}</span>
    </div>
    <div class="sc-cell">
      <span class="sc-cell-label">Branch hit rate</span>
      <span class="sc-cell-value">${sc.branch_hit_rate ?? '—'}</span>
    </div>
    <div class="sc-cell">
      <span class="sc-cell-label">Green, no branch</span>
      <span class="sc-cell-value muted">${sc.tests_passed_no_branch ?? '—'}</span>
    </div>
    <div class="sc-cell">
      <span class="sc-cell-label">Coverage Δ</span>
      <span class="sc-cell-value ${delta > 0 ? 'positive' : 'muted'}">${sc.avg_coverage_delta ?? '—'}</span>
    </div>
    <div class="sc-cell">
      <span class="sc-cell-label">Avg loops</span>
      <span class="sc-cell-value muted">${sc.avg_loops ?? '—'}</span>
    </div>
    <div class="sc-cell">
      <span class="sc-cell-label">LLM cost</span>
      <span class="sc-cell-value muted">${sc.llm_cost ?? '—'}</span>
    </div>
  `;
}

function renderDiff(testCode, suggestedPath) {
  const code = (testCode || '').replace(/\s+$/, '');
  if (!code) return '';
  const lines = code.split('\n');
  const rows = lines.map((line, i) => `
    <div class="diff-line add">
      <span class="diff-ln">${i + 1}</span>
      <span class="diff-mark">+</span>
      <span class="diff-content">${escHtml(line)}</span>
    </div>
  `).join('');
  return `
    <div class="diff-card">
      <div class="diff-head">
        <span class="diff-plus">+</span>
        <span class="diff-path">${escHtml(suggestedPath)}</span>
        <span class="diff-tag">new file · ${lines.length} lines</span>
      </div>
      <div class="diff-body">${rows}</div>
    </div>
  `;
}

function suggestedTestPath(r) {
  const symbol = (r.target_symbol || 'unknown').replace(/[^A-Za-z0-9_]/g, '_').toLowerCase();
  const parts = (r.file_path || '').split('/');
  const stem = parts[parts.length - 1].replace(/\.py$/, '');
  return `tests/test_${stem}__${symbol}.py`;
}

function renderGapCard(r) {
  const chip = statusChipFor(r);
  const path = suggestedTestPath(r);
  const diff = renderDiff(r.test_code, path);

  const auxParts = [];
  if (r.critique && !r.test_committed) {
    auxParts.push(`
      <details>
        <summary>Eval critique (loop ${r.loops_taken})</summary>
        <pre>${escHtml(r.critique)}</pre>
      </details>
    `);
  }
  if (r.stderr_trace) {
    auxParts.push(`
      <details>
        <summary>Sandbox stderr</summary>
        <pre>${escHtml(r.stderr_trace)}</pre>
      </details>
    `);
  }
  if (r.original_code) {
    auxParts.push(`
      <details class="target-source">
        <summary>Targeted source</summary>
        <pre>${escHtml(r.original_code)}</pre>
      </details>
    `);
  }

  return `
    <article class="gap-card">
      <aside class="gap-meta-block">
        <span class="gap-meta-file">${escHtml(r.file_path)}</span>
        <span class="gap-meta-symbol">${escHtml(r.target_symbol)}()</span>
        <span class="gap-meta-branch">branch ${escHtml(r.branch)}</span>
        <span class="gap-status-chip ${chip.cls}">${chip.label}</span>
        <div class="gap-stats">
          <span>Loops: <b>${r.loops_taken ?? '—'}</b></span>
          ${r.assertion_score != null ? `<span>Assertion: <b>${r.assertion_score}/5</b></span>` : ''}
          ${(r.coverage_delta ?? 0) !== 0 ? `<span>Coverage Δ: <b>${(r.coverage_delta > 0 ? '+' : '') + r.coverage_delta.toFixed(2)}%</b></span>` : ''}
        </div>
      </aside>

      <div class="gap-body">
        ${r.skip_reason ? `
          <div class="gap-why">
            <span class="gap-why-label">Why this didn't commit</span>
            ${escHtml(r.skip_reason)}
          </div>` : ''}
        ${r.recommendation ? `
          <div class="gap-why gap-recommendation">
            <span class="gap-why-label">What to try next</span>
            ${escHtml(r.recommendation)}
          </div>` : ''}
        ${diff || `<div class="gap-why"><span class="gap-why-label">No draft</span>The writer didn't produce a candidate test for this gap.</div>`}
        ${auxParts.length ? `<div class="gap-aux">${auxParts.join('')}</div>` : ''}
      </div>
    </article>
  `;
}

function renderRunNarrative(sc) {
  const cards = [];

  const reg = sc.regression;
  if (reg && !reg.skipped) {
    const cls = reg.regression_detected ? 'warn' : 'clean';
    const label = reg.regression_detected ? 'Regression detected' : 'Suite clean';
    cards.push(`
      <div class="rn-card">
        <span class="rn-label ${cls}">${label}</span>
        <p class="rn-body">${escHtml(reg.summary || '')}</p>
        <div class="rn-stats">
          <div class="rn-stat">
            <span class="rn-stat-label">Baseline passing</span>
            <span class="rn-stat-value">${reg.baseline_passed}</span>
          </div>
          <div class="rn-stat">
            <span class="rn-stat-label">Post-commit passing</span>
            <span class="rn-stat-value">${reg.post_passed}</span>
          </div>
          <div class="rn-stat">
            <span class="rn-stat-label">New failures</span>
            <span class="rn-stat-value">${reg.new_failures}</span>
          </div>
        </div>
      </div>
    `);
  }

  const summary = sc.summary;
  if (summary) {
    if (summary.pr_description) {
      cards.push(`
        <div class="rn-card">
          <div class="rn-row">
            <span class="rn-label">PR description</span>
            <button class="rn-copy" data-copy="pr">Copy</button>
          </div>
          <pre class="rn-pre" id="rn-pr">${escHtml(summary.pr_description)}</pre>
        </div>
      `);
    }
    if (summary.full_summary) {
      cards.push(`
        <div class="rn-card">
          <div class="rn-row">
            <span class="rn-label">Run summary</span>
            <button class="rn-copy" data-copy="full">Copy</button>
          </div>
          <p class="rn-body" id="rn-full">${escHtml(summary.full_summary).replace(/\n\n/g, '</p><p>')}</p>
        </div>
      `);
    }
  }

  $('run-narrative').innerHTML = cards.join('');

  document.querySelectorAll('.rn-copy').forEach(btn => {
    btn.addEventListener('click', async () => {
      const which = btn.dataset.copy;
      const src = which === 'pr'
        ? (sc.summary && sc.summary.pr_description) || ''
        : (sc.summary && sc.summary.full_summary) || '';
      try {
        await navigator.clipboard.writeText(src);
        const orig = btn.textContent;
        btn.textContent = 'Copied ✓';
        setTimeout(() => { btn.textContent = orig; }, 1200);
      } catch {}
    });
  });
}

function renderResults(data) {
  const sc = data.scorecard || {};
  const results = data.results || [];

  $('result-repo').textContent = repoName(_repoUrl || sc.repo || '');
  document.querySelectorAll('.ca-nav-link[data-view="results"]').forEach(a => a.classList.remove('disabled'));

  renderScorecard(sc);
  renderRunNarrative(sc);

  if (results.length === 0) {
    $('gap-results').innerHTML = `<div class="gap-why"><span class="gap-why-label">No gaps</span>The pipeline didn't return any gap results.</div>`;
  } else {
    $('gap-results').innerHTML = results.map(renderGapCard).join('');
  }

  const committed = results.filter(r => r.test_committed && r.test_code);
  const dl = $('download-btn');
  if (committed.length > 0) {
    dl.href = `/api/run/${_runId}/zip`;
    dl.download = `coverage_tests_${_runId.slice(0, 8)}.zip`;
    dl.style.display = 'inline-flex';
  } else {
    dl.style.display = 'none';
  }

  $('header-meta').textContent = '';
  showView('results');
}

// ── Action buttons ─────────────────────────────────────────────────────────
$('new-run-btn').addEventListener('click', () => {
  if (_sse) { _sse.close(); _sse = null; }
  _runId = null;
  _repoUrl = null;
  _lastReport = null;
  $('submit-btn').disabled = true;
  $('preflight-strip').style.display = 'none';
  $('preflight-strip').innerHTML = '';
  $('submit-error').textContent = '';
  document.querySelectorAll('.ca-nav-link').forEach(a => {
    if (a.dataset.view !== 'landing') a.classList.add('disabled');
  });
  showView('landing');
  validate();
});

$('copy-json-btn').addEventListener('click', async () => {
  if (!_lastReport) return;
  const payload = JSON.stringify({
    scorecard: _lastReport.scorecard,
    results: _lastReport.results,
  }, null, 2);
  try {
    await navigator.clipboard.writeText(payload);
    $('copy-json-btn').textContent = 'Copied ✓';
    setTimeout(() => { $('copy-json-btn').textContent = 'Copy report JSON'; }, 1500);
  } catch {
    // ignore
  }
});

// Nav links — only honor non-disabled
document.querySelectorAll('.ca-nav-link').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    if (a.classList.contains('disabled')) return;
    showView(a.dataset.view);
  });
});

validate();
