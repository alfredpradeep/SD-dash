const API_BASE = '/v4';
const COMPRESS_API = '/compress';

const SAMPLE_PROMPTS = [
  { label: 'Tamil', lang: 'ta', text: 'நான் முந்தைய ப்ராஜெக்ட்டில் மைக்ரோசர்வீசஸ் ஆர்க்கிடெக்சர் உபயோகித்தேன்' },
  { label: 'Hindi', lang: 'hi', text: 'मैंने पिछले प्रोजेक्ट में React और Node.js के साथ काम किया और माइक्रोसर्विसेज आर्किटेक्चर बनाया' },
  { label: 'Arabic', lang: 'ar', text: 'أريد إلغاء اشتراكي والحصول على استرداد كامل لأن الخدمة لم تعمل بشكل صحيح' },
  { label: 'Japanese', lang: 'ja', text: '前職でマイクロサービスアーキテクチャを使って大規模なシステムを構築しました' },
  { label: 'Korean', lang: 'ko', text: '이전 프로젝트에서 마이크로서비스 아키텍처를 사용하여 대규모 시스템을 구축했습니다' },
  { label: 'Bengali', lang: 'bn', text: 'আমার অর্ডার এখনও পৌঁছায়নি এবং আমি দ্রুত একটি আপডেট চাই কারণ এটি জরুরি' },
  { label: 'English', lang: 'en', text: 'I used microservices architecture with Kubernetes and Docker in my previous project to build a scalable distributed system' },
];

let state = {
  loading: false,
  result: null,
  health: null,
  activeTab: 'compress',
};

function render() {
  const app = document.getElementById('app');
  app.innerHTML = `
    ${renderHeader()}
    <div class="main">
      <div class="panel panel-left">
        ${renderInputPanel()}
      </div>
      <div class="panel results-area">
        ${renderResultsPanel()}
      </div>
    </div>
  `;
  attachEvents();
}

function renderHeader() {
  const statusClass = state.health?.status === 'ready' ? '' :
    state.health?.status === 'degraded' ? 'degraded' : 'failed';
  const statusText = state.health?.status || 'connecting...';

  return `
    <header class="header">
      <div class="header-left">
        <div class="logo">COMPRESS<span> v4.0</span></div>
        <span class="version-badge">Adaptive Multilingual Cost Intelligence</span>
      </div>
      <div class="status-indicator">
        <span class="status-dot ${statusClass}"></span>
        <span>${statusText} | Dev Mode</span>
      </div>
    </header>
  `;
}

function renderInputPanel() {
  return `
    <div class="panel-title">Input</div>
    <div class="tabs">
      <div class="tab ${state.activeTab === 'compress' ? 'active' : ''}" data-tab="compress">Compress</div>
      <div class="tab ${state.activeTab === 'trace' ? 'active' : ''}" data-tab="trace">Pipeline Trace</div>
      <div class="tab ${state.activeTab === 'engines' ? 'active' : ''}" data-tab="engines">Engines</div>
    </div>

    <label class="input-label">Try a sample prompt</label>
    <div class="samples">
      ${SAMPLE_PROMPTS.map((s, i) => `
        <div class="sample-chip" data-sample="${i}">${s.label}</div>
      `).join('')}
    </div>

    <div class="input-group">
      <label class="input-label">Text to compress</label>
      <textarea class="textarea" id="input-text" placeholder="Enter multilingual text here...">${state.result?.input?.text || ''}</textarea>
    </div>

    <div class="select-row">
      <div class="select-wrapper">
        <label class="input-label">Language</label>
        <select class="select" id="input-lang">
          <option value="auto">Auto-detect</option>
          <option value="ta">Tamil</option>
          <option value="hi">Hindi</option>
          <option value="ar">Arabic</option>
          <option value="ja">Japanese</option>
          <option value="zh">Chinese</option>
          <option value="ko">Korean</option>
          <option value="bn">Bengali</option>
          <option value="ur">Urdu</option>
          <option value="te">Telugu</option>
          <option value="ml">Malayalam</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="pt">Portuguese</option>
        </select>
      </div>
      <div class="select-wrapper">
        <label class="input-label">Tokenizer</label>
        <select class="select" id="input-tokenizer">
          <option value="gpt-4o">GPT-4o</option>
          <option value="gpt-4o-mini">GPT-4o-mini</option>
          <option value="claude-3-5-sonnet">Claude 3.5</option>
          <option value="llama-3-8b">Llama 3 8B</option>
        </select>
      </div>
    </div>

    <button class="btn btn-primary" id="btn-compress" ${state.loading ? 'disabled' : ''}>
      ${state.loading ? '<span class="spinner"></span> Processing...' : 'Compress'}
    </button>
  `;
}

function renderResultsPanel() {
  if (state.loading) {
    return `<div class="loading"><span class="spinner"></span> Running pipeline...</div>`;
  }

  if (state.activeTab === 'engines') {
    return renderEnginesView();
  }

  if (!state.result) {
    return `
      <div class="empty-state">
        <div class="empty-icon">&#x26A1;</div>
        <div class="empty-title">Ready to compress</div>
        <div class="empty-desc">Select a sample prompt or enter multilingual text to see the COMPRESS v4 pipeline in action.</div>
      </div>
    `;
  }

  if (state.activeTab === 'trace' && state.result.pipeline) {
    return renderTraceView();
  }

  return renderCompressView();
}

function renderCompressView() {
  const r = state.result;
  if (!r) return '';

  const isTrace = !!r.result;
  const data = isTrace ? r.result : r;

  const reductionPct = ((data.reduction_ratio || 0) * 100).toFixed(1);
  const applied = data.compression_applied;

  return `
    <div class="panel-title">Result</div>

    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Compression Output</span>
        <span class="result-badge ${applied ? 'badge-pass' : 'badge-kill'}">${applied ? 'APPLIED' : 'REJECTED'}</span>
      </div>
      <div class="compressed-output">${escapeHtml(data.compressed_text || data.original_text)}</div>
    </div>

    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Metrics</span>
        <span class="result-badge badge-mode-a">${data.language?.toUpperCase() || 'AUTO'}</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-value highlight">${reductionPct}%</div>
          <div class="metric-label">Reduction</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${data.original_tokens}</div>
          <div class="metric-label">Original</div>
        </div>
        <div class="metric-item">
          <div class="metric-value success">${data.compressed_tokens}</div>
          <div class="metric-label">Compressed</div>
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-value">${data.semantic_similarity?.toFixed(3) || '-'}</div>
          <div class="metric-label">Semantic Sim</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${data.graph_jaccard?.toFixed(3) || '-'}</div>
          <div class="metric-label">Graph Jaccard</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${data.stes_score?.toFixed(3) || '-'}</div>
          <div class="metric-label">STES Score</div>
        </div>
      </div>
    </div>

    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Processing</span>
      </div>
      <div class="cost-row">
        <span class="cost-label">Processing Time</span>
        <span class="cost-value">${data.processing_ms?.toFixed(1) || 0}ms</span>
      </div>
      <div class="cost-row">
        <span class="cost-label">Candidates Evaluated</span>
        <span class="cost-value">${data.candidates_evaluated || 0}</span>
      </div>
      <div class="cost-row">
        <span class="cost-label">Tokenizer</span>
        <span class="cost-value">${data.target_tokenizer || 'gpt-4o'}</span>
      </div>
      ${data.rejection_reason ? `
        <div class="cost-row">
          <span class="cost-label">Rejection Reason</span>
          <span class="cost-value" style="color: var(--error); font-size: 11px;">${data.rejection_reason}</span>
        </div>
      ` : ''}
    </div>
  `;
}

function renderTraceView() {
  const r = state.result;
  if (!r?.pipeline) return renderCompressView();

  const stages = r.pipeline.stages || [];

  return `
    <div class="panel-title">Pipeline Trace</div>

    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Pipeline Stages</span>
        <span class="result-badge badge-mode-a">${r.pipeline.total_ms?.toFixed(1) || 0}ms total</span>
      </div>
      <div class="pipeline-stages">
        ${stages.map((s, i) => {
          const passed = s.status === 'complete';
          return `
            <div class="stage-item">
              <span class="stage-num ${passed ? 'pass' : 'fail'}">${i + 1}</span>
              <span class="stage-name">${s.name}</span>
              <span class="stage-time">${s.duration_ms?.toFixed(1)}ms</span>
            </div>
          `;
        }).join('')}
      </div>
    </div>

    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Semantic Graph</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-value">${r.semantic_graph?.nodes?.length || 0}</div>
          <div class="metric-label">Nodes</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${r.semantic_graph?.edges?.length || 0}</div>
          <div class="metric-label">Edges</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${r.beam_candidates?.length || 0}</div>
          <div class="metric-label">Candidates</div>
        </div>
      </div>
    </div>

    ${renderCompressView()}
  `;
}

function renderEnginesView() {
  if (!state.health) {
    return `<div class="loading"><span class="spinner"></span> Loading engines...</div>`;
  }

  const caps = state.health.capabilities || {};
  const engines = state.health.engines || {};

  return `
    <div class="panel-title">Engine Status</div>
    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Capabilities</span>
        <span class="result-badge ${state.health.status === 'ready' ? 'badge-pass' : 'badge-kill'}">${state.health.status?.toUpperCase()}</span>
      </div>
      <div class="engines-grid">
        ${Object.entries(caps).map(([name, info]) => `
          <div class="engine-item">
            <span class="engine-dot ${info.status}"></span>
            <span class="engine-name">${formatEngineName(name)}</span>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="result-card">
      <div class="result-header">
        <span class="result-title">Six Engines</span>
      </div>
      <div class="engines-grid">
        ${Object.entries(engines).map(([name, status]) => `
          <div class="engine-item">
            <span class="engine-dot ${status === 'ready' ? 'ready' : status === 'degraded' ? 'degraded' : 'failed'}"></span>
            <span class="engine-name">${formatEngineName(name)}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function formatEngineName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function attachEvents() {
  document.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const idx = parseInt(chip.dataset.sample);
      const sample = SAMPLE_PROMPTS[idx];
      document.getElementById('input-text').value = sample.text;
      document.getElementById('input-lang').value = sample.lang;
    });
  });

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      state.activeTab = tab.dataset.tab;
      render();
    });
  });

  const btn = document.getElementById('btn-compress');
  if (btn) {
    btn.addEventListener('click', handleCompress);
  }
}

async function handleCompress() {
  const text = document.getElementById('input-text').value.trim();
  if (!text) return;

  const language = document.getElementById('input-lang').value;
  const tokenizer = document.getElementById('input-tokenizer').value;

  state.loading = true;
  state.result = null;
  render();

  try {
    if (state.activeTab === 'trace') {
      const resp = await fetch(`${COMPRESS_API}/trace`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language, tokenizer }),
      });
      state.result = await resp.json();
    } else {
      const resp = await fetch(`${COMPRESS_API}/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language, tokenizer }),
      });
      state.result = await resp.json();
    }
  } catch (err) {
    state.result = { error: err.message };
  }

  state.loading = false;
  render();
}

async function fetchHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    state.health = await resp.json();
  } catch {
    state.health = { status: 'disconnected', capabilities: {}, engines: {} };
  }
  render();
}

fetchHealth();
render();
