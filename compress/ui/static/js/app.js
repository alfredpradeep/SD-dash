/**
 * Haiku Platform — COMPRESS Dashboard Application
 *
 * Handles:
 *   - Compress with trace via POST /compress/trace
 *   - Real-time pipeline flow visualization
 *   - Obsidian-style D3.js semantic graph
 *   - Embedding space scatter plot
 *   - Beam search candidate display
 *   - Tooltip system for all fields
 *   - Dynamic gate rendering
 *   - Savings card
 *   - Health polling
 */

(function () {
  "use strict";

  /* ── Constants ──────────────────────────────────────────────── */
  const API_BASE = window.location.origin;
  const HEALTH_POLL_MS = 30000;
  const TOKENIZER_ENCODING_MAP = {
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "claude-3-5-sonnet": "anthropic",
    "claude-3-haiku": "anthropic",
    "claude-3-opus": "anthropic",
    "llama-3-8b": "sentencepiece_llama3",
    "llama-3-70b": "sentencepiece_llama3",
    "mistral-7b": "sentencepiece_mistral",
    "mistral-8x7b": "sentencepiece_mistral",
  };

  const NODE_COLORS = {
    entity: "#3A7D44",
    event: "#1565C0",
    attribute: "#E65100",
    relation: "#6A1B9A",
    negation: "#C62828",
    temporal: "#00838F",
    quantifier: "#F57F17",
    modal: "#4E342E",
    sentiment: "#AD1457",
    condition: "#00695C",
  };

  /* ── DOM ────────────────────────────────────────────────────── */
  const $ = (id) => document.getElementById(id);

  const els = {
    inputText: $("inputText"),
    inputLang: $("inputLang"),
    inputTokenizer: $("inputTokenizer"),
    inputCustomer: $("inputCustomer"),
    inputMinReduction: $("inputMinReduction"),
    inputSemanticThresh: $("inputSemanticThresh"),
    inputGraphThresh: $("inputGraphThresh"),
    btnCompress: $("btnCompress"),
    btnClear: $("btnClear"),
    gateSemanticVal: $("gateSemanticVal"),
    gateSemanticIcon: $("gateSemanticIcon"),
    gateGraphVal: $("gateGraphVal"),
    gateGraphIcon: $("gateGraphIcon"),
    gateReductionVal: $("gateReductionVal"),
    gateReductionIcon: $("gateReductionIcon"),
    origText: $("origText"),
    compText: $("compText"),
    origTokenCount: $("origTokenCount"),
    compTokenCount: $("compTokenCount"),
    origEncoding: $("origEncoding"),
    compEncoding: $("compEncoding"),
    densityFill: $("densityFill"),
    densitySaved: $("densitySaved"),
    savingsAmount: $("savingsAmount"),
    savingsCents: $("savingsCents"),
    trendText: $("trendText"),
    lastRun: $("lastRun"),
    healthStatus: $("healthStatus"),
    healthExtractor: $("healthExtractor"),
    healthSearcher: $("healthSearcher"),
    healthGate: $("healthGate"),
    healthCache: $("healthCache"),
    vizSection: $("vizSection"),
    pipelineFlow: $("pipelineFlow"),
    graphContainer: $("graphContainer"),
    graphSvg: $("graphSvg"),
    graphNodeCount: $("graphNodeCount"),
    graphEdgeCount: $("graphEdgeCount"),
    graphDensity: $("graphDensity"),
    embeddingSvg: $("embeddingSvg"),
    embeddingSimLabel: $("embeddingSimLabel"),
    candidatesTable: $("candidatesTable"),
    stesCard: $("stesCard"),
    stesValue: $("stesValue"),
    stesProcessingMs: $("stesProcessingMs"),
    stesCandidates: $("stesCandidates"),
    gateReconVal: $("gateReconVal"),
    gateReconIcon: $("gateReconIcon"),
    reconstructionContent: $("reconstructionContent"),
    healthReconstructor: $("healthReconstructor"),
    healthDistiller: $("healthDistiller"),
    tooltipPopup: $("tooltipPopup"),
  };

  /* ── State ──────────────────────────────────────────────────── */
  let compressionHistory = [];
  let cumulativeSavings = 0;
  let isCompressing = false;
  let isDecompressing = false;
  let graphSimulation = null;
  let lastCompressionResult = null;

  /* ── Utilities ──────────────────────────────────────────────── */
  function setGateState(iconEl, valueEl, value, threshold, isPercent) {
    const numVal = parseFloat(value);
    const passed = numVal >= threshold;
    const displayVal = isPercent
      ? (numVal * 100).toFixed(1) + "%"
      : numVal.toFixed(2);
    valueEl.textContent = displayVal;
    iconEl.className = "gate-icon " + (passed ? "pass" : "fail");
    iconEl.innerHTML = passed
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function estimateMonthlySavings(reductionRatio, tokensPerRequest) {
    const costPer1kTokens = 0.002;
    const monthlyRequests = 50000;
    return (tokensPerRequest * reductionRatio / 1000) * costPer1kTokens * monthlyRequests;
  }

  function updateSavingsCard(reduction, origTokens) {
    const savings = estimateMonthlySavings(reduction, origTokens);
    cumulativeSavings += savings;
    const dollars = Math.floor(cumulativeSavings);
    const cents = Math.round((cumulativeSavings - dollars) * 100);
    els.savingsAmount.textContent = dollars.toLocaleString();
    els.savingsCents.textContent = "." + String(cents).padStart(2, "0");
    const avgReduction = compressionHistory.length > 0
      ? (compressionHistory.reduce((s, h) => s + h.reduction, 0) / compressionHistory.length * 100).toFixed(1)
      : "0.0";
    els.trendText.textContent = avgReduction + "% avg efficiency across " + compressionHistory.length + " runs";
    els.lastRun.textContent = "just now";
  }

  function setLoading(loading) {
    isCompressing = loading;
    els.btnCompress.disabled = loading;
    els.btnCompress.innerHTML = loading
      ? '<span class="spinner"></span>Compressing...'
      : "Compress";
  }

  /* ── Tooltip System ────────────────────────────────────────── */
  function initTooltips() {
    document.querySelectorAll(".tooltip-icon").forEach((icon) => {
      icon.addEventListener("mouseenter", function (e) {
        const tip = this.getAttribute("data-tip");
        if (!tip) return;
        els.tooltipPopup.textContent = tip;
        els.tooltipPopup.classList.add("visible");
        const rect = this.getBoundingClientRect();
        let left = rect.left + rect.width / 2 - 160;
        let top = rect.bottom + 8;
        if (left < 8) left = 8;
        if (left + 320 > window.innerWidth) left = window.innerWidth - 328;
        if (top + 100 > window.innerHeight) top = rect.top - 8 - 80;
        els.tooltipPopup.style.left = left + "px";
        els.tooltipPopup.style.top = top + "px";
      });
      icon.addEventListener("mouseleave", function () {
        els.tooltipPopup.classList.remove("visible");
      });
    });
  }

  /* ── Viz Tab Switching ─────────────────────────────────────── */
  function initVizTabs() {
    document.querySelectorAll(".viz-tab").forEach((tab) => {
      tab.addEventListener("click", function () {
        document.querySelectorAll(".viz-tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".viz-panel").forEach((p) => p.classList.remove("active"));
        this.classList.add("active");
        const panelId = "panel" + this.dataset.tab.charAt(0).toUpperCase() + this.dataset.tab.slice(1);
        const panel = $(panelId);
        if (panel) panel.classList.add("active");
      });
    });
  }

  /* ── Pipeline Flow Renderer ────────────────────────────────── */
  function renderPipeline(stages) {
    els.pipelineFlow.innerHTML = "";
    stages.forEach((stage, i) => {
      const stateClass = stage.status === "complete" ? "complete" : stage.status === "rejected" ? "rejected" : "waiting";

      // Support both formats: `details` (object) or `detail` (string)
      let detailsHtml = "";
      if (stage.details && typeof stage.details === "object") {
        detailsHtml = Object.entries(stage.details).map(([k, v]) => {
          const label = k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
          let val = v;
          if (typeof v === "number") val = v < 1 && v > 0 ? v.toFixed(3) : v;
          if (typeof v === "boolean") val = v ? "Pass" : "Fail";
          return '<div class="stage-detail-row"><span class="detail-label">' + label + '</span><span class="detail-value">' + val + '</span></div>';
        }).join("");
      } else if (stage.detail) {
        detailsHtml = '<div class="stage-detail-row"><span class="detail-value">' + escapeHtml(String(stage.detail)) + '</span></div>';
      }

      const timing = stage.duration_ms || stage.ms || 0;

      const html = '<div class="pipeline-stage">' +
        '<div class="stage-header">' +
          '<div class="stage-number ' + stateClass + '">' + (i + 1) + '</div>' +
          '<div><div class="stage-title">' + escapeHtml(stage.name) + '</div>' +
          '<div class="stage-subtitle">' + escapeHtml(stage.subtitle || stage.detail || "") + '</div></div>' +
        '</div>' +
        '<div class="stage-details">' + detailsHtml + '</div>' +
        '<div class="stage-timing">' + (typeof timing === "number" ? timing.toFixed(1) : timing) + ' ms</div>' +
      '</div>';
      els.pipelineFlow.insertAdjacentHTML("beforeend", html);
    });

    // Animate stages sequentially
    const stageEls = els.pipelineFlow.querySelectorAll(".pipeline-stage");
    stageEls.forEach((el, i) => {
      const numEl = el.querySelector(".stage-number");
      numEl.className = "stage-number waiting";
      setTimeout(() => {
        numEl.className = "stage-number running";
        setTimeout(() => {
          const status = stages[i].status;
          numEl.className = "stage-number " + (status === "rejected" ? "rejected" : "complete");
        }, 400 + i * 100);
      }, i * 500);
    });
  }

  /* ── D3 Semantic Graph (Obsidian-style) ────────────────────── */
  function renderGraph(graphData) {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
      els.graphSvg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#6B6B6B" font-size="14">No graph data available</text>';
      return;
    }

    // Clear previous
    d3.select("#graphSvg").selectAll("*").remove();
    if (graphSimulation) graphSimulation.stop();

    const container = els.graphContainer;
    const width = container.clientWidth || 700;
    const height = container.clientHeight || 380;

    const svg = d3.select("#graphSvg")
      .attr("viewBox", [0, 0, width, height])
      .attr("width", width)
      .attr("height", height);

    // Defs for arrow markers and glow
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#555");

    // Glow filter
    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "blur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    const nodes = graphData.nodes.map(d => ({ ...d }));
    const edges = graphData.edges.map(d => ({
      ...d,
      source: d.source,
      target: d.target,
    }));

    // Force simulation
    graphSimulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id(d => d.id).distance(80).strength(0.5))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(25));

    // Edge lines
    const link = svg.append("g")
      .selectAll("line")
      .data(edges)
      .join("line")
      .attr("stroke", "#444")
      .attr("stroke-width", d => Math.max(1, (d.weight || 1) * 2))
      .attr("stroke-opacity", 0.5)
      .attr("marker-end", "url(#arrow)");

    // Edge labels (support both `type` and `relation` field names)
    const linkLabel = svg.append("g")
      .selectAll("text")
      .data(edges)
      .join("text")
      .attr("font-size", 8)
      .attr("fill", "#666")
      .attr("text-anchor", "middle")
      .text(d => d.type || d.relation || "");

    // Node groups
    const node = svg.append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Node circles with glow
    node.append("circle")
      .attr("r", d => 6 + (d.confidence || 0.8) * 8)
      .attr("fill", d => NODE_COLORS[d.type] || "#888")
      .attr("stroke", d => d3.color(NODE_COLORS[d.type] || "#888").brighter(0.5))
      .attr("stroke-width", 1.5)
      .attr("filter", "url(#glow)")
      .attr("opacity", 0.9);

    // Node labels
    node.append("text")
      .attr("dx", 14)
      .attr("dy", 4)
      .attr("font-size", 10)
      .attr("fill", "#CCC")
      .attr("font-family", "Inter, sans-serif")
      .text(d => d.value.length > 20 ? d.value.slice(0, 18) + "..." : d.value);

    // Node title hover
    node.append("title")
      .text(d => d.type + ": " + d.value + "\nConfidence: " + (d.confidence || "—") + (d.source ? "\nSource: " + d.source : ""));

    graphSimulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      linkLabel
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);

      node.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
    });

    function dragstarted(event) {
      if (!event.active) graphSimulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }
    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }
    function dragended(event) {
      if (!event.active) graphSimulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    // Graph info
    els.graphNodeCount.textContent = nodes.length + " nodes";
    els.graphEdgeCount.textContent = edges.length + " edges";
    const maxEdges = nodes.length * (nodes.length - 1) / 2;
    const density = maxEdges > 0 ? (edges.length / maxEdges).toFixed(3) : "0";
    els.graphDensity.textContent = "density: " + density;
  }

  /* ── Embedding Space Visualization ─────────────────────────── */
  function renderEmbedding(embData) {
    if (!embData) {
      els.embeddingSvg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#6B6B6B" font-size="14">No embedding data</text>';
      return;
    }

    d3.select("#embeddingSvg").selectAll("*").remove();

    const container = $("embeddingContainer");
    const width = container.clientWidth || 700;
    const height = container.clientHeight || 380;
    const margin = 50;

    const svg = d3.select("#embeddingSvg")
      .attr("viewBox", [0, 0, width, height])
      .attr("width", width)
      .attr("height", height);

    // Gather all points (support both `context_points` and `candidates`)
    const contextPts = embData.context_points || embData.candidates || [];
    const allPoints = [
      { ...embData.original, type: "original" },
      { ...embData.compressed, type: "compressed" },
      ...contextPts.map(p => ({ ...p, type: "candidate" })),
    ];

    const xExtent = d3.extent(allPoints, d => d.x);
    const yExtent = d3.extent(allPoints, d => d.y);
    const pad = 1;

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - pad, xExtent[1] + pad])
      .range([margin, width - margin]);
    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - pad, yExtent[1] + pad])
      .range([height - margin, margin]);

    // Grid lines
    svg.append("g")
      .selectAll("line")
      .data(xScale.ticks(8))
      .join("line")
      .attr("x1", d => xScale(d)).attr("x2", d => xScale(d))
      .attr("y1", margin).attr("y2", height - margin)
      .attr("stroke", "#E0DDD6").attr("stroke-width", 0.5);
    svg.append("g")
      .selectAll("line")
      .data(yScale.ticks(6))
      .join("line")
      .attr("x1", margin).attr("x2", width - margin)
      .attr("y1", d => yScale(d)).attr("y2", d => yScale(d))
      .attr("stroke", "#E0DDD6").attr("stroke-width", 0.5);

    // Similarity radius circle
    const simRadius = embData.similarity_radius || 0.1;
    svg.append("circle")
      .attr("cx", xScale(embData.original.x))
      .attr("cy", yScale(embData.original.y))
      .attr("r", Math.abs(xScale(simRadius) - xScale(0)))
      .attr("fill", "none")
      .attr("stroke", "#3A7D44")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,4")
      .attr("opacity", 0.5);

    // Connection line between original and compressed
    svg.append("line")
      .attr("x1", xScale(embData.original.x))
      .attr("y1", yScale(embData.original.y))
      .attr("x2", xScale(embData.compressed.x))
      .attr("y2", yScale(embData.compressed.y))
      .attr("stroke", "#1565C0")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "6,3")
      .attr("opacity", 0.6);

    // Candidate points
    svg.selectAll(".candidate-dot")
      .data(contextPts)
      .join("circle")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 5)
      .attr("fill", d => d.passed ? "#3A7D44" : "#9E9E9E")
      .attr("opacity", 0.5)
      .append("title")
      .text(d => (d.label || "Candidate") + (d.similarity != null ? " (sim: " + d.similarity + ")" : ""));

    // Original point
    svg.append("circle")
      .attr("cx", xScale(embData.original.x))
      .attr("cy", yScale(embData.original.y))
      .attr("r", 10)
      .attr("fill", "#3A7D44")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);
    svg.append("text")
      .attr("x", xScale(embData.original.x) + 14)
      .attr("y", yScale(embData.original.y) + 4)
      .attr("fill", "#3A7D44")
      .attr("font-size", 12)
      .attr("font-weight", 600)
      .text("Original");

    // Compressed point
    svg.append("circle")
      .attr("cx", xScale(embData.compressed.x))
      .attr("cy", yScale(embData.compressed.y))
      .attr("r", 10)
      .attr("fill", "#1565C0")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);
    svg.append("text")
      .attr("x", xScale(embData.compressed.x) + 14)
      .attr("y", yScale(embData.compressed.y) + 4)
      .attr("fill", "#1565C0")
      .attr("font-size", 12)
      .attr("font-weight", 600)
      .text("Compressed");

    els.embeddingSimLabel.textContent = "Cosine similarity: —";
  }

  /* ── Beam Candidates Table ─────────────────────────────────── */
  function renderCandidates(candidates) {
    if (!candidates || candidates.length === 0) {
      els.candidatesTable.innerHTML = '<p style="color:var(--gray-400);text-align:center;padding:40px;">No candidates evaluated</p>';
      return;
    }

    let html = '<table><thead><tr>' +
      '<th>Rank</th><th>Candidate Text</th><th>STES</th>' +
      '<th>Semantic Sim</th><th>Graph Jaccard</th><th>Status</th>' +
      '</tr></thead><tbody>';

    candidates.forEach((c) => {
      const rowClass = c.passed ? ' class="winner"' : '';
      const badge = c.passed
        ? '<span class="candidate-badge pass">Selected</span>'
        : '<span class="candidate-badge fail">Rejected</span>';
      const semSim = c.semantic_similarity != null ? c.semantic_similarity : (c.semantic_sim || 0);
      const graphJacc = c.graph_jaccard != null ? c.graph_jaccard : (c.graph_jacc || 0);
      const stesScore = c.stes_score || c.stes || 0;
      html += '<tr' + rowClass + '>' +
        '<td>' + (c.rank != null ? c.rank + 1 : "—") + '</td>' +
        '<td style="font-family:var(--font-mono);font-size:0.78rem;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="' + escapeHtml(c.text) + '">' + escapeHtml(c.text) + '</td>' +
        '<td style="font-family:var(--font-mono);font-weight:600;">' + stesScore.toFixed(3) + '</td>' +
        '<td style="font-family:var(--font-mono);">' + semSim.toFixed(3) + '</td>' +
        '<td style="font-family:var(--font-mono);">' + graphJacc.toFixed(3) + '</td>' +
        '<td>' + badge + (c.reason && !c.passed ? '<div style="font-size:0.7rem;color:var(--gray-400);margin-top:2px;">' + escapeHtml(c.reason) + '</div>' : '') + '</td>' +
        '</tr>';
    });

    html += '</tbody></table>';
    els.candidatesTable.innerHTML = html;
  }

  /* ── Reconstruction Visualization ──────────────────────────── */
  function renderReconstruction(reconData) {
    if (!reconData || Object.keys(reconData).length === 0) {
      els.reconstructionContent.innerHTML = '<p style="color:var(--gray-400);text-align:center;padding:40px;">No reconstruction data — LLM or extractor unavailable</p>';
      return;
    }

    const score = reconData.score != null ? reconData.score : 1.0;
    const method = reconData.method || "unknown";
    const missing = reconData.missing_items || [];
    const scoreColor = score >= 0.85 ? "var(--green)" : "var(--red)";

    let html = '<div style="max-width:600px;margin:0 auto;">';
    html += '<div style="text-align:center;margin-bottom:24px;">';
    html += '<div style="font-size:3rem;font-weight:700;color:' + scoreColor + ';">' + (score * 100).toFixed(1) + '%</div>';
    html += '<div style="font-size:0.85rem;color:var(--gray-400);">Knowledge Preservation Score</div>';
    html += '<div style="font-size:0.75rem;color:var(--gray-500);margin-top:4px;">Method: ' + escapeHtml(method) + '</div>';
    html += '</div>';

    if (missing.length > 0) {
      html += '<div style="margin-top:16px;">';
      html += '<h4 style="font-size:0.85rem;color:var(--gray-300);margin-bottom:8px;">Missing Items (' + missing.length + ')</h4>';
      html += '<div style="background:rgba(198,40,40,0.1);border:1px solid rgba(198,40,40,0.3);border-radius:8px;padding:12px;">';
      missing.forEach(function(item) {
        html += '<div style="font-size:0.8rem;color:#ef9a9a;padding:4px 0;font-family:var(--font-mono);">' + escapeHtml(item) + '</div>';
      });
      html += '</div></div>';
    } else {
      html += '<div style="text-align:center;color:var(--green);font-size:0.85rem;">All critical information preserved</div>';
    }

    // Show KG sizes if available
    if (reconData.original_kg_size || reconData.original_nodes) {
      html += '<div style="display:flex;gap:16px;justify-content:center;margin-top:20px;">';
      if (reconData.original_kg_size) {
        html += '<div style="text-align:center;"><div style="font-size:1.2rem;font-weight:600;">' + reconData.original_kg_size + '</div><div style="font-size:0.7rem;color:var(--gray-400);">Original KG Items</div></div>';
        html += '<div style="text-align:center;"><div style="font-size:1.2rem;font-weight:600;">' + (reconData.compressed_kg_size || 0) + '</div><div style="font-size:0.7rem;color:var(--gray-400);">Compressed KG Items</div></div>';
      }
      if (reconData.original_nodes) {
        html += '<div style="text-align:center;"><div style="font-size:1.2rem;font-weight:600;">' + reconData.original_nodes + '</div><div style="font-size:0.7rem;color:var(--gray-400);">Original Nodes</div></div>';
        html += '<div style="text-align:center;"><div style="font-size:1.2rem;font-weight:600;">' + (reconData.compressed_nodes || 0) + '</div><div style="font-size:0.7rem;color:var(--gray-400);">Compressed Nodes</div></div>';
      }
      html += '</div>';
    }

    html += '</div>';
    els.reconstructionContent.innerHTML = html;
  }

  /* ── API: Compress with Trace ──────────────────────────────── */
  async function compressText() {
    if (isCompressing) return;
    const text = els.inputText.value.trim();
    if (!text) { els.inputText.focus(); return; }

    setLoading(true);

    const payload = {
      text: text,
      language: els.inputLang.value,
      target_tokenizer: els.inputTokenizer.value,
      customer_id: els.inputCustomer.value || "default",
      min_reduction_threshold: parseFloat(els.inputMinReduction.value),
      semantic_threshold: parseFloat(els.inputSemanticThresh.value),
      graph_threshold: parseFloat(els.inputGraphThresh.value),
    };

    try {
      const resp = await fetch(API_BASE + "/compress/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || "Compression failed");
      }

      const data = await resp.json();
      renderFullResult(data);
    } catch (err) {
      console.error("Compression error:", err);
      alert("Compression failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  /* ── Full Result Renderer ──────────────────────────────────── */
  function renderFullResult(data) {
    const result = data.result;
    const encoding = TOKENIZER_ENCODING_MAP[result.target_tokenizer] || "cl100k_base";

    // Save for decompression
    lastCompressionResult = data;

    // Show/hide decompress section for cross-lingual
    var decompSection = $("decompressSection");
    if (decompSection) {
      if (result.language !== "en" && result.compression_applied) {
        decompSection.classList.remove("hidden");
        // Reset decompress UI
        var decompFlow = $("decompressFlow");
        var decompResults = $("decompressResults");
        if (decompFlow) decompFlow.classList.add("hidden");
        if (decompResults) decompResults.classList.add("hidden");
        // Set flow tokens
        var dOrigLang = $("decompOrigLang");
        var langNames = {ta:"Tamil",hi:"Hindi",ar:"Arabic",ja:"Japanese",zh:"Chinese",ko:"Korean",bn:"Bengali",ur:"Urdu",te:"Telugu",ml:"Malayalam",pa:"Punjabi",gu:"Gujarati",mr:"Marathi",pt:"Portuguese",es:"Spanish",fr:"French",de:"German",id:"Indonesian",ms:"Malay"};
        if (dOrigLang) dOrigLang.textContent = langNames[result.language] || result.language;
        var dReconLang = $("decompReconLang");
        if (dReconLang) dReconLang.textContent = langNames[result.language] || result.language;
        var dOrigTok = $("decompOrigTokens");
        if (dOrigTok) dOrigTok.textContent = result.original_tokens;
        var dCompTok = $("decompCompTokens");
        if (dCompTok) dCompTok.textContent = result.compressed_tokens;
        var dCompSav = $("decompCompSavings");
        if (dCompSav) dCompSav.textContent = (result.reduction_ratio * 100).toFixed(1) + "% saved";
      } else {
        decompSection.classList.add("hidden");
      }
    }

    // Show viz section
    els.vizSection.classList.remove("hidden");

    // Gates
    setGateState(els.gateSemanticIcon, els.gateSemanticVal, result.semantic_similarity, parseFloat(els.inputSemanticThresh.value), false);
    setGateState(els.gateGraphIcon, els.gateGraphVal, result.graph_jaccard, parseFloat(els.inputGraphThresh.value), false);
    setGateState(els.gateReductionIcon, els.gateReductionVal, result.reduction_ratio, parseFloat(els.inputMinReduction.value), true);

    // Comparison
    els.origText.innerHTML = escapeHtml(result.original_text);
    els.compText.innerHTML = result.compression_applied
      ? escapeHtml(result.compressed_text)
      : '<em style="color:var(--gray-400)">No valid compression found — original returned. ' + escapeHtml(result.rejection_reason || "") + '</em>';
    els.origTokenCount.textContent = result.original_tokens + " tokens";
    els.compTokenCount.textContent = result.compressed_tokens + " tokens";
    els.origEncoding.textContent = encoding;
    els.compEncoding.textContent = encoding;

    // Density bar
    const reductionPct = (result.reduction_ratio * 100).toFixed(1);
    els.densityFill.style.width = reductionPct + "%";
    els.densitySaved.textContent = "+" + reductionPct + "% SAVED";

    // STES card
    els.stesCard.classList.remove("hidden");
    els.stesValue.textContent = result.stes_score.toFixed(3);
    els.stesProcessingMs.textContent = result.processing_ms.toFixed(1) + " ms";
    els.stesCandidates.textContent = result.candidates_evaluated + " candidates";

    // Savings
    compressionHistory.push({ reduction: result.reduction_ratio, origTokens: result.original_tokens, timestamp: Date.now() });
    if (result.compression_applied) {
      updateSavingsCard(result.reduction_ratio, result.original_tokens);
    }

    // Pipeline
    if (data.pipeline && data.pipeline.stages) {
      renderPipeline(data.pipeline.stages);
    }

    // Semantic graph
    if (data.semantic_graph) {
      renderGraph(data.semantic_graph);
    }

    // Embedding
    if (data.embedding_visualization) {
      renderEmbedding(data.embedding_visualization);
      els.embeddingSimLabel.textContent = "Cosine similarity: " + result.semantic_similarity.toFixed(4);
    }

    // Candidates
    if (data.beam_candidates) {
      renderCandidates(data.beam_candidates);
    }

    // Reconstruction gate — cross-lingual uses LaBSE proxy, threshold is lower
    if (data.reconstruction && data.reconstruction.score != null) {
      var isCrossLingual = (data.reconstruction.method === "crosslingual_labse_proxy");
      var reconThreshold = isCrossLingual ? 0.60 : 0.85;
      setGateState(els.gateReconIcon, els.gateReconVal, data.reconstruction.score, reconThreshold, false);
      var reconLabel = document.getElementById("gateReconLabel");
      if (reconLabel) {
        reconLabel.textContent = isCrossLingual
          ? "Cross-lingual LaBSE proxy ≥ 0.60"
          : "Knowledge preservation ≥ 0.85";
      }
      renderReconstruction(data.reconstruction);
    } else {
      els.gateReconVal.textContent = "—";
      renderReconstruction(null);
    }

    // Scroll to viz
    els.vizSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ── Health Check ──────────────────────────────────────────── */
  async function fetchHealth() {
    try {
      const resp = await fetch(API_BASE + "/compress/health");
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      els.healthStatus.textContent = data.status;
      els.healthStatus.style.color = data.status.includes("healthy") ? "var(--green)" : "var(--red)";
      els.healthExtractor.textContent = data.extractor;
      els.healthSearcher.textContent = data.searcher;
      els.healthGate.textContent = data.gate;
      if (els.healthReconstructor) els.healthReconstructor.textContent = data.reconstructor || "—";
      if (els.healthDistiller) els.healthDistiller.textContent = data.distiller || "—";
      els.healthCache.textContent = data.cache;
    } catch {
      els.healthStatus.textContent = "unreachable";
      els.healthStatus.style.color = "var(--red)";
    }
  }

  /* ── Clear ─────────────────────────────────────────────────── */
  function clearForm() {
    els.inputText.value = "";
    els.gateSemanticVal.textContent = "—";
    els.gateGraphVal.textContent = "—";
    els.gateReductionVal.textContent = "—";
    ["gateSemanticIcon", "gateGraphIcon", "gateReductionIcon"].forEach((id) => {
      const el = $(id);
      el.className = "gate-icon";
      el.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    });
    els.origText.innerHTML = "Submit a compression request below to see the original text here.";
    els.compText.innerHTML = "Compressed output will appear here after processing.";
    els.origTokenCount.textContent = "— tokens";
    els.compTokenCount.textContent = "— tokens";
    els.densityFill.style.width = "0%";
    els.densitySaved.textContent = "+0% SAVED";
    els.vizSection.classList.add("hidden");
    els.stesCard.classList.add("hidden");
  }

  /* ── Decompression ─────────────────────────────────────────── */
  async function decompressText() {
    if (!lastCompressionResult || isDecompressing) return;
    var result = lastCompressionResult.result;
    if (!result.compression_applied || result.language === "en") return;

    isDecompressing = true;
    var btn = $("btnDecompress");
    var origLabel = btn.textContent;
    btn.textContent = "Decompressing...";
    btn.disabled = true;

    // Show loading in results
    var decompResults = $("decompressResults");
    var decompText = $("decompText");
    decompResults.classList.remove("hidden");
    decompText.innerHTML = '<div class="decompress-loading"><div class="spinner"></div><div>Translating back to ' + ($("decompOrigLang").textContent || result.language) + '...</div></div>';

    try {
      var resp = await fetch(API_BASE + "/compress/decompress", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          compressed_text: result.compressed_text,
          target_language: result.language,
          original_text: result.original_text,
          n_candidates: 3,
        }),
      });

      if (!resp.ok) throw new Error("Decompression failed: " + resp.status);
      var data = await resp.json();

      // Show flow diagram
      var decompFlow = $("decompressFlow");
      if (decompFlow) decompFlow.classList.remove("hidden");

      // Update reconstructed tokens
      var dReconTok = $("decompReconTokens");
      if (dReconTok && data.tokens) dReconTok.textContent = data.tokens.decompressed;

      var dNetSav = $("decompNetSavings");
      if (dNetSav && data.tokens && data.tokens.net_reduction_vs_original != null) {
        var netPct = (data.tokens.net_reduction_vs_original * 100).toFixed(1);
        dNetSav.textContent = netPct + "% net savings";
      }

      // Show decompressed text (with error notice if applicable)
      if (data.error) {
        decompText.innerHTML = '<p style="color:var(--red);">' + escapeHtml(data.error) + '</p>' +
          '<p>' + escapeHtml(data.decompressed_text) + '</p>';
      } else {
        decompText.innerHTML = escapeHtml(data.decompressed_text);
      }

      var dTokenCount = $("decompTokenCount");
      if (dTokenCount && data.tokens) dTokenCount.textContent = data.tokens.decompressed + " tokens";

      // Density bar
      var dDensityFill = $("decompDensityFill");
      var dNetSaved = $("decompNetSaved");
      if (dDensityFill && data.tokens) {
        var netPctNum = data.tokens.net_reduction_vs_original * 100;
        dDensityFill.style.width = Math.max(0, netPctNum) + "%";
        if (dNetSaved) dNetSaved.textContent = "+" + netPctNum.toFixed(1) + "% NET SAVED";
      }

      // Quality metrics
      var dQuality = $("decompQuality");
      if (dQuality && data.quality && data.quality.labse_to_original != null) {
        dQuality.classList.remove("hidden");
        var dSimOrig = $("decompSimOriginal");
        var dSimComp = $("decompSimCompressed");
        var dTime = $("decompTime");
        if (dSimOrig) dSimOrig.textContent = data.quality.labse_to_original.toFixed(3);
        if (dSimComp && data.quality.labse_to_compressed != null) dSimComp.textContent = data.quality.labse_to_compressed.toFixed(3);
        if (dTime && data.processing_ms != null) dTime.textContent = data.processing_ms.toFixed(0) + " ms";
      }

      // Candidates
      var dCandSection = $("decompCandidates");
      var dCandList = $("decompCandList");
      if (dCandSection && dCandList && data.candidates && data.candidates.length > 1) {
        dCandSection.classList.remove("hidden");
        dCandList.innerHTML = data.candidates.map(function(c, i) {
          var simOrig = c.sim_to_original != null ? c.sim_to_original.toFixed(3) : '—';
          var simComp = c.sim_to_compressed != null ? c.sim_to_compressed.toFixed(3) : '—';
          var combined = c.combined_score != null ? c.combined_score.toFixed(3) : '—';
          return '<div class="cand-item' + (i === 0 ? ' selected' : '') + '">' +
            '<div>' + escapeHtml(c.text || '') + '</div>' +
            '<div class="cand-meta">' +
              '<span>Rank #' + (i+1) + '</span>' +
              '<span>LaBSE→Original: ' + simOrig + '</span>' +
              '<span>LaBSE→Compressed: ' + simComp + '</span>' +
              '<span>Combined: ' + combined + '</span>' +
            '</div></div>';
        }).join("");
      }

    } catch (err) {
      decompText.innerHTML = '<p style="color:var(--red);">Decompression failed: ' + escapeHtml(err.message) + '</p>';
    } finally {
      isDecompressing = false;
      btn.textContent = origLabel;
      btn.disabled = false;
    }
  }

  /* ── Navigation ────────────────────────────────────────────── */
  function initNavigation() {
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.addEventListener("click", function () {
        document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
        this.classList.add("active");
        const page = this.dataset.page;
        const pageNames = {
          dashboard: "Dashboard", models: "Models", governance: "Governance",
          costs: "Unit Costs", experiments: "Experiments", datasets: "Datasets",
          monitoring: "Monitoring", policies: "Policies", risk: "Risk Assessment",
          analytics: "Analytics", org: "Organization", settings: "Settings",
        };
        document.querySelector(".breadcrumb").innerHTML = (pageNames[page] || page) + ' &rsaquo; <strong>SCL Compression Engine</strong>';
      });
    });
  }

  /* ── Init ───────────────────────────────────────────────────── */
  function init() {
    els.btnCompress.addEventListener("click", compressText);
    els.btnClear.addEventListener("click", clearForm);
    var btnDecomp = $("btnDecompress");
    if (btnDecomp) btnDecomp.addEventListener("click", decompressText);

    document.addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        compressText();
      }
    });

    initNavigation();
    initTooltips();
    initVizTabs();

    fetchHealth();
    setInterval(fetchHealth, HEALTH_POLL_MS);

    els.savingsAmount.textContent = "0";
    els.savingsCents.textContent = ".00";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
