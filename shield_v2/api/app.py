"""
SHIELD v2 FastAPI application — production-grade endpoints.

Exposes:
  GET   /                          — SPA (Claude-aesthetic dashboards)
  GET   /health                    — health + algorithm availability
  GET   /v2/config                 — runtime config (providers, simulator flag)
  POST  /v2/config/simulator       — toggle simulator-only mode
  GET   /v2/info, /v2/info/{key}   — tooltip content
  POST  /v2/audit                  — blocking audit run
  POST  /v2/audit/stream           — SSE per-probe streaming audit (recommended)
  GET   /v2/audit/{scan_id}        — fetch a completed audit
  GET   /v2/audits                 — list audits (summary rows)
  POST  /v2/guard/inbound          — scan user message
  POST  /v2/guard/outbound         — scan AI response
  POST  /v2/guard/turn             — scan both directions
  POST  /v2/guard/analyze          — full-depth analysis (SSE-streamed per pillar)
  GET   /v2/sessions               — list tracked sessions
  GET   /v2/sessions/{id}          — inspect one session
  DELETE /v2/sessions/{id}         — reset a session
  GET   /v2/demo/manifold          — live manifold projection demo
  GET   /v2/demo/antigenic         — live antigenic drift demo
  GET   /v2/demo/attack_graph      — attack graph snapshot
  GET   /v2/calibration            — calibration metrics (ECE, reliability, confusion)
  POST  /v2/calibration/retrain    — re-fit + re-calibrate the judge
  GET   /v2/budget                 — LLM budget snapshot
  GET   /v2/metrics                — aggregate live KPIs for the overview dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shield_v2 import __version__, __codename__
from shield_v2.api.schemas import (
    AuditRequest, GuardRequest, GuardStreamTurn, HealthResponse,
)
from shield_v2.audit.scanner import AuditScanner
from shield_v2.audit.report_generator import ReportGenerator
from shield_v2.core.attack_graph import AttackGraph
from shield_v2.core.brain_llm import BrainLLM, BudgetConfig, LLMBudgetTracker
from shield_v2.core.distilled_judge import DistilledCrossLingualJudge
from shield_v2.core.manifold_alignment import (
    AntigenicDriftMutator, CrossLingualManifoldAligner,
)
from shield_v2.guard.scanner import GuardScanner
from shield_v2.guard.session_tracker import SessionTracker

logger = logging.getLogger(__name__)


# =============================================================================
# Env helper — pick up .env if present
# =============================================================================


def _load_dotenv() -> None:
    """Minimal dotenv loader so users can drop a .env next to the launcher."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Walk up a couple levels to find .env
    for parent in [os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        p = os.path.join(parent, ".env")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception as e:
                logger.debug("dotenv load failed: %s", e)
            return


_load_dotenv()


# =============================================================================
# Singleton state
# =============================================================================


class SHIELDv2State:
    def __init__(self):
        simulator_only = os.getenv("SHIELD_SIMULATOR_ONLY", "false").lower() in {
            "1", "true", "yes", "y",
        }
        self.brain = BrainLLM(
            budget=LLMBudgetTracker(BudgetConfig()),
            simulator_only=simulator_only,
        )
        self.judge = DistilledCrossLingualJudge()
        # Real training + calibration pass on startup. In production this
        # is replaced with LLM-generated pairs persisted to disk.
        pairs = DistilledCrossLingualJudge.synthetic_distillation_dataset(
            n_per_class=40,
        )
        self.judge.train_and_calibrate(pairs)
        self.tracker = SessionTracker()
        self.graph = AttackGraph.build_standard(
            target_langs=["en", "hi", "ta", "ar", "zh", "es", "fr", "ja", "ko"]
        )
        self.aligner = CrossLingualManifoldAligner(embed_dim=64)
        self._fit_synthetic_projections()
        self.guard = GuardScanner(
            judge=self.judge, brain=self.brain, tracker=self.tracker,
        )
        self.audit_results: Dict[str, Any] = {}
        # Rolling KPI windows
        self.kpi_events: List[Dict[str, Any]] = []

    def _fit_synthetic_projections(self) -> None:
        rng = np.random.default_rng(0)
        d = 64
        n = 40
        src = rng.standard_normal((n, d))
        src_refusal = rng.standard_normal(d)
        src_compliance = rng.standard_normal(d)
        for lang in ["hi", "ta", "ar", "zh", "es", "fr", "ja", "ko"]:
            rot = np.linalg.qr(rng.standard_normal((d, d)))[0]
            tgt = src @ rot + 0.01 * rng.standard_normal((n, d))
            tgt_refusal = src_refusal @ rot
            tgt_compliance = src_compliance @ rot
            self.aligner.fit(
                target_lang=lang,
                source_embeds=src, target_embeds=tgt,
                source_refusal=src_refusal, target_refusal=tgt_refusal,
                source_compliance=src_compliance, target_compliance=tgt_compliance,
            )

    def record_kpi(self, kind: str, value: float, **extra):
        self.kpi_events.append({
            "t": time.time(), "kind": kind, "value": value, **extra,
        })
        # Keep last 500
        if len(self.kpi_events) > 500:
            self.kpi_events = self.kpi_events[-500:]


_state: Optional[SHIELDv2State] = None


def get_state() -> SHIELDv2State:
    global _state
    if _state is None:
        _state = SHIELDv2State()
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SHIELD v2 (%s) starting...", __codename__)
    get_state()
    yield
    logger.info("SHIELD v2 shutting down.")


# =============================================================================
# Tooltip info
# =============================================================================


PILLAR_INFO = {
    "pillar1": {
        "title": "Pillar 1 — The Brain (LLM APIs)",
        "what": "Frontier LLM calls (Groq, Gemini, Anthropic, OpenAI) reserved for deep cultural/semantic reasoning, low-confidence escalations, and Adversarial Translation for distillation.",
        "why": "Some safety judgments require world knowledge — caste connotations in Tamil, hawala idioms in Arabic — that no static classifier can encode. The Brain handles exactly those, and nothing more.",
        "complexity": "Latency: 200–2000ms per call. Cost: ~$0.001–$0.004/call. Budget-gated + cached.",
    },
    "pillar2": {
        "title": "Pillar 2 — The Reflexes (Deterministic Algorithms)",
        "what": "Script transliteration (Devanagari↔Latin, Arabic normalization, Han simplification), BPE-drift fingerprinting, attack-graph routing via Dijkstra.",
        "why": "These jobs require zero semantics — just Unicode tables, regex, and graph traversal. Running them via an LLM is a 5000× cost premium for no accuracy gain.",
        "complexity": "All O(n) or O((V+E) log V). Sub-millisecond latency.",
    },
    "pillar3": {
        "title": "Pillar 3 — Distilled Cross-Lingual Toxicity Evaluator",
        "what": "A small classifier trained via Model Distillation, then Platt-calibrated on held-out data. Returns a verdict + confidence in <5ms. Escalates to Brain when confidence < 0.65.",
        "why": "Covers 80% of judgment calls at 1/500th the cost of the Brain. Calibrated confidence drives the Brain/Reflex router.",
        "complexity": "Inference <5ms on CPU. Training: logistic-regression baseline; pluggable 4-layer transformer in production.",
    },
    "pillar4": {
        "title": "Pillar 4 — Cross-Lingual Manifold Alignment",
        "what": "Per-language orthogonal projection P_L: ℝᵈ_en → ℝᵈ_L that maps the English safety boundary into each target language's embedding manifold, plus an Antigenic-Drift Mutator that searches the boundary-parallel subspace for adversarial variants.",
        "why": "English-trained safety boundaries don't project into other languages' latent spaces. This layer projects them — and attacks them.",
        "complexity": "Fit: O(d³) per language. Projection: O(d²). Drift step: O(k·d²).",
    },
    "bidirectional": {
        "title": "Bidirectional Risk Tracking",
        "what": "Four risk streams per interaction: Inbound (user→AI), Outbound (AI→user), Cross-Channel correlation, and Session Trajectory (EWMA α=0.35 across turns).",
        "why": "Current safety tools only check outputs. In agentic products, the user-side attack surface is equally critical — and cross-channel correlation is where most incidents actually originate.",
        "complexity": "Real-time per-turn, P95 <80ms.",
    },
    "guard": {
        "title": "SHIELD Guard (Real-Time Inline)",
        "what": "Inline guardrail that scans every turn in both directions, P95 latency <80ms. SDK / sidecar proxy / gateway plugin.",
        "why": "Turns SHIELD from a pre-deployment audit into a production guardrail.",
        "complexity": "Distilled path <50ms. Brain escalation <120ms with timeout fallback.",
    },
    "audit": {
        "title": "SHIELD Audit (Batch Red-Teaming)",
        "what": "Runs the full 4-agent ensemble against a target LLM, produces the Safety Gap Matrix + monetary exposure + compliance attestations (EU AI Act, ISO 42001, NIST AI RMF, GDPR Article 22).",
        "why": "The CISO's artifact. Replaces 6–10 weeks of legal review per language with a streaming 4-hour automated run.",
        "complexity": "Scales with languages × categories × probes per cell, bounded per-cell concurrency.",
    },
    "consensus": {
        "title": "Bounded-Rounds Consensus Protocol",
        "what": "4 agents (Syntactic Mutator, Cultural Contextualizer, Target Oracle, Safety Verifier) run up to N rounds. Commits on verdict-margin crossing, quiesces on stable verdict, else escalates.",
        "why": "Provably terminates in O(max_rounds · n_agents) — no infinite translation loops.",
        "complexity": "Max 3 rounds × 4 agents per probe.",
    },
    "antigenic": {
        "title": "Antigenic Drift Mutator",
        "what": "Cross-disciplinary import from virology. Adversarial prompts evolve in the boundary-parallel subspace (orthogonal to the refusal direction) while a semantic anchor preserves meaning — exactly how influenza evolves to escape antibodies.",
        "why": "Finds adversarial variants that human red-teamers miss — searches directions that preserve semantic intent but geometrically evade the classifier.",
        "complexity": "O(T · k · d²) for T steps, k candidates, d-dim embedding.",
    },
    "calibration": {
        "title": "Confidence Calibration",
        "what": "Platt scaling on held-out data so that reported confidence equals empirical accuracy. Surfaces ECE (Expected Calibration Error), reliability curves, per-class AUC, and confusion matrix.",
        "why": "The Brain/Reflex router uses confidence to decide when to escalate. If confidence is miscalibrated, we over-call the Brain (cost) or under-call it (safety).",
        "complexity": "Fit: O(n) gradient descent on sigmoid — ~50ms for thousands of points.",
    },
    "simulator": {
        "title": "Simulator Toggle",
        "what": "When enabled, all Brain calls route through a deterministic local simulator that returns well-formed verdicts matching the real LLM schema. All other pillars run identically.",
        "why": "Guarantees reproducible demos, offline CI, and graceful degradation if API keys are missing. Flip back to real providers with one click.",
        "complexity": "~0ms per call (deterministic).",
    },
}


# =============================================================================
# SSE helpers
# =============================================================================


def _sse(event: str, data: Dict[str, Any]) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


# =============================================================================
# Extra request schemas
# =============================================================================


class SimulatorToggle(BaseModel):
    enabled: bool


class AnalyzeTurnRequest(BaseModel):
    user_message: str
    ai_response: str
    language: str = "en"
    session_id: str = "analyze"
    harm_category: str = "general"


class RetrainReq(BaseModel):
    n_per_class: int = 40
    calibration_frac: float = 0.25


# =============================================================================
# App factory
# =============================================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="SHIELD v2 — Cross-Lingual Guardrail Auditor",
        description=f"Code name: {__codename__}. Four-pillar architecture.",
        version=__version__,
        lifespan=lifespan,
    )

    static_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ui", "static")
    )
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        ui_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "ui", "templates", "index.html",
        ))
        if os.path.exists(ui_path):
            return FileResponse(ui_path)
        return HTMLResponse("<h1>SHIELD v2</h1><p>UI template missing.</p>")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version=__version__,
            pillars={
                "pillar1_brain": "active",
                "pillar2_reflexes": "active",
                "pillar3_distilled": "active",
                "pillar4_manifold": "active",
            },
            algorithms={
                "manifold_alignment": True,
                "antigenic_drift": True,
                "syntactic_mutator": True,
                "attack_graph": True,
                "distilled_judge": True,
                "brain_llm": True,
                "bounded_consensus": True,
                "bidirectional_risk": True,
                "session_tracker": True,
            },
        )

    @app.get("/v2/config")
    async def get_config():
        st = get_state()
        return {
            "simulator_only": st.brain.simulator_only,
            "configured_providers": st.brain.configured_providers,
            "call_counts_by_provider": st.brain.call_counts,
            "version": __version__,
            "codename": __codename__,
        }

    @app.post("/v2/config/simulator")
    async def set_simulator(tgl: SimulatorToggle):
        st = get_state()
        st.brain.set_simulator_only(tgl.enabled)
        return {"simulator_only": st.brain.simulator_only}

    # -------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------
    @app.get("/v2/info/{key}")
    async def info(key: str):
        if key not in PILLAR_INFO:
            raise HTTPException(status_code=404, detail=f"unknown key: {key}")
        return PILLAR_INFO[key]

    @app.get("/v2/info")
    async def info_all():
        return PILLAR_INFO

    # -------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------
    @app.post("/v2/audit")
    async def run_audit(req: AuditRequest):
        st = get_state()
        scanner = AuditScanner(
            brain=st.brain, judge=st.judge, aligner=st.aligner,
            attack_graph=st.graph,
            target_endpoint=req.target_endpoint,
            target_api_key=req.target_api_key,
            target_model=req.target_model,
        )
        report = await scanner.run(
            languages=req.languages,
            categories=req.categories,
            n_probes_per_cell=req.n_probes_per_cell,
        )
        doc = ReportGenerator.to_json_dict(report)
        st.audit_results[report.scan_id] = doc
        st.record_kpi("audit", report.bypass_rate,
                      scan_id=report.scan_id, total=report.total_probes)
        return doc

    @app.post("/v2/audit/stream")
    async def stream_audit(req: AuditRequest):
        """Server-sent events: one message per probe completion."""
        st = get_state()
        scanner = AuditScanner(
            brain=st.brain, judge=st.judge, aligner=st.aligner,
            attack_graph=st.graph,
            target_endpoint=req.target_endpoint,
            target_api_key=req.target_api_key,
            target_model=req.target_model,
        )

        async def gen() -> AsyncIterator[bytes]:
            yield _sse("hello", {"version": __version__,
                                 "simulator_only": st.brain.simulator_only})
            report = None
            async for ev in scanner.run_stream(
                languages=req.languages,
                categories=req.categories,
                n_probes_per_cell=req.n_probes_per_cell,
            ):
                kind = ev.get("event", "message")
                if kind == "completed":
                    report = ev["report"]
                    doc = ReportGenerator.to_json_dict(report)
                    st.audit_results[report.scan_id] = doc
                    st.record_kpi("audit", report.bypass_rate,
                                  scan_id=report.scan_id,
                                  total=report.total_probes)
                    yield _sse("completed", doc)
                else:
                    yield _sse(kind, ev)
            yield _sse("__close__", {"ok": True})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/v2/audit/{scan_id}")
    async def get_audit(scan_id: str):
        st = get_state()
        if scan_id not in st.audit_results:
            raise HTTPException(status_code=404, detail="not found")
        return st.audit_results[scan_id]

    @app.get("/v2/audits")
    async def list_audits():
        st = get_state()
        return [
            {
                "scan_id": sid,
                "status": r.get("status"),
                "total_probes": r.get("total_probes"),
                "bypass_rate": r.get("bypass_rate"),
                "languages": r.get("languages"),
                "categories": r.get("categories"),
                "monetary_exposure_usd": r.get("monetary_exposure_usd"),
                "duration_ms": r.get("duration_ms"),
            }
            for sid, r in st.audit_results.items()
        ]

    # -------------------------------------------------------------------
    # Guard
    # -------------------------------------------------------------------
    @app.post("/v2/guard/inbound")
    async def guard_inbound(req: GuardRequest):
        st = get_state()
        v = await st.guard.scan_inbound(
            text=req.text, language=req.language, session_id=req.session_id,
        )
        st.record_kpi("guard_inbound", v.inbound_risk,
                      action=v.action, language=v.language)
        return v.as_dict()

    @app.post("/v2/guard/outbound")
    async def guard_outbound(req: GuardRequest):
        if not req.prompt:
            raise HTTPException(
                status_code=400,
                detail="outbound scan requires the original prompt in the 'prompt' field",
            )
        st = get_state()
        v = await st.guard.scan_outbound(
            prompt=req.prompt, response=req.text,
            language=req.language, session_id=req.session_id,
        )
        st.record_kpi("guard_outbound", v.outbound_risk,
                      action=v.action, language=v.language)
        return v.as_dict()

    @app.post("/v2/guard/turn")
    async def guard_turn(req: GuardStreamTurn):
        st = get_state()
        inb = await st.guard.scan_inbound(
            text=req.user_message, language=req.language, session_id=req.session_id,
        )
        outb = await st.guard.scan_outbound(
            prompt=req.user_message, response=req.ai_response,
            language=req.language, session_id=req.session_id,
        )
        st.record_kpi("guard_turn", max(inb.inbound_risk, outb.outbound_risk),
                      language=req.language)
        return {"inbound": inb.as_dict(), "outbound": outb.as_dict()}

    @app.post("/v2/guard/analyze")
    async def guard_analyze(req: AnalyzeTurnRequest):
        """Full-depth analysis: runs every pillar, streams per-pillar result via SSE.

        This exercises the entire system end-to-end on a single long turn:
          1. Reflexes:  BPE drift + jailbreak/landmine regex
          2. Distilled: fast-path judge verdict + calibrated confidence
          3. Manifold:  per-language safety projection + antigenic-drift escape prob
          4. Brain:     cultural verifier + escalation judge (LLM or simulator)
          5. Session:   updated EWMA trajectory
        """
        st = get_state()

        async def gen() -> AsyncIterator[bytes]:
            yield _sse("started", {
                "language": req.language,
                "simulator_only": st.brain.simulator_only,
                "configured_providers": st.brain.configured_providers,
            })

            # Pillar 2: reflex scan (inbound + outbound)
            t0 = time.time()
            inb = await st.guard.scan_inbound(
                text=req.user_message, language=req.language,
                session_id=req.session_id,
            )
            outb = await st.guard.scan_outbound(
                prompt=req.user_message, response=req.ai_response,
                language=req.language, session_id=req.session_id,
            )
            yield _sse("reflex", {
                "latency_ms": int((time.time() - t0) * 1000),
                "inbound": inb.as_dict(),
                "outbound": outb.as_dict(),
            })

            # Pillar 3: distilled judge on the outbound
            t0 = time.time()
            judge_v = st.judge.judge(req.user_message, req.ai_response, req.language)
            yield _sse("distilled", {
                "latency_ms": int((time.time() - t0) * 1000),
                "verdict": judge_v.verdict,
                "confidence": round(judge_v.confidence, 3),
                "score": round(judge_v.score, 3),
                "should_escalate": bool(judge_v.should_escalate),
                "feature_summary": {
                    "refusal_hits": int(judge_v.features.refusal_hits),
                    "compliance_hits": int(judge_v.features.compliance_hits),
                    "harm_density": round(judge_v.features.harm_keyword_density, 4),
                    "script_consistency": round(judge_v.features.script_consistency, 3),
                },
            })

            # Pillar 4: manifold + antigenic drift estimate
            t0 = time.time()
            proj = st.aligner.projections.get(req.language[:2])
            if proj is None:
                proj = next(iter(st.aligner.projections.values()))
            rng = np.random.default_rng(abs(hash(req.user_message)) % (2**31))
            x0 = rng.standard_normal(proj.rotation.shape[0])
            mutator = AntigenicDriftMutator(max_steps=8)
            x_final, traj = mutator.drift(x0, proj)
            yield _sse("manifold", {
                "latency_ms": int((time.time() - t0) * 1000),
                "language": req.language[:2],
                "boundary_fidelity": round(proj.boundary_fidelity, 3),
                "margin": round(proj.margin, 3),
                "safety_score_initial": round(proj.safety_score(x0), 3),
                "safety_score_final": round(proj.safety_score(x_final), 3),
                "escape_probability": round(
                    mutator.escape_probability(x0, proj, n_trials=6), 3,
                ),
                "trajectory": [
                    {"step": s.step_index,
                     "safety_score": round(s.safety_score, 3),
                     "margin_drop": round(s.margin_drop, 3)}
                    for s in traj
                ],
            })

            # Pillar 1: Brain — only if the distilled path wasn't confident
            t0 = time.time()
            brain_out = None
            if judge_v.should_escalate or outb.cultural_landmine:
                brain_out = await st.brain.escalation_judge(
                    req.user_message, req.ai_response, req.language,
                    scan_id=req.session_id,
                )
                cultural = await st.brain.verify_cultural_risk(
                    req.user_message, req.ai_response, req.language,
                    scan_id=req.session_id,
                )
                yield _sse("brain", {
                    "latency_ms": int((time.time() - t0) * 1000),
                    "verdict": brain_out.get("verdict"),
                    "confidence": brain_out.get("confidence"),
                    "reasoning": brain_out.get("reasoning"),
                    "cultural_risk": cultural.get("cultural_risk"),
                    "cultural_recommendation": cultural.get("recommendation"),
                    "cultural_explanation": cultural.get("explanation"),
                    "provider_counts": dict(st.brain.call_counts),
                })
            else:
                yield _sse("brain", {
                    "skipped": True,
                    "reason": (
                        "Distilled-path confidence above threshold — no escalation."
                    ),
                    "latency_ms": 0,
                })

            # Aggregate decision
            session = st.tracker.get(req.session_id)
            decision = "PASS"
            if outb.action == "BLOCK" or inb.action == "ESCALATE":
                decision = "BLOCK"
            elif outb.action == "TRANSFORM" or inb.action in ("REDACT", "TRANSFORM"):
                decision = "MITIGATE"

            yield _sse("final", {
                "decision": decision,
                "inbound_risk": round(inb.inbound_risk, 3),
                "outbound_risk": round(outb.outbound_risk, 3),
                "cross_channel_risk": round(outb.cross_channel_risk, 3),
                "trajectory_risk": round(session.trajectory_risk, 3),
                "turn_count": session.turn_count,
                "jailbreak_attempts": session.jailbreak_attempts,
                "cultural_landmines": session.cultural_landmines,
            })
            yield _sse("__close__", {"ok": True})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # -------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------
    @app.get("/v2/sessions")
    async def list_sessions():
        st = get_state()
        return st.tracker.snapshot_all()

    @app.get("/v2/sessions/{sid}")
    async def get_session(sid: str):
        st = get_state()
        return st.tracker.get(sid).as_dict()

    @app.delete("/v2/sessions/{sid}")
    async def delete_session(sid: str):
        st = get_state()
        st.tracker.reset(sid)
        return {"ok": True}

    # -------------------------------------------------------------------
    # Demos
    # -------------------------------------------------------------------
    @app.get("/v2/demo/manifold")
    async def demo_manifold():
        st = get_state()
        out = []
        rng = np.random.default_rng(42)
        for lang, proj in st.aligner.projections.items():
            x_en = rng.standard_normal(proj.rotation.shape[0])
            projected = proj.project(x_en)
            safety = proj.safety_score(projected)
            out.append({
                "language": lang,
                "boundary_fidelity": round(proj.boundary_fidelity, 3),
                "margin": round(proj.margin, 3),
                "safety_score_of_random_point": round(safety, 3),
                "n_pairs": proj.n_pairs,
            })
        return {"projections": out,
                "thin_boundary": st.aligner.thin_boundary_languages(threshold=0.6)}

    @app.get("/v2/demo/antigenic")
    async def demo_antigenic(lang: str = "ta"):
        st = get_state()
        if lang not in st.aligner.projections:
            raise HTTPException(status_code=404, detail="no projection for language")
        proj = st.aligner.projections[lang]
        mutator = AntigenicDriftMutator(max_steps=12)
        x0 = np.random.default_rng(7).standard_normal(proj.rotation.shape[0])
        x_final, traj = mutator.drift(x0, proj)
        return {
            "language": lang,
            "initial_score": round(proj.safety_score(x0), 3),
            "final_score": round(proj.safety_score(x_final), 3),
            "trajectory_length": len(traj),
            "steps": [
                {"step": s.step_index, "safety_score": round(s.safety_score, 3),
                 "margin_drop": round(s.margin_drop, 3), "accepted": s.accepted}
                for s in traj
            ],
            "escape_probability_estimate": round(
                mutator.escape_probability(x0, proj, n_trials=8), 3,
            ),
        }

    @app.get("/v2/demo/attack_graph")
    async def demo_attack_graph(target: str = "ta"):
        """Expose the attack graph + a shortest-path route for visualization."""
        from shield_v2.core.attack_graph import AttackRouter, LanguageNode
        st = get_state()
        graph = st.graph

        nodes = list(graph.nodes.keys())[:40]
        edges = []
        for src_key, edge_list in list(graph.edges.items())[:40]:
            for e in edge_list[:6]:
                edges.append({
                    "source": src_key,
                    "target": e.dst.key(),
                    "cost": round(e.cost, 3),
                    "transform": e.transform,
                })
            if len(edges) >= 120:
                break
        edges = edges[:120]

        # Shortest path from English to a representative target-language node
        path_nodes = []
        path_cost = 0.0
        try:
            router = AttackRouter(graph)
            src = LanguageNode("en", "native", "formal", 0)
            dst_candidates = [
                LanguageNode(target, "native", "formal", 0),
                LanguageNode(target, "native", "colloquial", 0),
            ]
            path = None
            for dst in dst_candidates:
                if dst.key() in graph.nodes:
                    path = router.shortest_path(src, dst)
                    break
            if path:
                path_nodes = [n.key() for n in path.nodes]
                path_cost = round(path.total_cost, 3)
        except Exception as e:
            logger.debug("attack_graph path err: %s", e)

        return {
            "nodes": nodes,
            "edges": edges,
            "source": "en",
            "target": target,
            "path": path_nodes,
            "path_cost": path_cost,
        }

    # -------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------
    @app.get("/v2/calibration")
    async def calibration():
        st = get_state()
        if not st.judge.calibration_metrics:
            raise HTTPException(status_code=404,
                                detail="judge has not been calibrated yet")
        return st.judge.calibration_metrics

    @app.post("/v2/calibration/retrain")
    async def retrain_calibrate(req: RetrainReq):
        st = get_state()
        pairs = DistilledCrossLingualJudge.synthetic_distillation_dataset(
            n_per_class=req.n_per_class,
        )
        metrics = st.judge.train_and_calibrate(
            pairs, calibration_frac=req.calibration_frac,
        )
        return metrics

    # -------------------------------------------------------------------
    # Aggregate KPIs / budget
    # -------------------------------------------------------------------
    @app.get("/v2/budget")
    async def budget_snapshot():
        st = get_state()
        return st.brain.budget.snapshot()

    @app.get("/v2/metrics")
    async def metrics():
        """Live roll-up for the Overview dashboard."""
        st = get_state()
        now = time.time()
        recent = [e for e in st.kpi_events if now - e["t"] < 3600]
        by_kind: Dict[str, List[float]] = {}
        for e in recent:
            by_kind.setdefault(e["kind"], []).append(e["value"])

        guard_inbound = by_kind.get("guard_inbound", [])
        guard_outbound = by_kind.get("guard_outbound", [])
        audits = by_kind.get("audit", [])

        # Session tracker snapshot
        sessions = st.tracker.snapshot_all()
        n_sessions = len(sessions)
        hot = [s for s in sessions if s.get("trajectory_risk", 0) >= 0.5]

        # Provider counts
        prov = dict(st.brain.call_counts)
        calibration = st.judge.calibration_metrics or {}

        return {
            "timestamp": now,
            "counts": {
                "audits_last_hour": len(audits),
                "guard_inbound_last_hour": len(guard_inbound),
                "guard_outbound_last_hour": len(guard_outbound),
                "active_sessions": n_sessions,
                "hot_sessions": len(hot),
            },
            "averages": {
                "avg_inbound_risk": round(
                    sum(guard_inbound) / len(guard_inbound), 4
                ) if guard_inbound else 0.0,
                "avg_outbound_risk": round(
                    sum(guard_outbound) / len(guard_outbound), 4
                ) if guard_outbound else 0.0,
                "avg_bypass_rate": round(
                    sum(audits) / len(audits), 4
                ) if audits else 0.0,
            },
            "recent_events": recent[-80:],
            "provider_counts": prov,
            "budget": st.brain.budget.snapshot(),
            "calibration_summary": {
                "accuracy": calibration.get("accuracy"),
                "ece_calibrated": calibration.get("ece_calibrated"),
                "ece_uncalibrated": calibration.get("ece_uncalibrated"),
                "macro_auc": calibration.get("macro_auc"),
            },
            "simulator_only": st.brain.simulator_only,
        }

    return app


# Module-level app for uvicorn
app = create_app()
