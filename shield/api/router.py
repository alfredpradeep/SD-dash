"""
SHIELD API Router.

FastAPI endpoints for SHIELD operations:
  - Batch audit scanning (full, quick, streaming)
  - Real-time per-conversation monitoring
  - Reporting and benchmarks
  - Health checks
"""
from __future__ import annotations

import uuid
import time
from typing import Optional, TYPE_CHECKING
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from shield.api.schemas import (
    ScanRequest,
    QuickScanRequest,
    SingleProbeRequest,
    ScanResponse,
    ProbeResultResponse,
    HealthResponse,
    HealthResponseItem,
    EngineResult,
    GapMatrixCell,
    MonitorStartRequest,
    MonitorCheckRequest,
    MonitorStopRequest,
    MonitorInteractionResponse,
    SimulatorMode,
    InterviewSimRequest,
    InterviewSimResponse,
    InterviewExchangeResponse,
)

if TYPE_CHECKING:
    from shield.engine import ShieldEngine


def create_router(engine: ShieldEngine) -> APIRouter:
    """Create and return the SHIELD API router."""
    router = APIRouter(prefix="/shield", tags=["shield"])

    async def get_engine() -> ShieldEngine:
        return engine

    # ===========================================================================
    # BATCH AUDIT ENDPOINTS
    # ===========================================================================

    @router.post(
        "/scan",
        response_model=ScanResponse,
        summary="Full SHIELD scan",
        description="Execute comprehensive adversarial robustness scan (batch audit)"
    )
    async def scan(
        request: ScanRequest,
        engine: ShieldEngine = Depends(get_engine),
    ) -> ScanResponse:
        """Execute full SHIELD scan."""
        try:
            scan_id = str(uuid.uuid4())
            logger.info(
                f"Starting scan {scan_id}: {request.architecture} "
                f"on {request.target_endpoint} "
                f"(simulator={request.simulator_mode.value}, industry={request.target_industry})"
            )

            result = await engine.scan(request, scan_id=scan_id)

            # Include probe details and report in JSON response
            response_data = result.dict()
            if hasattr(result, '_probe_details'):
                response_data["probes"] = result._probe_details
            if hasattr(result, '_report'):
                response_data["report"] = result._report

            logger.info(f"Scan {scan_id} completed")
            return JSONResponse(content=response_data)
        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    @router.post(
        "/scan/stream",
        summary="Streaming scan (SSE)",
        description="Execute scan with real-time SSE events for each engine step",
    )
    async def scan_stream(
        request: ScanRequest,
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Execute streaming SHIELD scan with Server-Sent Events."""
        scan_id = str(uuid.uuid4())
        logger.info(f"Starting streaming scan {scan_id} (simulator={request.simulator_mode.value})")

        async def event_generator():
            async for event in engine.scan_stream(request, scan_id=scan_id):
                yield event

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/scan/quick",
        response_model=ScanResponse,
        summary="Quick scan",
        description="Execute reduced scan (fewer probes per cell)"
    )
    async def quick_scan(
        request: QuickScanRequest,
        engine: ShieldEngine = Depends(get_engine),
    ) -> ScanResponse:
        """Execute quick scan."""
        try:
            scan_id = str(uuid.uuid4())
            logger.info(f"Starting quick scan {scan_id}")

            full_request = ScanRequest(
                target_endpoint=request.target_endpoint,
                target_api_key=request.target_api_key,
                target_model=request.target_model,
                architecture=request.architecture,
                languages=request.languages,
                categories=request.categories,
                scan_depth="quick",
                pair_iterations=1,
                smoothing_samples=2,
                target_industry=request.target_industry,
                simulator_mode=request.simulator_mode,
                simulator_model_profile=request.simulator_model_profile,
                detection_scope=request.detection_scope,
            )

            result = await engine.quick_scan(full_request, scan_id=scan_id)
            return result
        except Exception as e:
            logger.error(f"Quick scan failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Quick scan failed: {str(e)}")

    @router.post(
        "/probe",
        response_model=ProbeResultResponse,
        summary="Single probe test",
        description="Test a single probe against an endpoint"
    )
    async def single_probe(
        request: SingleProbeRequest,
        engine: ShieldEngine = Depends(get_engine),
    ) -> ProbeResultResponse:
        """Test single probe."""
        try:
            logger.debug(
                f"Single probe: {request.probe_text[:50]}... "
                f"(simulator={request.simulator_mode.value})"
            )
            result = await engine.single_probe(request)
            return result
        except Exception as e:
            logger.error(f"Probe test failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Probe test failed: {str(e)}")

    # ===========================================================================
    # REAL-TIME MONITORING ENDPOINTS
    # ===========================================================================

    @router.post(
        "/monitor/start",
        summary="Start monitoring session",
        description="Begin real-time per-conversation safety monitoring"
    )
    async def monitor_start(
        request: MonitorStartRequest,
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Start a new monitoring session."""
        try:
            session = engine.session_manager.create_session(
                target_endpoint=request.target_endpoint,
                model_name=request.model_name,
                language=request.language,
                industry=request.industry,
            )
            logger.info(f"Monitoring session started: {session.session_id}")
            return {
                "session_id": session.session_id,
                "status": "active",
                "message": f"Monitoring session started for {request.model_name}",
            }
        except Exception as e:
            logger.error(f"Monitor start failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/monitor/check",
        response_model=MonitorInteractionResponse,
        summary="Check prompt/response pair",
        description="Real-time safety check on a single interaction (<500ms)"
    )
    async def monitor_check(
        request: MonitorCheckRequest,
        engine: ShieldEngine = Depends(get_engine),
    ) -> MonitorInteractionResponse:
        """Check a single prompt/response pair for safety."""
        try:
            session = engine.session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
            if not session.active:
                raise HTTPException(status_code=400, detail=f"Session {request.session_id} is no longer active")

            result = engine.monitor_check_interaction(
                session=session,
                user_prompt=request.user_prompt,
                model_response=request.model_response,
                language=request.language,
            )
            return MonitorInteractionResponse(**result)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Monitor check failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/monitor/session/{session_id}",
        summary="Get monitoring session status",
        description="Get current session status and aggregated metrics"
    )
    async def monitor_session_status(
        session_id: str,
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Get monitoring session status."""
        session = engine.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return session.to_dict()

    @router.post(
        "/monitor/stop",
        summary="Stop monitoring session",
        description="End session and generate a session report"
    )
    async def monitor_stop(
        request: MonitorStopRequest,
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Stop monitoring session and generate report."""
        try:
            session = engine.session_manager.stop_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

            report = engine.session_manager.generate_report(session)
            logger.info(f"Monitoring session stopped: {request.session_id} ({report.total_interactions} interactions)")
            return report.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Monitor stop failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/monitor/sessions",
        summary="List active monitoring sessions",
        description="Get all active monitoring sessions"
    )
    async def list_monitor_sessions(
        engine: ShieldEngine = Depends(get_engine),
    ):
        """List all active monitoring sessions."""
        sessions = [
            s.to_dict()
            for s in engine.session_manager.sessions.values()
            if s.active
        ]
        return {"active_sessions": sessions, "count": len(sessions)}

    # ===========================================================================
    # INDUSTRY PROFILES ENDPOINT
    # ===========================================================================

    @router.get(
        "/industries",
        summary="List available industry profiles",
        description="Get all available industry profiles and their configurations"
    )
    async def list_industries():
        """List available industry profiles."""
        from shield.industries.base import INDUSTRY_REGISTRY
        return {
            "industries": [
                {
                    "id": p.industry_id,
                    "name": p.name,
                    "description": p.description,
                    "harm_categories": p.harm_categories,
                    "bias_dimensions": p.bias_dimensions,
                    "regulations": p.regulations,
                    "probe_categories": p.probe_categories,
                }
                for p in INDUSTRY_REGISTRY.values()
            ]
        }

    # ===========================================================================
    # SIMULATOR PROFILES ENDPOINT
    # ===========================================================================

    @router.get(
        "/simulator/profiles",
        summary="List simulator model profiles",
        description="Get available model profiles for simulator mode"
    )
    async def list_simulator_profiles():
        """List simulator model profiles."""
        from shield.simulator.profiles import MODEL_PROFILES
        return {
            "profiles": [
                {
                    "model_id": p.model_id,
                    "provider": p.provider,
                    "display_name": p.display_name,
                    "bypass_rates": p.bypass_rates,
                    "language_safety": p.language_safety,
                }
                for p in MODEL_PROFILES.values()
            ]
        }

    # ===========================================================================
    # REPORTING ENDPOINTS
    # ===========================================================================

    @router.get(
        "/report/{scan_id}",
        response_model=ScanResponse,
        summary="Get scan report",
    )
    async def get_report(
        scan_id: str,
        engine: ShieldEngine = Depends(get_engine),
    ) -> ScanResponse:
        """Get scan report by ID."""
        try:
            result = await engine.get_report(scan_id)
            if not result:
                raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve report: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/report/{scan_id}/regulatory",
        summary="Regulatory compliance report",
    )
    async def get_regulatory_report(
        scan_id: str,
        region: str = "US",
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Get regulatory compliance report."""
        try:
            report = await engine.get_regulatory_report(scan_id, region)
            if not report:
                raise HTTPException(status_code=404, detail=f"Report not found for scan {scan_id}")
            return report
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Regulatory report failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===========================================================================
    # HEALTH CHECK
    # ===========================================================================

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Component health check",
    )
    async def health_check(
        engine: ShieldEngine = Depends(get_engine),
    ) -> HealthResponse:
        """Health check endpoint."""
        try:
            health = await engine.health_check()
            return health
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/benchmark/{model}",
        summary="Model benchmark data",
    )
    async def get_benchmark(
        model: str,
        language: str = "english",
        category: str = "financial",
        window_days: int = 90,
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Get benchmark data for model."""
        try:
            benchmark = await engine.get_benchmark(model, language, category, window_days)
            return benchmark
        except Exception as e:
            logger.error(f"Benchmark retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/export", summary="Export report")
    async def export_report(
        scan_id: str,
        format: str = "json",
        engine: ShieldEngine = Depends(get_engine),
    ):
        """Export scan report."""
        try:
            report = await engine.get_report(scan_id)
            if not report:
                raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
            if format == "json":
                return report.dict()
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===========================================================================
    # INTERVIEW SIMULATION ENDPOINT
    # ===========================================================================

    @router.post(
        "/interview/simulate",
        response_model=InterviewSimResponse,
        summary="Simulate AI interview with safety monitoring",
        description="Generate a realistic 20-minute management interview with SHIELD safety/bias/compliance monitoring"
    )
    async def simulate_interview(
        request: InterviewSimRequest,
        engine: ShieldEngine = Depends(get_engine),
    ) -> InterviewSimResponse:
        """Simulate an interview with real-time safety analysis."""
        try:
            from shield.simulator.interview import simulate_interview as sim_interview

            logger.info(
                f"Starting interview simulation: {request.language} / {request.industry} / {request.role}"
            )

            result = sim_interview(
                language=request.language,
                industry=request.industry,
                duration_minutes=request.duration_minutes,
                candidate_name=request.candidate_name,
                role=request.role
            )

            # Build response object
            response = InterviewSimResponse(
                metadata=result["metadata"],
                statistics=result["statistics"],
                exchanges=[
                    InterviewExchangeResponse(**exchange)
                    for exchange in result["exchanges"]
                ]
            )

            logger.info(
                f"Interview simulation completed: {len(response.exchanges)} exchanges, "
                f"{response.statistics['unsafe_count']} UNSAFE, {response.statistics['risky_count']} RISKY"
            )
            return response
        except Exception as e:
            logger.error(f"Interview simulation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Interview simulation failed: {str(e)}")

    return router
