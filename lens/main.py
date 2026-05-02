"""
LENS FastAPI application entrypoint.

Starts the Linguistic Entropy Attribution Engine and serves the API
on port 8003 by default.

Usage:
    uvicorn lens.main:app --host 0.0.0.0 --port 8003 --workers 4
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from loguru import logger
import sys

from lens.config import Config
from lens.engine import LEAEEngine
from lens.api.router import router

# Global engine instance — injected into router via get_engine()
engine_instance: LEAEEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global engine_instance
    config = Config()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level=config.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    logger.info("Starting LENS — Linguistic Entropy Attribution Engine v1.0.0")
    engine_instance = LEAEEngine(config)

    # Ensure ClickHouse schema
    try:
        await engine_instance.ch_store.ensure_schema()
    except Exception as e:
        logger.warning("Schema setup failed (non-fatal): {}", e)

    logger.info("LENS ready on port {}", config.api_port)
    yield

    logger.info("LENS shutting down")


app = FastAPI(
    title="LENS — Linguistic Entropy Attribution Engine",
    description=(
        "Production-grade token cost monitoring using information theory. "
        "Measures tokenization efficiency, equity, and predicts cost spikes "
        "before they materialise — across 20+ languages and 10+ AI models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", tags=["Root"], response_class=HTMLResponse)
async def root():
    """Serve the LENS dashboard at root. JSON info at /lens/health."""
    dashboard = Path(__file__).parent / "dashboard.html"
    if dashboard.exists():
        return HTMLResponse(content=dashboard.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>LENS</h1><p>Dashboard not found. Visit <a href='/docs'>/docs</a> for API.</p>")
