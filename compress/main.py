"""
COMPRESS Application Entry Point.

Serves:
  - FastAPI REST API at /compress/*
  - Haiku Platform UI at /
  - Prometheus metrics at :9101/metrics (separate server)
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from loguru import logger
from prometheus_client import start_http_server

from compress.config import Config
from compress.api.router import router
from compress.api.v4_router import v4_router
from compress.api.middleware import TimingMiddleware

# ── Global engine instance (dependency-injected via router) ──────
engine_instance = None  # SCLEngine or DevSCLEngine

UI_DIR = Path(__file__).parent / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:  load config, init engine, start metrics server.
    Shutdown: close engine resources.
    """
    global engine_instance

    logger.info("COMPRESS starting up...")
    config = Config()

    if config.dev_mode:
        from compress.dev_engine import DevSCLEngine
        engine_instance = DevSCLEngine(config)
        logger.info("DevSCLEngine initialised (dev mode — no ML deps)")
    else:
        from compress.engine import SCLEngine
        engine_instance = SCLEngine(config)
        logger.info("SCLEngine initialised (production)")

    # Start Prometheus metrics server on separate port
    try:
        start_http_server(config.metrics_port)
        logger.info("Prometheus metrics on :{}", config.metrics_port)
    except OSError as e:
        logger.warning("Could not start metrics server: {}", e)

    yield

    # Shutdown
    logger.info("COMPRESS shutting down...")
    if engine_instance and hasattr(engine_instance, "cache") and hasattr(engine_instance.cache, "close"):
        await engine_instance.cache.close()
    engine_instance = None


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="COMPRESS — Semantic Compression Lattice",
        description=(
            "Token-cost optimization engine that decomposes multilingual text "
            "into language-independent semantic graphs and finds minimum-token "
            "surface realizations verified through a triple-gate system."
        ),
        version="2.1.4",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(TimingMiddleware)

    # API routes
    app.include_router(router)
    app.include_router(v4_router)  # COMPRESS v4.0 endpoints at /v4/*

    # Static files (CSS, JS)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Serve UI
    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        """Serve the Haiku Platform dashboard."""
        index_path = TEMPLATES_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>COMPRESS API</h1><p>UI not found. API available at /compress/</p>"
        )

    # Serve COMPRESS v4.0 UI
    @app.get("/v4", response_class=HTMLResponse)
    async def serve_v4_ui():
        """Serve the COMPRESS v4.0 SD Console dashboard."""
        v4_path = TEMPLATES_DIR / "v4.html"
        if v4_path.exists():
            return HTMLResponse(content=v4_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>COMPRESS v4.0</h1><p>UI not found. API available at /v4/</p>"
        )

    @app.get("/favicon.ico")
    async def favicon():
        """Prevent 404 for favicon requests."""
        return HTMLResponse(content="", status_code=204)

    return app


app = create_app()


if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        "compress.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level="info",
        access_log=False,
    )
