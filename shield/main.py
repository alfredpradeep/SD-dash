"""
SHIELD Main Application.

FastAPI application entry point with lifespan management.
"""

import os
import json
import httpx
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware import Middleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from loguru import logger

from shield.engine import ShieldEngine, EngineConfig
from shield.llm.provider import ProviderConfig, ProviderType
from shield.api.router import create_router


async def _diagnose_llm_providers():
    """Run quick diagnostics on configured LLM providers at startup."""
    logger.info("=" * 60)
    logger.info("LLM PROVIDER DIAGNOSTICS")
    logger.info("=" * 60)

    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
        # Test Groq
        if groq_key:
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    logger.info("  ✓ Groq API key is VALID")
                elif resp.status_code == 401:
                    logger.error("  ✗ Groq API key is INVALID (401 Unauthorized)")
                elif resp.status_code == 429:
                    logger.warning("  ~ Groq API key is valid but RATE LIMITED (429)")
                else:
                    logger.warning(f"  ? Groq returned HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                logger.warning(f"  ? Groq unreachable: {e}")
        else:
            logger.info("  - Groq: no key configured")

        # Test Gemini — try multiple models
        if gemini_key:
            gemini_ok = False
            for model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
                    resp = await client.post(
                        url,
                        json={"contents": [{"parts": [{"text": "Say OK"}]}], "generationConfig": {"maxOutputTokens": 5}},
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        logger.info(f"  ✓ Gemini API key is VALID (model: {model})")
                        gemini_ok = True
                        break
                    elif resp.status_code == 404:
                        logger.debug(f"  Gemini model '{model}' not found, trying next...")
                        continue
                    elif resp.status_code in (401, 403):
                        logger.error(f"  ✗ Gemini API key is INVALID ({resp.status_code})")
                        break
                    else:
                        logger.warning(f"  ? Gemini {model} returned HTTP {resp.status_code}: {resp.text[:100]}")
                except Exception as e:
                    logger.warning(f"  ? Gemini unreachable: {e}")
                    break

            if not gemini_ok:
                # Try listing models to see what's available
                try:
                    resp = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}")
                    if resp.status_code == 200:
                        models = resp.json().get("models", [])
                        gen_models = [m["name"] for m in models if "generateContent" in str(m.get("supportedGenerationMethods", []))]
                        logger.info(f"  Available Gemini models: {gen_models[:5]}")
                    elif resp.status_code in (401, 403):
                        logger.error(f"  ✗ Gemini API key is INVALID (cannot list models, HTTP {resp.status_code})")
                    else:
                        logger.warning(f"  ? Gemini list models returned HTTP {resp.status_code}")
                except Exception as e:
                    logger.warning(f"  ? Gemini unreachable: {e}")
        else:
            logger.info("  - Gemini: no key configured")

        # Test OpenAI
        if openai_key:
            try:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_key}"},
                )
                if resp.status_code == 200:
                    logger.info("  ✓ OpenAI API key is VALID")
                elif resp.status_code == 401:
                    logger.error("  ✗ OpenAI API key is INVALID (401)")
                else:
                    logger.warning(f"  ? OpenAI returned HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"  ? OpenAI unreachable: {e}")
        else:
            logger.info("  - OpenAI: no key configured")

    logger.info("=" * 60)
from shield.api.middleware import TimingMiddleware


# ==============================================================================
# CONFIGURATION
# ==============================================================================

def get_engine_config() -> EngineConfig:
    """Load engine configuration from environment or defaults."""
    dev_mode = os.getenv("SHIELD_DEV_MODE", "false").lower() == "true"

    provider_configs = [
        ProviderConfig(
            provider_type=ProviderType.GROQ,
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            enabled=bool(os.getenv("GROQ_API_KEY")),
            rate_limit_rpm=15,
        ),
        ProviderConfig(
            provider_type=ProviderType.GEMINI,
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.0-flash",
            enabled=bool(os.getenv("GEMINI_API_KEY")),
            rate_limit_rpm=10,
        ),
        ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model="gpt-4",
            enabled=bool(os.getenv("OPENAI_API_KEY")),
            rate_limit_rpm=20,
        ),
        ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-opus",
            enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
            rate_limit_rpm=10,
        ),
        ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            api_key=os.getenv("HF_API_KEY"),
            model=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1"),
            enabled=bool(os.getenv("HF_API_KEY")),
            rate_limit_rpm=10,
        ),
        ProviderConfig(
            provider_type=ProviderType.LOCAL,
            model=os.getenv("LOCAL_MODEL", "mistral"),
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434"),
            enabled=os.getenv("LOCAL_MODEL") is not None,
            rate_limit_rpm=100,
        ),
    ]

    return EngineConfig(
        llm_providers=provider_configs,
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        max_concurrent_scans=int(os.getenv("MAX_CONCURRENT_SCANS", "5")),
        dev_mode=dev_mode,
    )


# ==============================================================================
# LIFESPAN
# ==============================================================================

engine_instance: Optional[ShieldEngine] = None


def get_engine() -> ShieldEngine:
    """Dependency: return engine instance."""
    if engine_instance is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup/shutdown.

    Initializes ShieldEngine on startup and closes resources on shutdown.
    """
    global engine_instance

    # Startup
    logger.info("SHIELD application starting...")
    try:
        config = get_engine_config()

        # Quick LLM provider diagnostics at startup
        await _diagnose_llm_providers()

        engine_instance = ShieldEngine(config)

        # Now mount the router (engine is ready)
        router = create_router(engine_instance)
        app.include_router(router)
        logger.info("ShieldEngine initialized and routes mounted")
    except Exception as e:
        logger.error(f"Failed to initialize ShieldEngine: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("SHIELD application shutting down...")
    try:
        if engine_instance:
            await engine_instance.close()
        logger.info("ShieldEngine closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# ==============================================================================
# APPLICATION FACTORY
# ==============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="SHIELD",
        description="Safety-aware Harm Identification and Evaluation for LLM Defense",
        version="1.0.0",
        lifespan=lifespan,
        middleware=[
            Middleware(TimingMiddleware),
        ],
    )

    # ===========================================================================
    # STATIC FILE SERVING
    # ===========================================================================

    # Mount static/css and static/js from ui/static/
    static_dir = os.path.join(os.path.dirname(__file__), "ui", "static")
    if os.path.exists(static_dir):
        try:
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
            logger.info(f"Mounted static files from {static_dir}")
        except Exception as e:
            logger.warning(f"Failed to mount static files: {e}")

    # ===========================================================================
    # ROOT ROUTE — Serve UI
    # ===========================================================================

    @app.get(
        "/",
        response_class=HTMLResponse,
        tags=["UI"],
        summary="Web UI",
        description="SHIELD web interface"
    )
    async def root():
        """Serve web UI."""
        ui_file = os.path.join(
            os.path.dirname(__file__),
            "ui", "templates", "index.html"
        )
        if os.path.exists(ui_file):
            try:
                return FileResponse(ui_file)
            except Exception:
                pass

        # Fallback HTML
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHIELD - Cross-Lingual Safety Gap Scanner</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 40px; background: #F5F2EC; }
                .container { max-width: 700px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; border: 1px solid #D4D4D4; }
                h1 { font-family: Georgia, serif; color: #1A1A1A; }
                .endpoint { color: #3B3F2E; font-family: monospace; background: #F5F2EC; padding: 2px 8px; border-radius: 4px; }
                a { color: #3A7D44; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SHIELD</h1>
                <p>Cross-Lingual Safety Gap Scanner &amp; Guardrail Auditor</p>
                <h2>API Endpoints</h2>
                <ul>
                    <li><a href="/docs">Interactive API Documentation (Swagger)</a></li>
                    <li><a href="/shield/health">Health Check</a></li>
                    <li><code class="endpoint">POST /shield/scan</code> &mdash; Full safety scan</li>
                    <li><code class="endpoint">POST /shield/scan/quick</code> &mdash; Quick scan</li>
                    <li><code class="endpoint">POST /shield/probe</code> &mdash; Single probe test</li>
                    <li><code class="endpoint">GET /shield/report/{scan_id}</code> &mdash; Retrieve report</li>
                </ul>
            </div>
        </body>
        </html>
        """)

    # ===========================================================================
    # FAVICON
    # ===========================================================================

    @app.get("/favicon.ico", tags=["UI"])
    async def favicon():
        """Serve favicon."""
        return HTMLResponse(status_code=204)

    # ===========================================================================
    # ERROR HANDLERS
    # ===========================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle unexpected exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # ===========================================================================
    # METRICS ENDPOINT
    # ===========================================================================

    @app.get(
        "/metrics",
        tags=["Monitoring"],
        summary="Prometheus metrics",
        description="Prometheus-compatible metrics endpoint"
    )
    async def metrics():
        """Serve Prometheus metrics."""
        try:
            from prometheus_client import generate_latest
            return generate_latest()
        except ImportError:
            return JSONResponse(
                status_code=503,
                content={"error": "Prometheus not available"},
            )

    return app


# ==============================================================================
# MAIN
# ==============================================================================

# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Parse config
    host = os.getenv("SHIELD_HOST", "0.0.0.0")
    port = int(os.getenv("SHIELD_PORT", "8000"))
    reload = os.getenv("SHIELD_RELOAD", "false").lower() == "true"
    workers = int(os.getenv("SHIELD_WORKERS", "1"))

    logger.info(f"Starting SHIELD on {host}:{port}")

    if workers > 1:
        uvicorn.run(
            "shield.main:app",
            host=host,
            port=port,
            workers=workers,
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
        )
