"""SHIELD v2 — production-grade entry point.

Run:
    python -m shield_v2.run
    # or
    uvicorn shield_v2.api.app:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    host = os.environ.get("SHIELD_HOST", "0.0.0.0")
    port = int(os.environ.get("SHIELD_PORT", "8000"))
    reload_flag = os.environ.get("SHIELD_RELOAD", "").lower() in {"1", "true", "yes"}

    # Ensure project root is importable when run as `python run.py` from within shield_v2/
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    uvicorn.run(
        "shield_v2.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload_flag,
        log_level=os.environ.get("SHIELD_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
