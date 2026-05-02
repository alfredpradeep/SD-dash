#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LENS — Production Setup Script
# Creates venv, installs deps, validates, and launches the server.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # Standard install + launch
#   ./setup.sh --full       # Includes torch + sentence-transformers (~2GB)
#   ./setup.sh --install    # Install only, don't launch
#   ./setup.sh --run        # Skip install, just launch (venv must exist)
#   ./setup.sh --test       # Run test suite
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON=""
PORT="${LENS_API_PORT:-8003}"
WORKERS="${LENS_WORKERS:-2}"
MODE="install_and_run"
FULL_INSTALL=false

# ── Parse args ──
for arg in "$@"; do
    case "$arg" in
        --full)     FULL_INSTALL=true ;;
        --install)  MODE="install" ;;
        --run)      MODE="run" ;;
        --test)     MODE="test" ;;
        --help|-h)
            echo "Usage: ./setup.sh [--full] [--install|--run|--test]"
            echo "  --full      Include torch + sentence-transformers (~2GB)"
            echo "  --install   Install dependencies only"
            echo "  --run       Launch server (venv must exist)"
            echo "  --test      Run test suite"
            exit 0
            ;;
    esac
done

# ── Colour helpers ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[LENS]${NC} $1"; }
warn()  { echo -e "${YELLOW}[LENS]${NC} $1"; }
err()   { echo -e "${RED}[LENS]${NC} $1"; }
header(){ echo -e "\n${CYAN}═══ $1 ═══${NC}"; }

# ── Find Python ──
find_python() {
    for cmd in python3.14 python3.13 python3.12 python3.11 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')"
            local major minor
            major="${ver%%.*}"
            minor="${ver#*.}"
            if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 10 ]]; then
                PYTHON="$cmd"
                info "Found $cmd ($ver)"
                return 0
            fi
        fi
    done
    err "Python 3.10+ required but not found."
    exit 1
}

# ── Create venv ──
create_venv() {
    if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_DIR/bin/python3" ]]; then
        info "Virtual environment exists at ${VENV_DIR}"
    else
        header "Creating virtual environment"
        "$PYTHON" -m venv "$VENV_DIR"
        info "Created venv at ${VENV_DIR}"
    fi
    # Upgrade pip  (use python3 — macOS /usr/bin/python stub triggers xcode-select)
    "$VENV_DIR/bin/python3" -m pip install --upgrade pip --quiet 2>/dev/null || true
}

# ── Install deps ──
install_deps() {
    header "Installing dependencies"

    if [[ "$FULL_INSTALL" == true ]]; then
        info "Full install — including torch + sentence-transformers (this may take a few minutes)"
        "$VENV_DIR/bin/python3" -m pip install -r "${SCRIPT_DIR}/requirements-full.txt" 2>&1 | \
            grep -E "^(Successfully|Requirement|ERROR|WARNING)" || true
    else
        "$VENV_DIR/bin/python3" -m pip install -r "${SCRIPT_DIR}/requirements.txt" 2>&1 | \
            grep -E "^(Successfully|Requirement|ERROR|WARNING)" || true
    fi

    # fasttext-wheel may fail on Python 3.13+ — install gracefully
    "$VENV_DIR/bin/python3" -m pip install fasttext-wheel 2>/dev/null && \
        info "fasttext-wheel installed" || \
        warn "fasttext-wheel unavailable — language detection will use Unicode script heuristic"

    info "Dependencies installed"
}

# ── Create .env if missing ──
setup_env() {
    if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
        cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
        info "Created .env from .env.example — edit to customise"
    else
        info ".env already exists"
    fi
}

# ── Validate install ──
validate() {
    header "Validating installation"

    "$VENV_DIR/bin/python3" -c "
import sys
print(f'  Python:    {sys.version.split()[0]}')

modules = {
    'fastapi':           'Core API',
    'uvicorn':           'ASGI Server',
    'tiktoken':          'Tokenizer',
    'numpy':             'Numerics',
    'scipy':             'Statistics',
    'sklearn':           'ML (anomaly detection)',
    'loguru':            'Logging',
    'pydantic':          'Validation',
    'pydantic_settings':  'Configuration',
}

optional = {
    'torch':                'LM Surprisal (Gap 1)',
    'transformers':         'HuggingFace Transformers',
    'sentence_transformers':'Semantic Entropy (Gap 2)',
    'fasttext':             'Language Detection',
    'clickhouse_connect':   'ClickHouse Storage',
    'redis':                'Redis Cache',
}

ok = True
for mod, label in modules.items():
    try:
        __import__(mod)
        print(f'  ✓ {label:<28s} ({mod})')
    except ImportError:
        print(f'  ✗ {label:<28s} ({mod}) — MISSING')
        ok = False

print()
for mod, label in optional.items():
    try:
        __import__(mod)
        print(f'  ✓ {label:<28s} ({mod})')
    except ImportError:
        print(f'  ○ {label:<28s} ({mod}) — optional, not installed')

if not ok:
    print('\n  ✗ Core dependencies missing — run ./setup.sh again')
    sys.exit(1)
print('\n  All core dependencies OK.')
"

    info "Validation complete"
}

# ── Run tests ──
run_tests() {
    header "Running test suite"
    cd "$SCRIPT_DIR"
    PYTHONPATH="$SCRIPT_DIR/.." "$VENV_DIR/bin/python3" -m pytest tests/ \
        -v --tb=short -q --ignore=../.venv 2>&1 || true
}

# ── Launch server ──
launch() {
    header "Starting LENS server"

    # Load .env
    if [[ -f "${SCRIPT_DIR}/.env" ]]; then
        set -a
        source "${SCRIPT_DIR}/.env"
        set +a
    fi

    info "Server: http://localhost:${PORT}"
    info "Docs:   http://localhost:${PORT}/docs"
    info "Health: http://localhost:${PORT}/lens/health"
    echo ""

    cd "${SCRIPT_DIR}/.."
    exec "$VENV_DIR/bin/python3" -m uvicorn lens.main:app \
        --host "${LENS_API_HOST:-0.0.0.0}" \
        --port "${PORT}" \
        --workers "${WORKERS}" \
        --log-level info
}

# ── Main ──
main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  LENS — Linguistic Entropy Attribution Engine v1.0.0    ║${NC}"
    echo -e "${CYAN}║  Production Setup                                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    case "$MODE" in
        install_and_run)
            find_python
            create_venv
            install_deps
            setup_env
            validate
            launch
            ;;
        install)
            find_python
            create_venv
            install_deps
            setup_env
            validate
            info "Install complete. Run ./setup.sh --run to launch."
            ;;
        run)
            if [[ ! -d "$VENV_DIR" ]] || [[ ! -f "$VENV_DIR/bin/python3" ]]; then
                err "No venv found. Run ./setup.sh --install first."
                exit 1
            fi
            launch
            ;;
        test)
            if [[ ! -d "$VENV_DIR" ]]; then
                find_python
                create_venv
                install_deps
            fi
            validate
            run_tests
            ;;
    esac
}

main
