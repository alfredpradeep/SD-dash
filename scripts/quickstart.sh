#!/usr/bin/env bash
# ================================================================
#  COMPRESS — Quick Start Script
#  Sets up the local development environment and starts the server.
#
#  Usage:
#    chmod +x scripts/quickstart.sh
#    ./scripts/quickstart.sh [--docker | --local | --test]
#
#  Modes:
#    --docker   Build and run via Docker Compose (default)
#    --local    Install deps, check Redis, start server locally
#    --test     Install deps and run the test suite
# ================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Helpers ───────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

header() {
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  COMPRESS — Semantic Compression Lattice Engine${NC}"
    echo -e "${BOLD}  v2.1.4${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        fail "$1 is required but not installed."
    fi
    success "$1 found: $(command -v "$1")"
}

# ── Docker Mode ───────────────────────────────────────────────
run_docker() {
    header
    info "Mode: Docker Compose"
    echo ""

    info "Checking prerequisites..."
    check_command docker

    # Check if docker compose (v2) or docker-compose (v1) is available
    if docker compose version &> /dev/null; then
        COMPOSE="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE="docker-compose"
    else
        fail "docker compose is required but not found."
    fi
    success "Docker Compose available ($COMPOSE)"
    echo ""

    info "Building and starting services..."
    $COMPOSE up --build -d

    echo ""
    info "Waiting for services to be healthy..."
    sleep 3

    # Check health
    MAX_RETRIES=20
    RETRY=0
    until curl -sf http://localhost:8000/compress/health > /dev/null 2>&1; do
        RETRY=$((RETRY + 1))
        if [ $RETRY -ge $MAX_RETRIES ]; then
            warn "Health check not responding yet. Services may still be loading models."
            warn "Check logs with: $COMPOSE logs -f compress"
            break
        fi
        sleep 2
    done

    if curl -sf http://localhost:8000/compress/health > /dev/null 2>&1; then
        success "Health check passed"
    fi

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}COMPRESS is running${NC}"
    echo ""
    echo -e "  Dashboard:  ${CYAN}http://localhost:8000${NC}"
    echo -e "  API:        ${CYAN}http://localhost:8000/compress/${NC}"
    echo -e "  API Docs:   ${CYAN}http://localhost:8000/docs${NC}"
    echo -e "  Health:     ${CYAN}http://localhost:8000/compress/health${NC}"
    echo -e "  Metrics:    ${CYAN}http://localhost:9101/metrics${NC}"
    echo ""
    echo -e "  Logs:       ${YELLOW}$COMPOSE logs -f compress${NC}"
    echo -e "  Stop:       ${YELLOW}$COMPOSE down${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ── Local Mode ────────────────────────────────────────────────
run_local() {
    header
    info "Mode: Local Development"
    echo ""

    # ── Python ────────────────────────────────────────────────
    info "Checking prerequisites..."
    check_command python3

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
        fail "Python 3.11+ required, found $PYTHON_VERSION"
    fi
    success "Python $PYTHON_VERSION"

    # ── Virtual Environment ───────────────────────────────────
    echo ""
    VENV_DIR="$PROJECT_DIR/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at .venv..."
        python3 -m venv "$VENV_DIR"
        success "Virtual environment created"
    else
        success "Virtual environment exists at .venv"
    fi

    # Activate
    source "$VENV_DIR/bin/activate"
    success "Activated: $(which python)"

    # ── Dependencies ──────────────────────────────────────────
    echo ""
    info "Installing dependencies (this may take a few minutes on first run)..."
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    success "All dependencies installed"

    # ── spaCy Model ───────────────────────────────────────────
    echo ""
    info "Checking spaCy multilingual model..."
    if python -c "import spacy; spacy.load('xx_ent_wiki_sm')" 2>/dev/null; then
        success "spaCy xx_ent_wiki_sm already installed"
    else
        info "Downloading spaCy multilingual NER model..."
        python -m spacy download xx_ent_wiki_sm -q
        success "spaCy model downloaded"
    fi

    # ── Redis ─────────────────────────────────────────────────
    echo ""
    info "Checking Redis..."
    if command -v redis-cli &> /dev/null && redis-cli ping 2>/dev/null | grep -q PONG; then
        success "Redis is running"
    else
        warn "Redis is not running. Cache will be disabled (graceful degradation)."
        warn "To enable caching, start Redis: redis-server --daemonize yes"
    fi

    # ── ML Models ─────────────────────────────────────────────
    echo ""
    info "Pre-downloading ML models (first run only, ~2-3 GB)..."
    python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading LaBSE...')
SentenceTransformer('sentence-transformers/LaBSE')
print('  LaBSE ready.')
" 2>/dev/null || warn "LaBSE download deferred — will download on first request."

    # ── Launch ────────────────────────────────────────────────
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}Starting COMPRESS server...${NC}"
    echo ""
    echo -e "  Dashboard:  ${CYAN}http://localhost:8000${NC}"
    echo -e "  API:        ${CYAN}http://localhost:8000/compress/${NC}"
    echo -e "  API Docs:   ${CYAN}http://localhost:8000/docs${NC}"
    echo -e "  Health:     ${CYAN}http://localhost:8000/compress/health${NC}"
    echo -e "  Metrics:    ${CYAN}http://localhost:9101/metrics${NC}"
    echo ""
    echo -e "  Stop:       ${YELLOW}Ctrl+C${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    python -m compress.main
}

# ── Test Mode ─────────────────────────────────────────────────
run_tests() {
    header
    info "Mode: Test Suite"
    echo ""

    # Activate venv if it exists
    VENV_DIR="$PROJECT_DIR/.venv"
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
        success "Using virtual environment: $(which python)"
    else
        info "No .venv found. Using system Python."
        check_command python3
    fi

    # Ensure test deps
    info "Installing test dependencies..."
    pip install pytest pytest-asyncio httpx -q 2>/dev/null || true
    pip install -r requirements.txt -q 2>/dev/null || true
    echo ""

    info "Running test suite..."
    echo ""
    python -m pytest tests/ -v --tb=short --no-header -q

    echo ""
    success "Test suite complete."
}

# ── Entrypoint ────────────────────────────────────────────────
MODE="${1:---docker}"

case "$MODE" in
    --docker|-d)
        run_docker
        ;;
    --local|-l)
        run_local
        ;;
    --test|-t)
        run_tests
        ;;
    --help|-h)
        header
        echo "Usage: ./scripts/quickstart.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  --docker, -d   Build and run via Docker Compose (default)"
        echo "  --local,  -l   Install deps locally, check Redis, start server"
        echo "  --test,   -t   Install deps and run the test suite"
        echo "  --help,   -h   Show this help message"
        echo ""
        ;;
    *)
        fail "Unknown mode: $MODE. Use --docker, --local, --test, or --help."
        ;;
esac
