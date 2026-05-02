#!/usr/bin/env bash
# ================================================================
#  SHIELD — Cross-Lingual Safety Gap Scanner
#  One-shot script: installs deps, starts server.
#
#  Tested on: macOS (Apple Silicon M1/M2/M3), Linux x86_64
#  Requirements: Python 3.10+, ~2GB disk, ~2GB RAM
#
#  Usage:
#    chmod +x run_shield.sh
#    ./run_shield.sh
#
#  Required (at least one LLM key):
#    export GROQ_API_KEY="gsk_..."       # FREE — recommended
#    export GEMINI_API_KEY="..."          # FREE tier available
#    export OPENAI_API_KEY="sk-..."       # Paid
#    export ANTHROPIC_API_KEY="sk-ant-..."# Paid
#
#  Optional:
#    export SHIELD_DEV_MODE=true          # Run without real LLM calls
#    export SHIELD_PORT=8002              # Custom port (default: 8002)
# ================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
VENV_DIR="$PROJECT_DIR/.venv"

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}  ✓${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
step()    { echo -e "\n${BOLD}── $* ──${NC}"; }

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  SHIELD — Cross-Lingual Safety Gap Scanner${NC}"
echo -e "${BOLD}  Haiku AI Governance Platform${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Step 1: Python ────────────────────────────────────────────
step "Step 1/5: Checking Python"

if ! command -v python3 &>/dev/null; then
    fail "Python 3 is required. Install from https://python.org"
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYMAJOR=$(echo "$PYVER" | cut -d. -f1)
PYMINOR=$(echo "$PYVER" | cut -d. -f2)

if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required, found $PYVER"
fi
success "Python $PYVER"

# ── Step 2: Virtual Environment ───────────────────────────────
step "Step 2/5: Setting up virtual environment"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q 2>/dev/null
success "Virtual environment active: $(which python)"

# ── Step 3: Install Dependencies ──────────────────────────────
step "Step 3/5: Installing dependencies"

info "Installing core framework..."
pip install -q \
    "fastapi>=0.110,<1.0" \
    "uvicorn[standard]>=0.29,<1.0" \
    "pydantic>=2.6,<3.0" \
    "pydantic-settings>=2.2,<3.0" \
    "loguru>=0.7,<1.0" \
    "httpx>=0.27,<1.0" \
    "numpy>=1.26,<3.0" \
    "scipy>=1.12,<2.0"
success "Core framework installed"

info "Installing ML stack..."
pip install -q \
    "sentence-transformers>=2.6,<4.0" \
    "transformers>=4.38,<5.0" \
    "torch>=2.2,<3.0" \
    "tiktoken>=0.7,<1.0" \
    "spacy>=3.7,<4.0" \
    "scikit-learn>=1.4,<2.0"
success "ML stack installed"

info "Installing LLM provider SDKs..."
pip install -q \
    "openai>=1.12,<2.0" \
    "anthropic>=0.25,<1.0"
success "LLM SDKs installed"

# Optional: prometheus + redis
pip install -q "prometheus-client>=0.20,<1.0" "redis>=5.0,<6.0" 2>/dev/null || true

# ── Step 4: Redis Check ───────────────────────────────────────
step "Step 4/5: Checking services"

if command -v redis-cli &>/dev/null && redis-cli ping 2>/dev/null | grep -q PONG; then
    success "Redis running — caching & benchmark storage enabled"
else
    warn "Redis not running — in-memory fallback (all features still work)"
fi

# ── Step 5: Launch ────────────────────────────────────────────
step "Step 5/5: Starting SHIELD server"

export SHIELD_HOST="${SHIELD_HOST:-0.0.0.0}"
export SHIELD_PORT="${SHIELD_PORT:-8002}"

# ── Detect LLM Provider ──────────────────────────────────────
LLM_STATUS="${RED}No LLM configured${NC}"
DEV_NOTE=""
if [ -n "${GROQ_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}Groq (FREE)${NC}"
elif [ -n "${GEMINI_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}Google Gemini${NC}"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}OpenAI${NC}"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}Anthropic${NC}"
elif [ "${SHIELD_DEV_MODE:-}" = "true" ]; then
    LLM_STATUS="${YELLOW}Dev Mode (simulated responses)${NC}"
else
    DEV_NOTE="\n  ${YELLOW}⚠  No LLM key detected! Starting in DEV MODE (simulated).${NC}"
    DEV_NOTE="$DEV_NOTE\n     To use real LLM calls, set one of these:"
    DEV_NOTE="$DEV_NOTE\n       ${BOLD}export GROQ_API_KEY=\"gsk_...\"${NC}   (FREE at https://console.groq.com)"
    DEV_NOTE="$DEV_NOTE\n       ${BOLD}export GEMINI_API_KEY=\"...\"${NC}    (FREE at https://aistudio.google.com)"
    DEV_NOTE="$DEV_NOTE\n       ${BOLD}export OPENAI_API_KEY=\"sk-...\"${NC}"
    DEV_NOTE="$DEV_NOTE\n"
    export SHIELD_DEV_MODE=true
    LLM_STATUS="${YELLOW}Dev Mode (simulated)${NC}"
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}SHIELD Safety Scanner Starting${NC}"
echo ""
echo -e "  Dashboard:    ${CYAN}http://localhost:${SHIELD_PORT}${NC}"
echo -e "  API Docs:     ${CYAN}http://localhost:${SHIELD_PORT}/docs${NC}"
echo -e "  Health:       ${CYAN}http://localhost:${SHIELD_PORT}/shield/health${NC}"
echo ""
echo -e "  LLM Provider: $LLM_STATUS"
if [ -n "${DEV_NOTE:-}" ]; then
    echo -e "$DEV_NOTE"
fi
echo ""
echo -e "  ${BOLD}API Endpoints:${NC}"
echo -e "    POST /shield/scan        Full safety scan"
echo -e "    POST /shield/scan/quick   Quick scan (10 probes/cell)"
echo -e "    POST /shield/probe        Test single probe"
echo -e "    GET  /shield/health       Health check"
echo ""
echo -e "  ${DIM}Press Ctrl+C to stop${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python -m shield.main
