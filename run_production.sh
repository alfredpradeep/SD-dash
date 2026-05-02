#!/usr/bin/env bash
# ================================================================
#  COMPRESS — Production Setup & Run
#  One-shot script: installs deps, downloads ML models, starts server.
#
#  Tested on: macOS (Apple Silicon M1/M2/M3), Linux x86_64
#  Requirements: Python 3.10+, ~4GB disk, ~4GB RAM
#
#  Usage:
#    chmod +x run_production.sh
#    ./run_production.sh
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
MODEL_DIR="$PROJECT_DIR/.models"

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}  ✓${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
step()    { echo -e "\n${BOLD}── $* ──${NC}"; }

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  COMPRESS — Semantic Compression Lattice Engine${NC}"
echo -e "${BOLD}  Production Setup${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Step 1: Python ────────────────────────────────────────────
step "Step 1/6: Checking Python"

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
step "Step 2/6: Setting up virtual environment"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q 2>/dev/null
success "Virtual environment active: $(which python)"

# ── Step 3: Install Dependencies ──────────────────────────────
step "Step 3/6: Installing dependencies"

info "Installing core framework..."
pip install -q \
    "fastapi>=0.110,<1.0" \
    "uvicorn[standard]>=0.29,<1.0" \
    "pydantic>=2.6,<3.0" \
    "pydantic-settings>=2.2,<3.0" \
    "loguru>=0.7,<1.0" \
    "prometheus-client>=0.20,<1.0" \
    "redis[hiredis]>=5.0,<6.0" \
    "httpx>=0.27,<1.0"
success "Core framework installed"

info "Installing ML stack (this may take a few minutes on first run)..."
pip install -q \
    "torch>=2.2,<3.0" \
    "transformers>=4.38,<5.0" \
    "sentence-transformers>=2.6,<4.0" \
    "sentencepiece>=0.2,<1.0" \
    "tiktoken>=0.7,<1.0" \
    "spacy>=3.7,<4.0" \
    "penman>=1.2,<2.0" \
    "numpy>=1.26,<3.0" \
    "Levenshtein>=0.25,<1.0"
success "ML stack installed"

info "Installing LLM provider SDKs..."
pip install -q \
    "google-genai>=1.0,<2.0" \
    "openai>=1.12,<2.0" \
    "anthropic>=0.25,<1.0"
success "LLM provider SDKs installed"

info "Installing test dependencies..."
pip install -q pytest pytest-asyncio httpx
success "Test dependencies installed"

# ── Step 4: Download ML Models ────────────────────────────────
step "Step 4/6: Downloading ML models (first run: ~3GB, cached afterward)"

mkdir -p "$MODEL_DIR"

python3 - <<'PYEOF'
import os, sys

MODEL_DIR = os.environ.get("MODEL_DIR", ".models")
os.environ["HF_HOME"] = MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODEL_DIR, "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(MODEL_DIR, "sentence_transformers")

# 1. LaBSE — cross-lingual semantic similarity (THE core model)
print("  Downloading LaBSE (cross-lingual embeddings, ~1.8GB)...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")
    # Quick validation
    emb = model.encode(["hello", "hola", "こんにちは"])
    assert emb.shape == (3, 768), f"Unexpected shape: {emb.shape}"
    print("  ✓ LaBSE loaded and validated (768-dim embeddings)")
except Exception as e:
    print(f"  ✗ LaBSE failed: {e}", file=sys.stderr)
    sys.exit(1)

# 2. spaCy multilingual NER
print("  Downloading spaCy multilingual NER model...")
try:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "xx_ent_wiki_sm", "-q"])
    import spacy
    nlp = spacy.load("xx_ent_wiki_sm")
    print("  ✓ spaCy xx_ent_wiki_sm loaded")
except Exception as e:
    print(f"  ✗ spaCy model failed: {e}", file=sys.stderr)
    print("    (NER will use regex fallback)")

# 3. mDeBERTa — zero-shot NLI (for sentiment/temporal extraction)
print("  Downloading mDeBERTa-v3-base-xnli (zero-shot NLI, ~1.1GB)...")
try:
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    result = classifier("This is great", candidate_labels=["positive", "negative"])
    print(f"  ✓ mDeBERTa-xnli loaded (test: {result['labels'][0]} = {result['scores'][0]:.2f})")
except Exception as e:
    print(f"  ✗ mDeBERTa failed: {e}", file=sys.stderr)
    print("    (NLI stream will use fallback heuristics)")

print("\n  All models ready.")
PYEOF

MODEL_STATUS=$?
if [ $MODEL_STATUS -ne 0 ]; then
    warn "Some models failed to download. Server will start with available models."
fi
success "Model download complete"

# ── Step 5: Redis Check ───────────────────────────────────────
step "Step 5/6: Checking Redis"

if command -v redis-cli &>/dev/null && redis-cli ping 2>/dev/null | grep -q PONG; then
    success "Redis running — caching enabled"
else
    warn "Redis not running — caching disabled (graceful degradation)"
    info "To enable: brew install redis && redis-server --daemonize yes"
fi

# ── Step 6: Launch ────────────────────────────────────────────
step "Step 6/6: Starting COMPRESS production server"

export COMPRESS_MODEL_PATH="$MODEL_DIR"
export COMPRESS_DEVICE="cpu"
export COMPRESS_API_HOST="0.0.0.0"
export COMPRESS_API_PORT="8000"
export COMPRESS_METRICS_PORT="9101"
export COMPRESS_DEV_MODE="0"
export HF_HOME="$MODEL_DIR"
export TRANSFORMERS_CACHE="$MODEL_DIR/transformers"
export SENTENCE_TRANSFORMERS_HOME="$MODEL_DIR/sentence_transformers"

# ── Detect LLM Provider ──────────────────────────────────────
LLM_STATUS="${RED}No LLM configured (rule-based fallback only)${NC}"
if [ -n "${GROQ_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}Groq (FREE)${NC} — ${COMPRESS_LLM_MODEL:-llama-3.1-8b-instant}"
elif [ -n "${GEMINI_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}Google Gemini${NC} — ${COMPRESS_LLM_MODEL:-gemini-2.0-flash}"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}OpenAI${NC} — ${COMPRESS_LLM_MODEL:-gpt-4o-mini}"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    LLM_STATUS="${GREEN}Anthropic${NC} — ${COMPRESS_LLM_MODEL:-claude-3-haiku}"
elif [ -n "${HF_API_TOKEN:-}" ]; then
    LLM_STATUS="${GREEN}HuggingFace (FREE)${NC} — ${COMPRESS_LLM_MODEL:-Mistral-7B}"
elif [ -n "${COMPRESS_LOCAL_LLM_URL:-}" ]; then
    LLM_STATUS="${GREEN}Local (Ollama)${NC} — ${COMPRESS_LLM_MODEL:-llama3}"
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}COMPRESS Production Server Starting${NC}"
echo ""
echo -e "  Dashboard:    ${CYAN}http://localhost:8000${NC}"
echo -e "  API:          ${CYAN}http://localhost:8000/compress/${NC}"
echo -e "  Swagger Docs: ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  Health:       ${CYAN}http://localhost:8000/compress/health${NC}"
echo -e "  Metrics:      ${CYAN}http://localhost:9101/metrics${NC}"
echo ""
echo -e "  LLM Provider: $LLM_STATUS"
echo ""
if [ -z "${GROQ_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ] && \
   [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] && \
   [ -z "${HF_API_TOKEN:-}" ] && [ -z "${COMPRESS_LOCAL_LLM_URL:-}" ]; then
    echo -e "  ${YELLOW}⚠  No LLM key set! FREE options:${NC}"
    echo -e "     ${BOLD}Option 1 — Groq (recommended, fastest):${NC}"
    echo -e "       1. Go to ${CYAN}https://console.groq.com${NC} → Sign up → Create API key"
    echo -e "       2. export GROQ_API_KEY=\"gsk_your_key_here\""
    echo ""
    echo -e "     ${BOLD}Option 2 — Ollama (local, no internet needed):${NC}"
    echo -e "       1. Install: ${CYAN}https://ollama.com${NC}"
    echo -e "       2. ollama pull llama3.1:8b"
    echo -e "       3. export COMPRESS_LOCAL_LLM_URL=http://localhost:11434"
    echo ""
    echo -e "     ${BOLD}Option 3 — HuggingFace (free tier):${NC}"
    echo -e "       1. Go to ${CYAN}https://huggingface.co/settings/tokens${NC}"
    echo -e "       2. export HF_API_TOKEN=\"hf_your_token_here\""
    echo ""
fi
echo -e "  ${DIM}Press Ctrl+C to stop${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python -m compress.main
