# LENS — Linguistic Entropy Attribution Engine
# ─────────────────────────────────────────────
# make setup      Install deps + launch server
# make setup-full Install with torch/sentence-transformers + launch
# make install    Install only
# make run        Launch server (venv must exist)
# make test       Run test suite
# make demo       Run demo requests against a running server
# make docker     Build + run with Docker Compose (needs Docker)
# make clean      Remove venv

.PHONY: setup setup-full install run test demo docker clean help

SHELL := /bin/bash
PORT  ?= 8003

help:
	@echo "LENS — Linguistic Entropy Attribution Engine"
	@echo ""
	@echo "  make setup       One-command install + launch"
	@echo "  make setup-full  Install with ML models (~2GB) + launch"
	@echo "  make install     Install dependencies only"
	@echo "  make run         Launch server"
	@echo "  make test        Run test suite"
	@echo "  make demo        Hit all endpoints on running server"
	@echo "  make docker      Docker Compose build + up"
	@echo "  make clean       Remove virtual environment"

setup:
	@chmod +x setup.sh && ./setup.sh

setup-full:
	@chmod +x setup.sh && ./setup.sh --full

install:
	@chmod +x setup.sh && ./setup.sh --install

run:
	@chmod +x setup.sh && ./setup.sh --run

test:
	@chmod +x setup.sh && ./setup.sh --test

demo:
	@echo "── Health ──"
	@curl -s http://localhost:$(PORT)/lens/health | python3 -m json.tool
	@echo ""
	@echo "── English Profile (gpt-4o) ──"
	@curl -s -X POST http://localhost:$(PORT)/lens/profile \
		-H "Content-Type: application/json" \
		-d '{"text":"How do I reset my password?","model_name":"gpt-4o","language":"en","include_token_records":true}' | python3 -m json.tool
	@echo ""
	@echo "── Tamil Profile (claude-opus-4-6) ──"
	@curl -s -X POST http://localhost:$(PORT)/lens/profile \
		-H "Content-Type: application/json" \
		-d '{"text":"என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?","model_name":"claude-opus-4-6","language":"ta","include_token_records":true}' | python3 -m json.tool
	@echo ""
	@echo "── Batch: 5 languages ──"
	@curl -s -X POST http://localhost:$(PORT)/lens/profile/batch \
		-H "Content-Type: application/json" \
		-d '{"items":[{"text":"How do I reset my password?","model":"gpt-4o","language":"en"},{"text":"என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?","model":"gpt-4o","language":"ta"},{"text":"पासवर्ड कैसे रीसेट करें?","model":"claude-opus-4-6","language":"hi"},{"text":"كيف أعيد تعيين كلمة المرور؟","model":"claude-sonnet-4-6","language":"ar"},{"text":"パスワードをリセットするにはどうすればよいですか？","model":"gpt-4o","language":"ja"}]}' | python3 -m json.tool
	@echo ""
	@echo "── Arbitrage: Tamil across all models ──"
	@curl -s -X POST http://localhost:$(PORT)/lens/arbitrage \
		-H "Content-Type: application/json" \
		-d '{"text":"என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?","language":"ta"}' | python3 -m json.tool
	@echo ""
	@echo "── Spike Predictions ──"
	@curl -s "http://localhost:$(PORT)/lens/predict/spikes" | python3 -m json.tool
	@echo ""
	@echo "── Prediction Accuracy (Gap 6) ──"
	@curl -s "http://localhost:$(PORT)/lens/accuracy" | python3 -m json.tool
	@echo ""
	@echo "── Swagger docs: http://localhost:$(PORT)/docs ──"

docker:
	@docker compose up --build -d
	@echo "LENS running at http://localhost:$(PORT)"
	@echo "Swagger docs:   http://localhost:$(PORT)/docs"

clean:
	@rm -rf .venv __pycache__ .pytest_cache
	@find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
