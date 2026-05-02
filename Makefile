# ================================================================
#  COMPRESS — Makefile
#  Common commands for development, testing, and deployment.
# ================================================================

.PHONY: help docker local test lint clean logs stop health

COMPOSE := docker compose
PYTHON  := python3
VENV    := .venv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────

docker: ## Build and start via Docker Compose
	./scripts/quickstart.sh --docker

stop: ## Stop Docker services
	$(COMPOSE) down

logs: ## Tail Docker logs
	$(COMPOSE) logs -f compress

rebuild: ## Force rebuild Docker images
	$(COMPOSE) up --build --force-recreate -d

# ── Local Development ─────────────────────────────────────────

venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "\nActivate with: source $(VENV)/bin/activate"

local: ## Start server locally
	./scripts/quickstart.sh --local

# ── Testing ───────────────────────────────────────────────────

test: ## Run full test suite
	./scripts/quickstart.sh --test

test-verbose: ## Run tests with full output
	$(PYTHON) -m pytest tests/ -v --tb=long

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=compress --cov-report=term-missing

# ── Quality ───────────────────────────────────────────────────

lint: ## Run syntax check on all Python files
	$(PYTHON) -c "import py_compile, os; \
		[py_compile.compile(os.path.join(r,f), doraise=True) \
		 for r,_,fs in os.walk('compress') for f in fs if f.endswith('.py')]"
	@echo "All files pass syntax check."

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy compress/ --ignore-missing-imports

# ── Utilities ─────────────────────────────────────────────────

health: ## Check API health endpoint
	@curl -sf http://localhost:8000/compress/health | python3 -m json.tool || \
		echo "Server not reachable at localhost:8000"

supported: ## List supported languages and tokenizers
	@curl -sf http://localhost:8000/compress/supported | python3 -m json.tool || \
		echo "Server not reachable at localhost:8000"

clean: ## Remove build artifacts, caches, and venv
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	$(COMPOSE) down -v --remove-orphans 2>/dev/null || true
	@echo "Cleaned."
