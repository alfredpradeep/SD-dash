# SD-dash

SD-dash is a multi-service Python workspace with four main modules:

- `compress/` - COMPRESS, the semantic compression engine
- `lens/` - LENS, the linguistic entropy attribution engine
- `shield/` - SHIELD, the cross-lingual safety gap scanner
- `shield_v2/` - SHIELD v2, the newer production-grade scanner

The repository also includes shared tests, launch scripts, Docker Compose files, and environment templates.

## Repository Layout

- `compress/` - main COMPRESS FastAPI app, UI, lattice/search/verification code, and storage helpers
- `lens/` - LENS API, analytics, entropy, moat, prediction, storage, and tests
- `shield/` - SHIELD API, attack benchmarking, probing, judge, drift, and simulator code
- `shield_v2/` - newer SHIELD implementation with agents, API, audit, guard, core, and tests
- `tests/` - top-level COMPRESS test suite and fixtures
- `scripts/` - helper scripts such as `quickstart.sh`
- `run_production.sh` - one-shot COMPRESS production launcher
- `run_shield.sh` - one-shot SHIELD launcher
- `run_shield_v2.sh` - one-shot SHIELD v2 launcher
- `Dockerfile` and `docker-compose.yml` - root container setup for COMPRESS
- `lens/Dockerfile` and `lens/docker-compose.yml` - LENS container setup
- `README.md` - this guide
- `.env.example` and `lens/.env.example` - safe environment templates

## Prerequisites

- Python 3.10+ for `shield/` and `shield_v2/`
- Python 3.11+ for `compress/`
- Optional but recommended:
  - `redis`
  - `docker` and Docker Compose
  - API keys for LLM-backed paths:
    - `GROQ_API_KEY`
    - `GEMINI_API_KEY`
    - `OPENAI_API_KEY`
    - `ANTHROPIC_API_KEY`

## Environment Files

The repo ships with templates:

- [`.env.example`](.env.example)
- [`lens/.env.example`](lens/.env.example)

Use them as the base for local config:

```bash
cp .env.example .env
cp lens/.env.example lens/.env
```

Do not commit your real `.env` files. They are intentionally ignored.

## COMPRESS

COMPRESS is the root semantic compression service.

### What it does

- Serves the COMPRESS API
- Serves the dashboard UI
- Uses model caches in `.models/` when available
- Can run in dev mode or production mode

### Run locally

From the repository root:

```bash
make venv
make local
```

Or run the app directly:

```bash
python -m compress.main
```

### Run with Docker

From the repository root:

```bash
make docker
```

Or:

```bash
docker compose up --build
```

### Useful commands

- `make test` - run the full COMPRESS test suite
- `make lint` - syntax check Python files
- `make health` - call the health endpoint
- `make supported` - list supported tokenizers and languages
- `make clean` - remove caches and local build artifacts

### Main entry files

- [`compress/main.py`](compress/main.py)
- [`compress/engine.py`](compress/engine.py)
- [`compress/dev_engine.py`](compress/dev_engine.py)

## LENS

LENS is the linguistic entropy attribution engine.

### What it does

- Tracks token efficiency and entropy
- Serves a FastAPI app with dashboard support
- Uses Redis and ClickHouse when available
- Falls back gracefully if storage services are unavailable

### Run locally

From the `lens/` directory:

```bash
cd lens
make setup
```

That installs dependencies and launches the service.

If you only want dependencies:

```bash
cd lens
make install
```

To run without the setup wrapper:

```bash
cd lens
python -m uvicorn lens.main:app --host 0.0.0.0 --port 8003
```

### Run with Docker

From the `lens/` directory:

```bash
cd lens
make docker
```

Or:

```bash
cd lens
docker compose up --build
```

### Useful commands

- `make setup-full` - install the heavier ML stack and launch
- `make run` - start the server after setup
- `make test` - run the LENS test suite
- `make demo` - hit the main API endpoints against a running server
- `make clean` - remove the virtual environment and caches

### Main entry files

- [`lens/main.py`](lens/main.py)
- [`lens/engine.py`](lens/engine.py)
- [`lens/config.py`](lens/config.py)

## SHIELD

SHIELD is the cross-lingual safety gap scanner.

### What it does

- Provides a FastAPI API for safety scanning
- Runs probe-based evaluation and benchmarking
- Supports multiple LLM providers
- Can run in dev mode if no provider key is set

### Run locally

From the repository root:

```bash
chmod +x run_shield.sh
./run_shield.sh
```

Or directly:

```bash
python -m shield.main
```

### Environment variables

Set at least one provider key for real model-backed runs:

- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

Optional runtime settings:

- `SHIELD_DEV_MODE=true`
- `SHIELD_PORT=8002`
- `SHIELD_HOST=0.0.0.0`

### Useful commands

- `./run_shield.sh` - full launcher
- `python -m shield.main` - start the app manually

### Main entry files

- [`shield/main.py`](shield/main.py)
- [`shield/engine.py`](shield/engine.py)
- [`shield/config.py`](shield/config.py)

## SHIELD v2

SHIELD v2 is the newer production-oriented scanner.

### What it does

- Launches a FastAPI application
- Exposes the `shield_v2` API
- Includes agent, audit, guard, and core modules
- Can run tests or serve the app from a single launcher

### Run locally

From the repository root:

```bash
chmod +x run_shield_v2.sh
./run_shield_v2.sh
```

Or:

```bash
python -m shield_v2.run
```

### Useful commands

- `./run_shield_v2.sh` - start the service
- `./run_shield_v2.sh test` - run the SHIELD v2 test suite
- `./run_shield_v2.sh install` - install dependencies only
- `./run_shield_v2.sh --port 9000` - run on a different port

### Main entry files

- [`shield_v2/run.py`](shield_v2/run.py)
- [`shield_v2/api/app.py`](shield_v2/api/app.py)

## Testing

### Root COMPRESS tests

```bash
make test
```

Or:

```bash
python -m pytest tests/ -v
```

### LENS tests

```bash
cd lens
make test
```

### SHIELD tests

```bash
./run_shield.sh
```

### SHIELD v2 tests

```bash
./run_shield_v2.sh test
```

## Notes on Large Local Artifacts

The following are intentionally not committed:

- `.venv/`
- `.models/`
- `.pytest_cache/`
- `__pycache__/`
- `.env`
- large binary exports such as `.pdf`, `.docx`, `.zip`

Those files are local-only runtime or cache artifacts and are safe to regenerate.

## Recommended First Run

1. Copy the example env files.
2. Create and activate the Python environment you need.
3. Run the module you want:
   - COMPRESS: `make local`
   - LENS: `cd lens && make setup`
   - SHIELD: `./run_shield.sh`
   - SHIELD v2: `./run_shield_v2.sh`
4. Open the printed localhost URL in your browser.
