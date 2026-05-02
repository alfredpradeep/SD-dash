# SD-dash

This repository contains two Python services:

- `compress/` - the COMPRESS semantic compression engine
- `lens/` - the LENS linguistic entropy attribution engine

## What is tracked

The repo is set up to keep the runnable source code, scripts, configs, tests, and text documentation.
Local-only artifacts are ignored, including:

- `.venv/`
- `.models/`
- `.pytest_cache/`
- `.env`
- binary exports like `*.pdf`, `*.docx`, and `*.zip`

## Run COMPRESS

From the repository root:

```bash
make docker
```

or for local development:

```bash
make venv
make local
```

The root service uses `requirements.txt`, `Dockerfile`, and `docker-compose.yml`.

## Run LENS

From the `lens/` directory:

```bash
make setup
```

or with Docker:

```bash
make docker
```

## Notes

- Copy `.env.example` to `.env` when you need local secrets.
- The ML model cache is intentionally not committed; it is downloaded on first run.
