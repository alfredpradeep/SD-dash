FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — core layer (cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fasttext-wheel || true

# Application code (changes more often — separate layer)
COPY . .

# Create non-root user
RUN useradd -m lens && chown -R lens:lens /app
USER lens

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8003/lens/health || exit 1

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "lens.main:app", \
     "--host", "0.0.0.0", "--port", "8003", "--workers", "2"]
