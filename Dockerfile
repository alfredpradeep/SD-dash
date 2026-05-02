# ── Stage 1: Build ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY compress/ /app/compress/
COPY requirements.txt /app/

# Create non-root user
RUN groupadd -r compress && useradd -r -g compress -d /app compress
RUN chown -R compress:compress /app

# Model cache directory (mount a volume here)
RUN mkdir -p /app/models && chown compress:compress /app/models
ENV COMPRESS_MODEL_PATH=/app/models
ENV COMPRESS_AMR_MODEL_PATH=/app/models/amr

# Switch to non-root
USER compress

# Expose ports: API + Prometheus metrics
EXPOSE 8000 9101

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/compress/health')" || exit 1

# Entry point
CMD ["python", "-m", "compress.main"]
