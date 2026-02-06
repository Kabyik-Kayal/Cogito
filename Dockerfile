# ============================================================================
# Cogito - Self-Correcting RAG System
# Multi-stage Dockerfile optimized for minimal image size
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies & compile C++ extensions
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build-time deps for llama-cpp-python (only what's needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Install deps in venv with cache-less pip
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what's needed to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Minimal runtime libs (curl for healthchecks only)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual-env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user and setup app directory in one layer
RUN groupadd --gid 1000 cogito && \
    useradd --uid 1000 --gid cogito --shell /bin/bash --create-home cogito && \
    mkdir -p /app/data/raw/uploads /app/data/processed /app/db/chroma /app/db/graph \
             /app/models /app/logs /app/results && \
    chown -R cogito:cogito /app

WORKDIR /app

# Copy application source code
COPY --chown=cogito:cogito config/  config/
COPY --chown=cogito:cogito src/     src/
COPY --chown=cogito:cogito utils/   utils/
COPY --chown=cogito:cogito scripts/ scripts/

# Switch to non-root user
USER cogito

# FastAPI default port
EXPOSE 8000

# Healthcheck — hit the root endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default entrypoint: run the FastAPI server
CMD ["uvicorn", "src.frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]