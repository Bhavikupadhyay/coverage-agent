# syntax=docker/dockerfile:1.7
# CoverageAgent — production image
# Multi-stage build keeps the runtime layer lean by isolating uv + build deps.

# ---------------------------------------------------------------------------
# Stage 1 — install deps with uv into a virtual environment we can copy.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY coverage_agent ./coverage_agent

# Compile a frozen, runtime-only env. Dev deps stay out of the runtime image.
RUN uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Stage 2 — minimal runtime image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# git is still needed at runtime — the orchestrator shells out to git when
# preparing repos for the E2B sandbox.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app.py run_benchmark.py ./
COPY coverage_agent ./coverage_agent
COPY public ./public

# .local stores the SQLite run history. Mount a volume here in compose to
# persist runs across container restarts.
RUN mkdir -p /app/.local

EXPOSE 8000

# Use exec form so SIGTERM reaches uvicorn cleanly (graceful shutdown).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
