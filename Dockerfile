# ──────────────────────────────────────────────────────────────────────
# Multi-stage Enterprise Dockerfile for Customer Churn Prediction System
# Base Image: Python 3.11 Slim (Debian Bookworm)
# Security: Runs under unprivileged non-root user (appuser:1000)
# ──────────────────────────────────────────────────────────────────────

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt


# Stage 2: Final Production Runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="Sofia Azzahra Alamsyah" \
      version="1.1.0" \
      description="Enterprise Customer Churn Prediction & Retention Analytics System"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    FLASK_ENV="production" \
    HOST="0.0.0.0" \
    PORT="5000"

# Install curl for container healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root system user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source code, assets & templates
COPY --chown=appuser:appgroup app.py config.py customer_churn_crisp_dm.py run.py ./
COPY --chown=appuser:appgroup services/ ./services/
COPY --chown=appuser:appgroup templates/ ./templates/
COPY --chown=appuser:appgroup static/ ./static/
COPY --chown=appuser:appgroup database/ ./database/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup reports/ ./reports/
COPY --chown=appuser:appgroup "Raw dataset/" "./Raw dataset/"
COPY --chown=appuser:appgroup requirements.txt ./
COPY --chown=appuser:appgroup .env.example ./.env

RUN mkdir -p uploads && chown -R appuser:appgroup uploads

USER appuser

EXPOSE 5000

# Docker healthcheck checking /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]
