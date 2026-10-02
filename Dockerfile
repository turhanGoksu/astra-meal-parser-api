# syntax=docker/dockerfile:1
# Multi-stage build: compilers stay in the builder; the runtime image gets
# only the virtualenv and the app. Model weights are never baked in: they
# live in the /models volume (downloaded on first start).
#
#   docker compose build                     # slim (default), ~0.5 GB
#   WITH_JUDGE=true docker compose build     # + LLM judge (CPU PyTorch), ~2 GB

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv

WORKDIR /build
# Dependencies first: this layer is rebuilt only when requirements change.
# It compiles llama.cpp (minutes), so it comes before WITH_JUDGE is declared:
# a changed build arg invalidates the cache from its declaration on, and both
# image variants share this layer.
COPY requirements.txt ./
RUN pip install -r requirements.txt

ARG WITH_JUDGE=false
COPY requirements-judge.txt ./
RUN if [ "$WITH_JUDGE" = "true" ]; then pip install -r requirements-judge.txt; fi

# The astra-nutrition package (its dependencies are already installed).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# llama.cpp uses OpenMP at runtime (libgomp); nothing else from the builder.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/models \
    HF_HOME=/hf-cache

# Non-root user; the volume mount points are created owned by it so named
# volumes start with the right permissions.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models /hf-cache \
    && chown app:app /models /hf-cache

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
COPY scripts/download_model.py ./scripts/download_model.py
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

USER app
EXPOSE 8000

# /health checks the database only (it never runs the model). The long start
# period covers the first start, when the ~1 GB model is downloaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"

ENTRYPOINT ["entrypoint.sh"]
