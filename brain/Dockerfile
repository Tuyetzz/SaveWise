# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# ---- Dev: used by the devcontainer; source is bind-mounted, deps installed on create
FROM base AS dev
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*
CMD ["sleep", "infinity"]

# ---- Prod
FROM base AS prod
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
