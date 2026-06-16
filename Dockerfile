# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml, uv.lock, and README.md first to build dependencies layer
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (they will be cached if pyproject.toml or uv.lock don't change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now copy the source code
COPY src ./src

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Final runtime image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the compiled virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create and switch to non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "commerce_engine.api:app", "--host", "0.0.0.0", "--port", "8000"]
