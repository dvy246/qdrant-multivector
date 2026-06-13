.PHONY: build up down test lint format

build:
	docker compose build --no-cache

up:
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest tests/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-e2e:
	uv run pytest tests/e2e/ -v

lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run ruff format .
