.PHONY: build up down test lint format run-ui run-api

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

run-ui:
	PYTHONPATH=src uv run --extra ui streamlit run streamlit_app.py

run-api:
	PYTHONPATH=src uv run uvicorn commerce_engine.api:app --host 127.0.0.1 --port 8000

