.PHONY: setup format lint typecheck test check build wheel-smoke formula-spike walking-skeleton backend-spike oracle

setup:
	uv sync --frozen --all-extras --dev

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test build

build:
	uv build --offline

wheel-smoke: build
	uv run python tools/wheel_smoke.py

formula-spike:
	uv run --extra formula-spike python tools/formula_spike.py

walking-skeleton:
	uv run --extra formula-spike --extra backend-spike python tools/walking_skeleton.py

backend-spike:
	uv run --extra backend-spike python tools/backend_spike.py

oracle:
	docker build --provenance=false --tag kamino-oracle:phase0 --file oracle/Dockerfile .
	docker run --rm --mount type=bind,source=$(CURDIR),target=/work kamino-oracle:phase0
	uv run python tools/verify_oracle_output.py --image
	uv run pytest -q tests/test_lme4_oracle.py
