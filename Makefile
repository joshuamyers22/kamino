.PHONY: setup format lint typecheck test check build wheel-smoke formula-spike walking-skeleton backend-spike benchmark benchmark-sparse benchmark-smoke statistical statistical-i02 statistical-i03 statistical-a02 oracle

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

benchmark:
	uv run pytest -q tests/test_lme4_oracle.py tests/test_random_intercept_block.py tests/test_resource_benchmark.py
	uv run python tools/benchmark_single_group.py
	uv run python tools/benchmark_general_sparse.py

benchmark-sparse:
	uv run python tools/benchmark_general_sparse.py

benchmark-smoke:
	uv run python tools/benchmark_single_group.py --smoke

statistical:
	uv run python tools/statistical_i01.py --verify statistical/i01_report.json
	uv run python tools/statistical_i02.py --verify statistical/i02_report.json
	uv run python tools/statistical_i03.py --verify statistical/i03_report.json
	uv run python tools/statistical_a02.py --verify statistical/a02_report.json

statistical-i02:
	uv run python tools/statistical_i02.py --verify statistical/i02_report.json

statistical-i03:
	uv run python tools/statistical_i03.py --verify statistical/i03_report.json

statistical-a02:
	uv run python tools/statistical_a02.py --verify statistical/a02_report.json

oracle:
	docker build --provenance=false --tag kamino-oracle:phase0 --file oracle/Dockerfile .
	docker run --rm --mount type=bind,source=$(CURDIR),target=/work kamino-oracle:phase0
	uv run python tools/verify_oracle_output.py --image
	uv run pytest -q tests/test_lme4_oracle.py
