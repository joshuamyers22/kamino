.PHONY: setup format lint typecheck repository-check test check build wheel-smoke supply-chain release-verify formula-spike walking-skeleton backend-spike benchmark benchmark-sparse benchmark-smoke benchmark-e03 benchmark-e03-native benchmark-e03-lme4 benchmark-e03-assess benchmark-e03-verify statistical statistical-i02 statistical-i03 statistical-a02 oracle oracle-verify

setup:
	uv sync --frozen --all-extras --dev

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run pyright

repository-check:
	uv run python tools/verify_repository.py

test:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck repository-check test build

build:
	uv build --offline

wheel-smoke: build
	uv run python tools/wheel_smoke.py

supply-chain: build
	uv run python tools/release_artifacts.py

release-verify:
	uv run python tools/verify_release.py --tag "$(RELEASE_TAG)"

formula-spike:
	uv run python tools/formula_spike.py

walking-skeleton:
	uv run python tools/walking_skeleton.py

backend-spike:
	uv run python tools/backend_spike.py

benchmark:
	uv run pytest -q tests/test_lme4_oracle.py tests/test_random_intercept_block.py tests/test_resource_benchmark.py
	uv run python tools/benchmark_single_group.py
	uv run python tools/benchmark_general_sparse.py

benchmark-sparse:
	uv run python tools/benchmark_general_sparse.py

benchmark-smoke:
	uv run python tools/benchmark_single_group.py --smoke

benchmark-e03: benchmark-e03-native benchmark-e03-lme4 benchmark-e03-assess

benchmark-e03-native:
	mkdir -p .work/e03
	uv run python tools/benchmark_single_group.py --manifest benchmarks/e03_single_group_v1.json --output .work/e03/kamino_single_group.json
	uv run python tools/benchmark_general_sparse.py --manifest benchmarks/e03_general_sparse_v1.json --output .work/e03/kamino_general_sparse.json

benchmark-e03-lme4:
	mkdir -p .work/e03
	docker run --rm --mount type=bind,source=$(CURDIR),target=/work --workdir /work --entrypoint Rscript ghcr.io/joshuamyers22/kamino-oracle@sha256:17e45268be294316967064d0600a727463d738d7dfb0c68ec43769baebe8eb4d oracle/benchmark_e03.R --single-manifest benchmarks/e03_single_group_v1.json --sparse-manifest benchmarks/e03_general_sparse_v1.json --output .work/e03/lme4.json --revision "$$(git rev-parse HEAD)"

benchmark-e03-assess:
	uv run python tools/assess_e03.py --single .work/e03/kamino_single_group.json --sparse .work/e03/kamino_general_sparse.json --lme4 .work/e03/lme4.json --output .work/e03/assessment.json

benchmark-e03-verify:
	uv run python tools/verify_e03.py

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

oracle-verify:
	uv run python tools/verify_oracle_output.py --fixtures-only
