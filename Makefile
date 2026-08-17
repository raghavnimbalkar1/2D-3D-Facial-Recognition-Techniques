# ivafr Makefile — self-contained: every target resolves its own .venv (Python 3.11).
PY := .venv/bin/python
IVAFR := .venv/bin/ivafr
PYTEST := .venv/bin/pytest
UV := $(shell command -v uv 2>/dev/null)

.PHONY: setup setup-full toy ingest preprocess splits run aggregate robustness timing all test lint fmt

.venv/bin/python:
	@if [ -n "$(UV)" ]; then \
		uv venv --python 3.11 .venv && uv pip install -e '.[dev]'; \
	else \
		python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'; \
	fi

setup: .venv/bin/python
	$(PY) -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
	$(UV) pip freeze > docs/env_lockfile.txt || .venv/bin/pip freeze > docs/env_lockfile.txt

setup-full: setup
	$(UV) pip install -e '.[full,deep2d]' || .venv/bin/pip install -e '.[full,deep2d]'
	$(UV) pip freeze > docs/env_lockfile.txt || .venv/bin/pip freeze > docs/env_lockfile.txt

toy: .venv/bin/python
	$(IVAFR) dataset-build --name toy --data-root data

ingest: toy
	$(IVAFR) ingest --dataset toy --data-root data

preprocess: ingest
	$(IVAFR) preprocess --dataset toy --data-root data --modality both

splits: preprocess
	$(IVAFR) splits --dataset toy --data-root data --protocol P1_closed --protocol P2_disjoint --seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4

run: splits
	$(IVAFR) run --exp E00 --data-root data --results-root results

aggregate: run
	$(IVAFR) aggregate --results-root results --out results --preamble docs/RESULTS_PREAMBLE.md

robustness: aggregate
	$(IVAFR) robustness --exp E00 --results-root results

timing: aggregate
	$(IVAFR) timing --exp E00 --results-root results

all: aggregate robustness timing

test:
	$(PYTEST)

lint:
	$(UV) run ruff check src tests || ruff check src tests

fmt:
	$(UV) run black src tests || black src tests