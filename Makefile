# Linux/macOS convenience targets; PowerShell equivalents are in README.
PYTHON ?= python3.12
PY := .venv/bin/python
IVAFR := .venv/bin/ivafr
DATA ?= data
RESULTS ?= results/toy-current
TUFTS_RESULTS ?= results/tufts-current

.PHONY: setup setup-full toy ingest preprocess splits run verify aggregate all test lint fmt tufts-ingest tufts-preprocess tufts-splits tufts-run tufts-verify tufts-all

.venv/bin/python:
	$(PYTHON) -m venv .venv

setup: .venv/bin/python
	$(PY) -m pip install -r docs/env_lockfile.txt
	$(PY) -m pip install --no-deps --no-build-isolation -e .
	$(PY) -m pip check

setup-full: setup
	$(PY) -m pip install -e '.[full,deep2d]'

toy: .venv/bin/python
	$(IVAFR) dataset-build --name toy --data-root $(DATA)

ingest: toy
	$(IVAFR) ingest --dataset toy --data-root $(DATA)

preprocess: ingest
	$(IVAFR) preprocess --dataset toy --data-root $(DATA) --modality both

splits: preprocess
	$(IVAFR) splits --dataset toy --data-root $(DATA) --modality both

run: splits
	$(IVAFR) run --exp E00 --data-root $(DATA) --results-root $(RESULTS)

verify: run
	$(PY) scripts/verify_no_leakage.py --data-root $(DATA)
	$(PY) scripts/verify_experiment.py --exp E00 --data-root $(DATA) --results-root $(RESULTS)

aggregate: verify
	$(IVAFR) aggregate --results-root $(RESULTS) --out $(RESULTS)

all: aggregate

tufts-ingest: .venv/bin/python
	$(IVAFR) ingest --dataset tufts3d --data-root $(DATA)

tufts-preprocess: tufts-ingest
	$(IVAFR) preprocess --dataset tufts3d --data-root $(DATA) --modality 2d

tufts-splits: tufts-preprocess
	$(IVAFR) splits --dataset tufts3d --data-root $(DATA) --modality 2d

tufts-run: tufts-splits
	$(IVAFR) run --exp E11 --exp E14 --data-root $(DATA) --results-root $(TUFTS_RESULTS)

tufts-verify: tufts-run
	$(PY) scripts/verify_no_leakage.py --data-root $(DATA)
	$(PY) scripts/verify_experiment.py --exp E11 --data-root $(DATA) --results-root $(TUFTS_RESULTS)
	$(PY) scripts/verify_experiment.py --exp E14 --data-root $(DATA) --results-root $(TUFTS_RESULTS)

tufts-all: tufts-verify
	$(IVAFR) robustness --exp E14 --results-root $(TUFTS_RESULTS)
	$(IVAFR) aggregate --results-root $(TUFTS_RESULTS) --out $(TUFTS_RESULTS)

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check --select E9,F821,F822,F823 src tests scripts

fmt:
	$(PY) -m black src tests scripts
