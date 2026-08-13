.PHONY: setup toy ingest preprocess splits run aggregate robustness timing all test lint

setup:
	python -m pip install -e '.[dev]'

toy:
	ivafr dataset-build --name toy --data-root data

ingest: toy
	ivafr ingest --dataset toy --data-root data

preprocess: ingest
	ivafr preprocess --dataset toy --data-root data --modality both

splits: preprocess
	ivafr splits --dataset toy --data-root data --protocol P1_closed --protocol P2_disjoint --seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4

run: splits
	ivafr run --exp E00 --data-root data --results-root results

aggregate: run
	ivafr aggregate --results-root results --out results

robustness: aggregate
	ivafr robustness --exp E00 --results-root results

timing: aggregate
	ivafr timing --exp E00 --results-root results

all: aggregate robustness timing

test:
	pytest

lint:
	ruff check src tests
