# IVAFR: 2D and 3D Facial Recognition Techniques

IVAFR is a configuration-driven research benchmark for comparing classical 2D facial-recognition features with reconstructed 3D facial-geometry features.

The project asks which representation preserves identity information most reliably when appearance, expression, illumination, or the evaluation identity changes. Every experiment records its dataset, preprocessing configuration, feature extractor, matcher, protocol, seed, metrics, and runtime information.

## Current status

The real-data 2D benchmark on the Tufts Face Database is complete.

- 550 usable 2D portrait photographs from 110 subjects
- Four methods: PCA, LBP, HOG, and spatial Gabor
- P1 closed-set and P2 subject-disjoint protocols
- Five seeds per method and protocol
- 40 valid E11 metric files
- 20 split files audited with 0 leakage violations
- 69 focused tests passing

The Tufts 3D and final 2D-versus-3D experiments are configured but remain separate follow-up work. Extended Yale B results are retained as historical 2D checkpoints; the earlier Yale pseudo-3D route is environment-blocked by the local MediaPipe runtime.

Final 2D results are available in [results/RESULTS.md](results/RESULTS.md). The complete project explanation and presentation guide is in [PROJECT_REPORT_AND_PRESENTATION_GUIDE.md](PROJECT_REPORT_AND_PRESENTATION_GUIDE.md).

## Methods

- **PCA / Eigenfaces:** global appearance representation learned from the permitted training samples.
- **LBP:** local texture descriptor based on neighborhood intensity comparisons.
- **HOG:** local edge-orientation descriptor that emphasizes facial contours.
- **Gabor:** multi-scale, multi-orientation response maps with spatial downsampling and train-only PCA.

The benchmark reports Rank-1 and Rank-5 identification, CMC, EER, ROC/AUC, verification operating points, and timing where configured.

## Evaluation protocols

- **P1 closed-set:** gallery and probe identities belong to the same identity pool.
- **P2 subject-disjoint:** representation fitting uses one subject pool and evaluation uses separate unseen subjects.

All splits are subject-aware and generated with fixed seeds. Feature reductions such as PCA are fitted only on the permitted training partition to avoid leakage.

## Reproducibility

The project uses Python 3.11 and a self-contained virtual environment managed by the Makefile.

```bash
make setup
make test
```

The functional test command used in the restricted audit environment is:

```bash
./.venv/bin/python -m pytest tests --ignore=tests/test_e2e.py -q -o addopts='' -p no:cov
```

## Tufts 2D benchmark

The Tufts Face Database is not included in this repository. It must be obtained under its applicable research-use terms.

```bash
bash scripts/fetch_tufts.sh data/raw
make tufts-ingest
./.venv/bin/ivafr preprocess --dataset tufts3d --data-root data --modality 2d
make tufts-splits
./.venv/bin/ivafr run --exp E11 --data-root data --results-root results
./.venv/bin/python scripts/verify_no_leakage.py --data-root data
./.venv/bin/ivafr aggregate --results-root results --out results --preamble docs/RESULTS_PREAMBLE.md
```

E11 evaluates four 2D methods across P1 and P2 with five seeds. The expected output is 40 `metrics.json` files and an updated `results/RESULTS.md`.

## Pipeline

```text
raw images
  -> ingestion and manifest
  -> face preprocessing and alignment
  -> subject-aware splits
  -> feature extraction
  -> similarity matching
  -> identification and verification metrics
  -> aggregation and report generation
```

For Tufts studio portraits, Haar detection is attempted first. If it cannot detect the centered portrait, the configured centered-portrait fallback is recorded explicitly in the preprocessing output. This assumption is specific to the controlled dataset and is not presented as a general-purpose detector.

## Repository layout

```text
configs/       Dataset, preprocessing, feature, matcher, and experiment configs
src/ivafr/     Dataset adapters, preprocessing, features, matching, and evaluation
scripts/       Dataset and verification utilities
tests/         Unit and integration tests
docs/          Dataset, protocol, ethics, decisions, and result preamble
results/       Tracked summaries; generated runs and tables are ignored
```

## Data and generated files

Raw datasets, processed data, caches, model weights, virtual environments, and generated run artifacts are excluded by `.gitignore`. Source code, configuration, tests, documentation, and the aggregated result summary remain trackable.

See [docs/DATASETS.md](docs/DATASETS.md) and [docs/ETHICS.md](docs/ETHICS.md) for dataset terms and handling guidance.

## License

The project code is released under the MIT License. Dataset licenses and research-use terms are separate from the code license.
