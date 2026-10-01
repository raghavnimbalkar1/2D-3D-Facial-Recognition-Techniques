# IVAFR: 2D and 3D Facial Recognition Techniques

IVAFR is a configuration-driven research benchmark for comparing classical 2D facial-recognition features with reconstructed 3D facial-geometry features.

The project asks which representation preserves identity information most reliably when appearance, expression, illumination, or the evaluation identity changes. Every experiment records its dataset, preprocessing configuration, feature extractor, matcher, protocol, seed, metrics, and runtime information.

## Current status

The four 2D methods (PCA, LBP, HOG and spatial Gabor) are implemented. The
historical Tufts E11 summary describes 550 photos, 110 subjects and 40 runs,
but the raw data and run artifacts are absent from this checkout. Those
numbers have **not been recertified** under the current validation rules.

The pipeline now consumes audited, persisted splits; checks raw/cache/run
content hashes; records conservative verification operating points and
subject-cluster confidence intervals; and measures feature/matching time.
E14 adds controlled probe occlusion for all four 2D methods.

Real E12/E13 evaluation is blocked by the available Tufts design: one mesh
per person cannot supply independent same-modality gallery and probe captures.
Reprojections or image-derived geometry cannot fill that requirement.

See [implementation status and acceptance commands](docs/IMPLEMENTATION_STATUS.md)
for current evidence and remaining gates. [Historical results](results/RESULTS.md)
are retained for comparison, not as proof of current completion.

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

Use Python 3.10–3.12. The checked-in dependency lock is consumed during setup,
never overwritten by it. From PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r docs/env_lockfile.txt
.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m pytest
```

On Linux/macOS, `make setup PYTHON=python3.12` installs the same lock;
`make test` includes end-to-end tests and coverage. `make all` runs and
audits the complete synthetic E00 matrix. Synthetic results validate the
methodology, not real-world recognition accuracy.

## Tufts 2D benchmark

The Tufts Face Database is not included in this repository. It must be obtained under its applicable research-use terms.

```bash
bash scripts/fetch_tufts.sh data/raw
make tufts-ingest
./.venv/bin/ivafr preprocess --dataset tufts3d --data-root data --modality 2d
./.venv/bin/ivafr splits --dataset tufts3d --data-root data --modality 2d
./.venv/bin/ivafr run --exp E11 --data-root data --results-root results/tufts-current
./.venv/bin/python scripts/verify_no_leakage.py --data-root data
./.venv/bin/python scripts/verify_experiment.py --exp E11 --data-root data --results-root results/tufts-current
./.venv/bin/ivafr aggregate --results-root results/tufts-current --out results/tufts-current
```

E11 requires 40 completed runs. On Windows, use `.venv/Scripts/ivafr.exe`
and `.venv/Scripts/python.exe` in the commands above; extract the downloaded
dataset to `data/raw/TD_RGB_E` (optionally `data/raw/TD_3D`). Photo-only and
mesh-matched cohorts are distinguished in `ingestion.json`. Generate splits
after preprocessing. Use a fresh results root when configuration, code or
data changes; aggregation rejects duplicate or incompatible run families.

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
