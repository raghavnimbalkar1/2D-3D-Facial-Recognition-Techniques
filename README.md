# ivafr

`ivafr` is a configuration-driven research benchmark for comparing 2D facial
recognition with monocular pseudo-3D facial geometry.

The project investigates how appearance-based and geometry-based facial
features behave under changing illumination, and whether combining both
modalities improves recognition. It is designed to make every experiment
reproducible: datasets, preprocessing, features, matchers, evaluation
protocols, and output locations are controlled by configuration files.

## Scope

This sprint focuses on two data sources:

- **Toy dataset:** deterministic synthetic faces with rendered RGB images,
  exact depth maps, and landmarks. This is used for development and CI.
- **Extended Yale Face Database B:** real frontal face images captured under
  varied lighting conditions. Raw Yale B images are not included in this
  repository and are used for the 2D benchmark only this sprint.

The 3D modality is pseudo-3D rather than sensor-based. Toy data provides
ground-truth depth. Yale B pseudo-3D is currently environment-blocked after
three failed MediaPipe runtime paths. Real depth sensors, point-cloud
processing, ICP, and depth CNNs are outside the current scope.

## Methodology

The benchmark provides a common pipeline for both modalities:

1. Ingest images and create a manifest with subject, sample, illumination, and
   quality metadata.
2. Detect and align faces, then apply configurable illumination normalization.
3. Extract 2D features such as PCA, LDA, LBP, HOG, and Gabor descriptors.
4. Extract pseudo-3D features from depth-like maps, surface derivatives, and
   FaceMesh landmarks.
5. Compare features using nearest-neighbour and scikit-learn matchers, with
   optional score-level and feature-level fusion.
6. Evaluate closed-set identification and verification using rank accuracy,
   CMC, precision/recall/F1, ROC/DET, EER, TAR at fixed FAR levels, and timing.

All experiments use subject-aware splits and fixed seeds. Each run records its
resolved configuration, system information, metrics, curves, and intermediate
artifacts under `results/runs/`.

## Quick start

Create an editable development installation and run the toy benchmark:

```bash
make setup
make all
```

For the optional toy pseudo-3D path, install the additional dependency:

```bash
python -m pip install -e '.[dev,full]'
```

Run the test suite with:

```bash
make test
```

The command-line interface exposes the individual stages as well:

```bash
ivafr --help
ivafr dataset-build --name toy --data-root data
ivafr ingest --dataset toy --data-root data
ivafr preprocess --dataset toy --data-root data --modality both
```

## Tufts Face Database setup

Tufts Face Database provides 112 participant SfM-reconstructed 3D meshes (TD_3D)
and 5-expression 2D photos (TD_RGB_E).

Fetch and extract the dataset via:

```bash
bash scripts/fetch_tufts.sh data/raw
```

Then run the full pipeline:

```bash
make tufts-all
```

Or individual stages:

```bash
ivafr ingest --dataset tufts3d --data-root data
ivafr preprocess --dataset tufts3d --data-root data --modality both
ivafr splits --dataset tufts3d --data-root data --protocol P1_closed --protocol P2_disjoint --seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4
ivafr run --exp E11 --exp E12 --exp E13 --data-root data --results-root results
ivafr aggregate --results-root results --out results --preamble docs/RESULTS_PREAMBLE.md
```

## Yale B setup

Download Extended Yale Face Database B from its [official distribution](https://cvc.cs.yale.edu/cvc/projects/yalefacesB/yalefacesB.html), extract it under:

```text
data/raw/yaleb/
```

Then run the dataset stages:

```bash
ivafr ingest --dataset yaleb --data-root data
ivafr preprocess --dataset yaleb --data-root data --modality 2d
```

Set the OpenCV DNN detector model paths before preprocessing real Yale B:

```bash
export IVAFR_FACE_DNN_PROTO=/path/to/deploy.prototxt
export IVAFR_FACE_DNN_MODEL=/path/to/res10_300x300_ssd_iter_140000.caffemodel
```

Yale B images remain local and are never copied into result artifacts. Real
Yale B metrics are tagged `data_modality: real`; toy 3D and fusion metrics are
tagged `data_modality: synthetic_toy` and must not be merged with real rows.

## Repository layout

```text
configs/       Dataset, preprocessing, feature, matcher, and experiment configs
src/ivafr/     Dataset adapters, preprocessing, features, matching, and evaluation
scripts/       Utility scripts for toy data and leakage checks
tests/         Unit and end-to-end tests
docs/          Dataset, protocol, ethics, decisions, and environment notes
results/       Generated runs, figures, tables, and summaries
```

## Current status

The project supports procedural toy validation (`make all`) and real 2D vs 3D
evaluation on the Tufts Face Database (`make tufts-all`). Reconstructed 3D PLY
meshes are projected to range images, surface normals, and curvature maps,
enabling comparison against 2D appearance features (PCA, LBP, HOG, Gabor).

## License and data

The project code is released under the MIT License. Dataset terms are separate
from the code license; see `docs/DATASETS.md` and `docs/ETHICS.md` before
redistributing data or derived artifacts.
