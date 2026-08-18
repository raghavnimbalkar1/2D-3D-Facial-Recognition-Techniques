# Session Status — ivafr

Updated: 2026-08-18, ~22:10 IST

## What we did this session

1. **Clean-clone & repo audit**:
   - Clean git working tree; all previous code committed.
   - Identified that prior `/private/tmp` batch directory was cleaned by macOS.
   - Decided to pivot to the **Tufts Face Database** for real 2D-vs-3D comparison instead of Yale B.

2. **Tufts Face Database Integration (Real 3D & 2D)**:
   - Fetched and unzipped TD_3D (110 SfM PLY meshes) and TD_RGB_E (550 expression photos: neutral, smile, eyes_closed, shocked, sunglasses) under `data/raw/`.
   - Created `src/ivafr/datasets/tufts3d.py` adapter with fast ASCII PLY parser (<100ms/mesh).
   - Created `src/ivafr/preprocess/mesh_to_depth.py` for orthographic mesh projection, centering, outlier filtering, and depth-grid interpolation.
   - Extended `preprocess_run.py` to seamlessly project PLY meshes into range images (`_r64.npy`), surface normals (`_n64.npy`), and curvature maps (`_c64.npy`).
   - Fixed frozen dataclass `Cloud3D.__post_init__` and elementwise NA validation in `manifest.py`.
   - Ingested Tufts dataset: 110 subjects, 660 total samples (550 2D, 110 3D).
   - Generated P1_closed and P2_disjoint splits across seeds 0–4 (10 split files).
   - Leakage audit passed: 10 split files checked, 0 violations.

3. **Experiment Configuration**:
   - `configs/datasets/tufts3d.yaml`: Tufts dataset definition.
   - `configs/preprocess/p3d_tufts.yaml`: Tufts 3D preprocessing settings.
   - `configs/experiments/E11.yaml`: Tufts 2D baselines (PCA, LBP, HOG, Gabor).
   - `configs/experiments/E12.yaml`: Tufts 3D baselines (DepthPCA, DepthLBP, NormalHOG, CurvHist).
   - `configs/experiments/E13.yaml`: Tufts 2D vs 3D benchmark comparison.

4. **Testing & Tooling**:
   - Added `tests/test_tufts3d.py` covering PLY parsing, mesh projection, adapter registration, and manifest validation.
   - Full test suite: **71 passed** (0 failures).
   - Added `tufts-ingest`, `tufts-preprocess`, `tufts-splits`, `tufts-run`, `tufts-all` targets to `Makefile`.

## Current State

- Preprocessing running in background (`ivafr preprocess --dataset tufts3d`).
- Next step: Run Tufts benchmark (`ivafr run --exp E11 E12 E13`) and aggregate results (`ivafr aggregate`).