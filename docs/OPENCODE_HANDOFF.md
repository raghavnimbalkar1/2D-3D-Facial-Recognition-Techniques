# OpenCode Handoff

**Snapshot date:** 2026-08-17
**Repository:** `/Users/raghavnimbalkar/Desktop/2d-3d-face`

## Project objective

`ivafr` is a configuration-driven facial-recognition benchmark comparing
classical 2D facial features with monocular pseudo-3D geometry. The original
plan was a real-data 2D-versus-3D comparison. The amended sprint scope is:

- rigorous real-data 2D benchmarking on Extended Yale Face Database B;
- complete pseudo-3D and fusion validation on the deterministic toy dataset;
- explicit documentation that real Yale B pseudo-3D is blocked in the current
  macOS environment.

Do not reopen the MediaPipe investigation during this sprint. The legacy API,
Tasks API with CPU delegate, and isolated MediaPipe 0.10.9 all failed during
native graph initialization.

## Current implementation state

- Yale B adapter: discovers 2,414 usable images across 38 subjects.
- Yale B preprocessing: OpenCV ResNet-10 DNN detection, similarity alignment,
  corrected illumination normalization, and 64x64 grayscale crops.
- All 2,414 Yale images were successfully detected and aligned.
- Real and synthetic data are separated using `data_modality`:
  `real` versus `synthetic_toy`.
- Pseudo-3D arms are rejected for real manifests and allowed only for toy data.
- P1 uses a 38-image gallery; P2 uses 15 background subjects and 23
  evaluation subjects with a 23-way gallery.
- Aggregation now reports raw Rank-1, chance Rank-1, and Rank-1 divided by
  chance.
- The CLAHE cache-collapse bug was fixed and a pixel-spread regression test
  was added.

## Corrected anchor results

These are single-seed PCA checks after deleting and rebuilding the Yale crop
cache. They are valid engineering checkpoints, not final five-seed results.

| Protocol | Rank-1 | EER | AUC | Chance | Rank-1 / Chance |
|---|---:|---:|---:|---:|---:|
| P1 closed | 39.56% | 40.58% | 0.621 | 2.63% | 15.03x |
| P2 disjoint | 48.82% | 48.30% | 0.539 | 4.35% | 11.23x |

P1 condition-level performance reaches 100% on several moderate-light
conditions and falls to chance on extreme conditions. P2 shows the same
pattern, while verification remains weak for unseen identities.

## Active full benchmark

The active command is running the complete amended real-data matrix:

```bash
mkdir -p /private/tmp/ivafr-mplconfig
MPLCONFIGDIR=/private/tmp/ivafr-mplconfig \
IVAFR_FACE_DNN_PROTO=/private/tmp/ivafr-yale-dnn/deploy.prototxt \
IVAFR_FACE_DNN_MODEL=/private/tmp/ivafr-yale-dnn/res10_300x300_ssd_iter_140000.caffemodel \
./.venv/bin/ivafr run \
  --exp E01 --exp E05 --exp E08 --exp E10 \
  --data-root /private/tmp/ivafr-yale-valid \
  --results-root /private/tmp/ivafr-yale-valid/final-results
```

At this snapshot, 19 `metrics.json` files exist under:

```text
/private/tmp/ivafr-yale-valid/final-results/runs/
```

Refresh the count before making a final status statement:

```bash
find /private/tmp/ivafr-yale-valid/final-results/runs \
  -name metrics.json -print | wc -l
```

Gabor is the slowest arm. Do not interpret its long runtime as a failure
unless the process exits or produces an error.

## Required next steps after the run finishes

1. Confirm all configured outputs exist and contain `data_modality: real`.
2. Aggregate the final results:

   ```bash
   ./.venv/bin/ivafr aggregate \
     --results-root /private/tmp/ivafr-yale-valid/final-results \
     --out /private/tmp/ivafr-yale-valid/final-results
   ```

3. Review `T1_main_comparison.csv`, `T1_extended.csv`, and the robustness
   condition tables.
4. Transfer the reviewed summary into the repository's
   `results/RESULTS.md`; do not copy raw Yale images or model weights into the
   repository.
5. Run the toy reproduction in a clean environment:

   ```bash
   make setup
   make all
   make test
   ```

6. Run the explicit leakage audit:

   ```bash
   ./.venv/bin/python scripts/verify_no_leakage.py \
     --data-root /private/tmp/ivafr-yale-valid
   ```

7. Verify that toy rows are tagged `synthetic_toy` and are never mixed with
   Yale rows tagged `real`.
8. Keep the MediaPipe limitation and the CLAHE correction in
   `docs/DECISIONS.md`.

## Important paths

- Main README: `README.md`
- Scope amendment: `docs/EXECUTION_SPEC_AMENDMENT.md`
- Decisions: `docs/DECISIONS.md`
- Results status: `results/RESULTS.md`
- Progress report: `docs/PROJECT_PROGRESS.md`
- Yale data root used for validation: `/private/tmp/ivafr-yale-valid`
- DNN model files: `/private/tmp/ivafr-yale-dnn/`
- Active results: `/private/tmp/ivafr-yale-valid/final-results/`

The temporary Yale and model paths are machine-local and are not part of the
repository transfer. The next agent must configure equivalent local paths.

## Do not do

- Do not report the pre-CLAHE-fix E05, E08, or E10 numbers.
- Do not run real Yale pseudo-3D or fusion arms.
- Do not merge toy and Yale metrics.
- Do not add Open3D, Torch, Trimesh, or a replacement pseudo-3D pipeline in
  this sprint.
- Do not claim a completed benchmark until all configured runs, aggregation,
  results review, and clean toy reproduction are complete.
