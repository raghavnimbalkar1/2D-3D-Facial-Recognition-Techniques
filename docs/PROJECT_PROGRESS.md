# Project Progress

**Snapshot date:** 2026-08-18 (mid-batch)

## Overall estimate

| Basis | Progress | Meaning |
|---|---:|---|
| Amended sprint deliverable | **70%** | Real Yale 2D benchmark plus toy pseudo-3D methodology validation |
| Original real 2D-versus-3D goal | **35%** | Real pseudo-3D is intentionally blocked by the environment |

## Milestone breakdown

| Area | Progress | Status |
|---|---:|---|
| Repository architecture and configuration | 90% | Implemented and tested |
| Toy 2D and pseudo-3D pipeline | 90% | Reproducible development path |
| Yale B ingestion and manifest | 100% | 2,414 images, 38 subjects |
| Yale B detection and alignment | 100% | All usable images succeeded |
| Illumination preprocessing correctness | 100% | CLAHE conversion fixed and regression-tested |
| Gabor descriptor rework | 100% | Spatial maps + train-only PCA; validated P1 seed-0 43.43% vs old 17.21% |
| Run idempotency | 100% | Skip-bug fixed: existing `metrics.json` now detected by glob, re-runs skip |
| Test suite | 100% | 67 passed, 70% coverage |
| E01 P1 full matrix | 100% | PCA/LBP/HOG/Gabor x seeds 0-4 (Gabor uses the new descriptor) |
| E01 P2 full matrix | In progress | PCA/LBP/HOG s0 done (pre-batch); s1-4 + Gabor s0-4 running now |
| E05 illumination sweep | In progress | Queued in the running batch (5 seeds, PCA) |
| E08 occlusion sweep | In progress | Queued in the running batch (5 seeds x 4 conditions) |
| E10 timing sweep | In progress | Queued in the running batch (5 seeds, PCA) |
| Final aggregation and results review | 20% | Requires completed matrix |
| Explicit leakage audit | 100% | 4 split files checked, 0 violations; rerun after batch |
| Clean-clone `make all` verification | 0% | Pending |

## Run matrix (corrected)

The handoff estimated 40 E01 runs; the post-interruption disk held **23
metrics.json** (P1: 20 runs; P2: only seed-0 for PCA/LBP/HOG). The interrupt
hit mid-P2. Remaining work was relaunched on 2026-08-18 as one background
batch: P2 s1-4 (PCA/LBP/HOG), Gabor P1 s1-4 + P2 s0-4 (new descriptor),
then E05/E08/E10.

The five pre-interruption P1 Gabor runs used the old 40-scalar descriptor and
were moved to `final-results/archive_old_gabor/` so aggregation can never mix
them with the new descriptor runs.

## Corrected anchor numbers (single seed, pre-batch)

| Protocol | Rank-1 | EER | AUC | Rank-1 / Chance |
|---|---:|---:|---:|---:|
| P1 closed, PCA | 39.56% | 40.58% | 0.621 | 15.03x |
| P2 disjoint, PCA | 48.82% | 48.30% | 0.539 | 11.23x |

Gabor rework validation (P1, seed 0): old descriptor rank-1 17.21% with
genuine-impostor score separation 0.0031; new descriptor rank-1 43.43%
(16.5x chance), separation 0.1974, EER 38.0%. Score separation is the
decisive check — the old implementation collapsed spatial structure.

## Completion criteria still open

- Batch: E01 P2 remainder + E05 + E08 + E10 ~~ (running)
- Dedupe run dirs created before the skip-bug fix (keep newest per
  exp/arm/protocol/seed)
- Aggregation generated from the final results directory
- `RESULTS.md` updated with final real-data values and limitations
- Explicit leakage audit rerun after the batch
- Clean toy `make all` run completed
- No real/synthetic modality mixing
- No real Yale pseudo-3D claim