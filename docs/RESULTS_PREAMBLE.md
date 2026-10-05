> Historical snapshot. Current implementation and acceptance status: [IMPLEMENTATION_STATUS](IMPLEMENTATION_STATUS.md). Earlier completion counts and running-process claims below are not current evidence. The real 3D protocol requires independent repeated captures.

# Results

## Current status

The Tufts real-data 2D benchmark is complete. The dataset contains 110
reconstructed meshes and 550 usable RGB expression photographs. E11 evaluated
four 2D methods—PCA, LBP, HOG, and spatial Gabor—under P1 closed-set and P2
subject-disjoint protocols, using five seeds per method and protocol.

This produces 40 valid E11 metric files. The split audit checked 20 split files
across toy and Tufts data and found 0 leakage violations. The 2D preprocessing
path produced valid aligned crops for all 550 Tufts photos. Because Tufts is a
controlled centered-portrait dataset, preprocessing attempts Haar detection
first and records an explicit centered-portrait fallback when Haar cannot find
the face.

## Main Tufts 2D results

Values below are means ± standard deviations over five seeds.

| Method | P1 Rank-1 | P1 EER | P1 AUC | P2 Rank-1 | P2 EER | P2 AUC |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 90.23% ± 0.00 | 4.61% ± 0.03 | 0.986 | 90.08% ± 2.18 | 6.78% ± 0.35 | 0.979 |
| LBP | 90.91% ± 0.00 | 6.14% ± 0.05 | 0.982 | 91.97% ± 1.68 | 6.08% ± 0.56 | 0.982 |
| HOG | 96.14% ± 0.00 | 4.09% ± 0.00 | 0.986 | 96.52% ± 1.08 | 3.84% ± 0.45 | 0.986 |
| Gabor | 96.59% ± 0.00 | 2.18% ± 0.00 | 0.993 | 94.77% ± 1.45 | 5.20% ± 0.57 | 0.984 |

P1 has 110 gallery identities, so chance Rank-1 is 0.91%. P2 has 66
evaluation identities, so chance Rank-1 is 1.52%. HOG is strongest on P2
identification, while Gabor is strongest on P1 identification and P1 EER.

## Dataset separation and limitations

Toy metrics are synthetic development checks. Yale B values in the repository
are historical corrected 2D checkpoints and are not mixed with Tufts results.
The Yale pseudo-3D route remains blocked by the local MediaPipe runtime. The
remaining real-data work is E12/E13, which covers the Tufts 3D and 2D-vs-3D
comparison; it does not block the completed 2D E11 benchmark.

## Reproduction

```bash
bash scripts/fetch_tufts.sh data/raw
make tufts-ingest
./.venv/bin/ivafr preprocess --dataset tufts3d --data-root data --modality 2d
make tufts-splits
./.venv/bin/ivafr run --exp E11 --data-root data --results-root results
./.venv/bin/python scripts/verify_no_leakage.py --data-root data
./.venv/bin/ivafr aggregate --results-root results --out results --preamble docs/RESULTS_PREAMBLE.md
```
