> Historical snapshot. Current implementation and acceptance status: [IMPLEMENTATION_STATUS](../docs/IMPLEMENTATION_STATUS.md). Earlier completion counts and running-process claims below are not current evidence. The real 3D protocol requires independent repeated captures.

# RESULTS

Generated from 41 run directories on `2` protocols.

<!-- preamble: docs/RESULTS_PREAMBLE.md — preserved verbatim on regeneration -->

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

## Data separation and limitations

Real Yale B rows and synthetic toy rows are kept separate by the `data_modality` field.
The toy pseudo-3D and fusion results are synthetic methodology validation only.
Real-data pseudo-3D is environment-blocked by the documented MediaPipe native runtime failure; no real 2D-versus-3D claim is made.
A prior Yale B checkpoint was invalidated after a CLAHE float-to-uint8 conversion bug collapsed cached crops; the cache was rebuilt and the conversion is regression-tested.
Aggregated tables report Rank-1 divided by the protocol's chance baseline (`1 / n_gallery`) as `Rank-1 / Chance`.

## T1 — Main comparison (mean ± std over seeds)

| dataset | data_modality | Method | Accuracy | Rank-1 / Chance | Precision | Recall | F1-Score | Processing Time |
|---|---|---|---|---|---|---|---|---|
| toy | synthetic_toy | 2D-PCA (E00) | 0.4167±0.0000 | 5.00x±0.00x | 0.6620±0.0000 | 0.4167±0.0000 | 0.4312±0.0000 | NA |
| tufts3d | real | 2D-Gabor (E11) | 0.9659±0.0000 | 106.25x±0.00x | 0.9685±0.0000 | 0.9659±0.0000 | 0.9617±0.0000 | NA |
| tufts3d | real | 2D-Gabor (E11) | 0.9477±0.0145 | 62.55x±0.96x | 0.9470±0.0104 | 0.9477±0.0145 | 0.9410±0.0154 | NA |
| tufts3d | real | 2D-HOG (E11) | 0.9614±0.0000 | 105.75x±0.00x | 0.9538±0.0000 | 0.9614±0.0000 | 0.9537±0.0000 | NA |
| tufts3d | real | 2D-HOG (E11) | 0.9652±0.0108 | 63.70x±0.72x | 0.9576±0.0114 | 0.9652±0.0108 | 0.9580±0.0125 | NA |
| tufts3d | real | 2D-LBP (E11) | 0.9091±0.0000 | 100.00x±0.00x | 0.9156±0.0000 | 0.9091±0.0000 | 0.9022±0.0000 | NA |
| tufts3d | real | 2D-LBP (E11) | 0.9197±0.0168 | 60.70x±1.11x | 0.9216±0.0208 | 0.9197±0.0168 | 0.9122±0.0201 | NA |
| tufts3d | real | 2D-PCA (E11) | 0.9023±0.0000 | 99.25x±0.00x | 0.8964±0.0000 | 0.9023±0.0000 | 0.8915±0.0000 | NA |
| tufts3d | real | 2D-PCA (E11) | 0.9008±0.0218 | 59.45x±1.44x | 0.9020±0.0222 | 0.9008±0.0218 | 0.8903±0.0264 | NA |

## Extended metrics (Rank-5, EER, AUC, MRR)

| dataset | data_modality | exp | arm | protocol | rank1 | chance_rank1 | rank1_over_chance | rank5 | mrr | eer | auc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| toy | synthetic_toy | E00 | 2D-PCA | P1_closed | 0.4167±0.0000 | 0.0833±0.0000 | 5.0000x±0.0000x | 0.6190±0.0000 | 0.5364±0.0000 | 0.4659±0.0000 | 0.5763±0.0000 |
| tufts3d | real | E11 | 2D-Gabor | P1_closed | 0.9659±0.0000 | 0.0091±0.0000 | 106.2500x±0.0000x | 0.9841±0.0000 | 0.9733±0.0000 | 0.0218±0.0000 | 0.9932±0.0000 |
| tufts3d | real | E11 | 2D-Gabor | P2_disjoint | 0.9477±0.0145 | 0.0152±0.0000 | 62.5500x±0.9585x | 0.9803±0.0068 | 0.9620±0.0091 | 0.0520±0.0057 | 0.9839±0.0030 |
| tufts3d | real | E11 | 2D-HOG | P1_closed | 0.9614±0.0000 | 0.0091±0.0000 | 105.7500x±0.0000x | 0.9727±0.0000 | 0.9668±0.0000 | 0.0409±0.0000 | 0.9858±0.0000 |
| tufts3d | real | E11 | 2D-HOG | P2_disjoint | 0.9652±0.0108 | 0.0152±0.0000 | 63.7000x±0.7159x | 0.9758±0.0087 | 0.9704±0.0092 | 0.0384±0.0045 | 0.9856±0.0026 |
| tufts3d | real | E11 | 2D-LBP | P1_closed | 0.9091±0.0000 | 0.0091±0.0000 | 100.0000x±0.0000x | 0.9523±0.0000 | 0.9298±0.0000 | 0.0614±0.0005 | 0.9816±0.0002 |
| tufts3d | real | E11 | 2D-LBP | P2_disjoint | 0.9197±0.0168 | 0.0152±0.0000 | 60.7000x±1.1096x | 0.9629±0.0115 | 0.9387±0.0130 | 0.0608±0.0056 | 0.9819±0.0021 |
| tufts3d | real | E11 | 2D-PCA | P1_closed | 0.9023±0.0000 | 0.0091±0.0000 | 99.2500x±0.0000x | 0.9727±0.0000 | 0.9340±0.0000 | 0.0461±0.0003 | 0.9861±0.0001 |
| tufts3d | real | E11 | 2D-PCA | P2_disjoint | 0.9008±0.0218 | 0.0152±0.0000 | 59.4500x±1.4405x | 0.9750±0.0087 | 0.9346±0.0120 | 0.0678±0.0035 | 0.9791±0.0025 |
