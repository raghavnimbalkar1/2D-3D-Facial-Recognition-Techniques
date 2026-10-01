> Historical snapshot. Current implementation and acceptance status: [IMPLEMENTATION_STATUS](docs/IMPLEMENTATION_STATUS.md). Earlier completion counts and running-process claims below are not current evidence. The real 3D protocol requires independent repeated captures.

# 2D and 3D Facial Recognition Techniques

## Current project report and presentation guide

**Audit date:** 30 September 2026
**Purpose:** Explain the project, record what is implemented and validated, and provide a safe way to present the 2D work.

## 1. What we are building

This project is a reproducible benchmark for comparing facial-recognition methods that use:

- **2D appearance:** intensity, texture, edges, and frequency responses from face images.
- **3D shape:** depth, surface normals, and curvature derived from reconstructed face geometry.

The scientific question is: **when lighting, expression, occlusion, or identity changes, which representation and feature extractor preserves identity information most reliably?**

The benchmark is config-driven. Every experiment records its dataset, protocol, feature extractor, matcher, random seed, metrics, and timing so that results can be regenerated rather than demonstrated as a one-off script.

## 2. What is implemented

### Core pipeline

The repository contains the following stages:

1. **Dataset adapter and ingestion** create a validated manifest of sample IDs, subjects, modalities, and source files.
2. **Preprocessing** detects or loads faces, aligns them, normalizes them, and writes cached 2D features or 3D-derived maps.
3. **Splitting** creates reproducible gallery/probe and train/evaluation partitions for multiple seeds.
4. **Feature extraction** produces comparable descriptors.
5. **Matching** computes similarity or distance between gallery and probe descriptors.
6. **Evaluation** reports identification and verification metrics.
7. **Aggregation** combines runs into tables and a written results report.

### 2D feature families

- **PCA / Eigenfaces:** projects a face image into a lower-dimensional subspace learned from training images. It is a useful classical appearance baseline, but can be sensitive to lighting.
- **LBP:** summarizes local texture using binary intensity comparisons. It is comparatively robust to monotonic illumination changes.
- **HOG:** summarizes local edge-orientation structure. It emphasizes facial shape and contour rather than raw pixel brightness.
- **Gabor:** measures responses to multiple spatial frequencies and orientations. The descriptor was corrected to preserve spatial response maps, downsample them, and fit PCA only on training data.

### 3D feature families

The current real-data 3D path uses Tufts reconstructed meshes and derives:

- depth/range images;
- surface-normal maps;
- curvature histograms or maps;
- depth and normal feature baselines.

The earlier Yale pseudo-3D route using MediaPipe was blocked by the local native graph runtime. The project therefore pivoted to Tufts, where real 2D expression photographs and reconstructed 3D meshes are available in one dataset family. This is methodologically stronger than presenting a synthetic pseudo-3D result as a real-data result.

## 3. Datasets and protocols

### Toy data

The procedural toy dataset is for development, determinism, and testing. It is not evidence of real-world recognition performance.

### Extended Yale B

The Yale B work established a real 2D path and exposed important engineering issues, including face-detection/runtime limitations and a preprocessing conversion bug. The corrected PCA checkpoint recorded in the repository was:

| Protocol | Rank-1 | EER | AUC | Interpretation |
|---|---:|---:|---:|---|
| P1 closed-set | 39.56% | 40.58% | 0.621 | Corrected real-data checkpoint |
| P2 disjoint | 48.82% | 48.30% | 0.539 | Unseen-subject transfer checkpoint |

These values are historical checkpoints from the earlier Yale work. They are not a replacement for current Tufts metrics, and they should not be presented as the final 2D-vs-3D benchmark.

### Tufts Face Database

Tufts is now the intended real-data comparison dataset. The repository includes an adapter, ASCII PLY parsing, 2D/3D preprocessing support, split generation, E11/E12/E13 experiment configurations, and Tufts-specific tests.

The real Tufts data has now been restored locally: 110 meshes, 561 RGB files
discovered by the download, 660 manifest records, and 550 usable 2D photo
samples. The 2D preprocessing path uses Haar first and an explicit centered-
portrait fallback for this controlled studio dataset. All 550 photo samples
produced valid aligned crops; the 110 mesh-only records are intentionally not
included in the 2D E11 experiment.

The two evaluation protocols are:

- **P1 closed-set:** identities in the gallery are also represented among the probe identities.
- **P2 subject-disjoint:** identities used to fit the representation are separated from identities used for evaluation. This tests generalization and is the more demanding protocol.

The split design includes five random seeds. Verification metrics must be interpreted alongside identification metrics because Rank-1 depends on the number of gallery subjects, while EER and AUC are less affected by gallery size.

## 4. Verification status as of this audit

The percentages below are intentionally split between engineering completion and scientific-result completion.

During the final audit, 69 non-end-to-end tests passed. The Tufts-specific tests, feature tests, metric tests, and the new centered-portrait fallback regression test passed. The default coverage plugin still cannot create its SQLite data file in this restricted environment, so the functional test count is reported without coverage. The leakage checker reports **20 split files checked and 0 violations**: ten toy splits and ten Tufts splits.

| Area | Progress | Evidence and remaining work |
|---|---:|---|
| Pipeline architecture and CLI | 95% | Dataset, preprocessing, feature, matcher, metric, aggregation, and CLI modules exist. |
| 2D feature implementation | 95% | PCA, LBP, HOG, and spatial Gabor are implemented, tested, and executed in the full E11 matrix. |
| Toy validation | 85% | The current checkout generated toy data, splits, and one E00 PCA P1 seed successfully. That current synthetic checkpoint is Rank-1 41.67%, Rank-5 61.90%, EER 46.59%, and AUC 0.576; it is a pipeline sanity check, not real-world evidence. |
| Yale 2D investigation | 85% | Real-data preprocessing and corrected PCA checkpoints exist; final refreshed Yale experiment artifacts are not present locally. |
| Tufts dataset integration | 95% | Adapter, PLY parser, mesh projection, configs, Makefile targets, downloaded data, preprocessing, splits, and tests are present. |
| Tufts 2D benchmark results | 100% | E11 completed all 40 configured runs: four methods, two protocols, and five seeds. Results were aggregated and reviewed. |
| Real 3D benchmark results | 45% | E12/E13 are configured and the processing path exists, but current result artifacts are absent. |
| Leakage validation | 50% current checkout | The checker is implemented, but it found no split files locally, so it could not verify current data. Historical session notes report zero violations across ten Tufts split files; rerun this after data restoration. |
| Presentation documentation | 80% | README and project notes exist; this file adds the current audit, teaching notes, and live-demo plan. |

### Honest overall status

**Overall engineering progress: approximately 85%.**
**Progress toward a publishable real 2D-vs-3D result: approximately 70%.**
**Presentation-ready real-data 2D result from this exact checkout: yes.**

The remaining gap is now the real-data 3D/E12/E13 comparison and any optional robustness or timing extensions. The 2D-only E11 benchmark is complete and has a reproducible 40-run score table.

## 5.1 Completed Tufts 2D benchmark

E11 contains 40 valid metric files: four methods × two protocols × five seeds.
The table below reports mean ± standard deviation over the five seeds.

| Method | P1 Rank-1 | P1 EER | P1 AUC | P2 Rank-1 | P2 EER | P2 AUC |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 90.23% ± 0.00 | 4.61% ± 0.03 | 0.986 | 90.08% ± 2.18 | 6.78% ± 0.35 | 0.979 |
| LBP | 90.91% ± 0.00 | 6.14% ± 0.05 | 0.982 | 91.97% ± 1.68 | 6.08% ± 0.56 | 0.982 |
| HOG | 96.14% ± 0.00 | 4.09% ± 0.00 | 0.986 | 96.52% ± 1.08 | 3.84% ± 0.45 | 0.986 |
| Gabor | 96.59% ± 0.00 | 2.18% ± 0.00 | 0.993 | 94.77% ± 1.45 | 5.20% ± 0.57 | 0.984 |

P1 uses 110 gallery identities, so chance Rank-1 is approximately 0.91%.
P2 uses 66 evaluation identities, so chance Rank-1 is approximately 1.52%.
All methods are far above chance. HOG is strongest on P2 Rank-1, while Gabor
has the strongest P1 Rank-1 and lowest P1 EER. These are real Tufts 2D results,
not toy metrics.

## 5.2 What was required to finish the 2D benchmark

For the 2D-only presentation, the minimum defensible evidence is:

1. Restore or download the Tufts data.
2. Run ingestion, 2D preprocessing, and both P1/P2 split protocols for seeds 0–4.
3. Run **E11** for PCA, LBP, HOG, and Gabor.
4. Confirm that E11 creates metrics for every configured arm, protocol, and seed.
5. Run the leakage audit and record the number of checked split files and violations.
6. Aggregate Rank-1, Rank-5, EER, AUC, CMC/ROC information, and timing.
7. Inspect the score table for impossible results, especially 100% P2 performance or missing arms.
8. Regenerate `results/RESULTS.md` and label synthetic, Yale, and Tufts results separately.

All items above are now complete for E11. The current final claim is specifically
that the Tufts 2D benchmark is complete; E12/E13 remain separate 3D and
2D-vs-3D experiments.

## 6. Recommended live demonstration

### A. Start with the project map

Show the repository and explain the flow:

```text
dataset -> manifest -> preprocessing -> splits -> features -> matcher -> metrics -> aggregation
```

Point out the experiment configuration rather than opening implementation files first:

```text
configs/experiments/E11.yaml
```

Explain that E11 is the 2D-only Tufts experiment and contains four feature arms, two protocols, and five seeds.

### B. Show the architecture and tests

Run:

```bash
make test
```

If the coverage plugin fails because of the local machine’s protected temporary location, run the functional tests without coverage:

```bash
./.venv/bin/python -m pytest -q -o addopts='' -p no:cov
```

State clearly that tests validate implementation behavior; benchmark metrics still require the dataset and completed runs.

### C. Run the 2D benchmark when Tufts data is available

```bash
make tufts-ingest
make tufts-preprocess
make tufts-splits
./.venv/bin/ivafr run --exp E11 --data-root data --results-root results
./.venv/bin/ivafr aggregate --results-root results --out results --preamble docs/RESULTS_PREAMBLE.md
./.venv/bin/python scripts/verify_no_leakage.py --data-root data
```

For a short presentation, do not wait for every experiment if the full run is long. Use already-generated `metrics.json` files and show the aggregate table. If no metrics exist, demonstrate the toy or a small configured run and label it as a pipeline demonstration, not as the real benchmark result.

### D. What to show on screen

Show these items in order:

1. `configs/experiments/E11.yaml` — the declared 2D comparison.
2. A sample manifest row — subject ID, sample ID, modality, and source path.
3. An aligned/preprocessed face image — evidence that the input is normalized consistently.
4. One feature representation — for example an Eigenface/PCA vector or HOG visualization.
5. A `metrics.json` file — one run’s machine-readable result.
6. The aggregate table — comparison across methods, protocols, and seeds.
7. The leakage-audit output — evidence that gallery/probe and train/evaluation identities are separated as designed.

### E. Suggested 5-minute explanation

“We are comparing four classical 2D facial descriptors under the same data split and matching procedure. PCA captures global appearance, LBP captures local texture, HOG captures edge structure, and Gabor captures multi-scale oriented responses. P1 measures closed-set recognition, while P2 measures transfer to identities excluded from representation fitting. We report Rank-1 for identification and EER/AUC for verification, because Rank-1 alone can be misleading when gallery size changes. The implementation is config-driven and reproducible across five seeds. The key integrity check is that no gallery/probe or train/evaluation identity leakage is allowed.”

## 7. How to understand the scores

- **Rank-1:** fraction of probes whose top-ranked gallery identity is correct.
- **Rank-5:** fraction where the correct identity appears among the five highest-ranked candidates.
- **EER:** operating point where false accepts and false rejects are approximately equal. Lower is better.
- **AUC:** area under the ROC curve. 0.5 is chance-level verification; higher is better.
- **Timing:** measures computational cost per stage or per sample.
- **Chance-normalized Rank-1:** raw Rank-1 divided by the expected chance rate. This prevents a smaller gallery from looking better solely because there are fewer candidates.

Never compare one method’s raw Rank-1 to another method’s Rank-1 if they used different gallery sizes or protocols without explaining the difference.

## 8. Likely questions and concise answers

**Why compare 2D and 3D?** 2D is inexpensive and widely available but strongly affected by illumination and appearance. 3D aims to represent geometry and can reduce sensitivity to lighting, although it requires more specialized data and processing.

**Why use several 2D features?** They represent different information: global appearance, local texture, edge geometry, and oriented frequency structure. This makes the comparison more informative than testing several nearly identical classifiers.

**Why is P2 important?** A method can memorize the subjects used during fitting and still fail on unseen identities. P2 tests whether the representation transfers beyond the training identities.

**Why five seeds?** A single split can be lucky or unlucky. Multiple seeds show whether the reported behavior is stable.

**How do you prevent leakage?** Splits are generated explicitly, PCA and other learned reductions are fit only on the permitted training partition, and `verify_no_leakage.py` checks gallery/probe and P2 subject separation.

**What is the main limitation?** The current checkout does not contain the real Tufts data or final Tufts metrics. The implementation is present, but the final scientific claim must wait for a successful E11 run and aggregation.

## 9. Handoff checklist for tomorrow

- [ ] Confirm the data directory exists and contains the intended dataset.
- [ ] Confirm `find results -name metrics.json` returns the expected E11 files.
- [ ] Run the leakage audit and save its output.
- [ ] Regenerate the aggregate results.
- [ ] Replace any historical or stale tables in the presentation with current tables.
- [ ] Keep the Yale checkpoint values labeled as Yale checkpoints.
- [ ] Keep toy values labeled synthetic.
- [ ] If Tufts cannot be run, state the limitation explicitly and present the working pipeline rather than implying a completed real-data result.
