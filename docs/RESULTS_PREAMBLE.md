# Results

## Current status

This repository does not yet contain a publishable real-data 2D-versus-3D
comparison.

The deterministic toy dataset is the current end-to-end validation path for
the pipeline and pseudo-3D methodology. Toy outputs are generated locally by
`make all`; they are synthetic proof-of-concept results and must not be
interpreted as evidence about real-world recognition performance.

Yale B ingestion has been verified on real images: the adapter discovers 2,414
images across 38 subjects. However, real-data pseudo-3D preprocessing is
blocked in the current macOS execution environment. MediaPipe 0.10.21 legacy
FaceMesh, MediaPipe Tasks with an explicit CPU delegate, and an isolated
MediaPipe 0.10.9 installation all fail during native graph initialization
with an OpenGL/GPU service error. Consequently, no Yale B pseudo-3D metrics
are reported here.

## Corrected real-data 2D checkpoint

The initial Yale B checkpoint was invalid: the CLAHE conversion collapsed
most cached crops to near-constant black images. The cache was deleted,
preprocessing was rebuilt, and these corrected results were produced from
single-seed PCA runs. They are not the final five-seed report:

| Experiment | Protocol | Arm | Rank-1 | EER | AUC |
|---|---|---|---:|---:|---:|
| E01 | P1 closed | 2D-PCA | 0.3956 | 0.4058 | 0.6209 |
| E01 | P2 disjoint | 2D-PCA | 0.4882 | 0.4830 | 0.5394 |

P1 is a 38-way identification problem with a 2.63% chance baseline, so the
corrected Rank-1 result is 15.03x chance. P2 uses 15 background subjects and
23 evaluation subjects, giving a 23-way gallery and a 4.35% chance baseline;
its corrected Rank-1 result is 11.23x chance. Aggregated tables now report
both raw Rank-1 and Rank-1 divided by the protocol's chance baseline.

For P1, several moderate-light conditions reached 100% Rank-1, while
conditions such as `A+000E+90` and `A+035E+65` reached 2.63% (1/38). For P2,
the strongest conditions reached 100% Rank-1, while `A+110E+15` reached 0%
and several neighboring extreme-light conditions reached 4.35% (1/23).
The corrected crop cache contains nonzero pixel spread for all 2,414 images.

The earlier E05, E08, and E10 single-seed numbers were generated before this
fix and are superseded; those experiments must be rerun before being used.
All corrected values remain engineering checkpoints until the configured
seeds and arms have been run and reviewed.

## Interpretation and limitations

- Any toy metrics are synthetic development checks, not final scientific
  results.
- The preliminary Yale B 2D checkpoint is not yet the complete five-seed
  benchmark.
- A real Yale B 2D-versus-pseudo-3D comparison remains future work until the
  MediaPipe runtime is available in a compatible execution environment.
- The pseudo-3D modality must be labelled as monocular reconstructed geometry,
  not physical 3D sensor data.

## Reproduction

Use the following commands to regenerate the current toy outputs:

```bash
make setup
make all
```

Generated run artifacts belong under `results/runs/` and are intentionally
excluded from version control.
