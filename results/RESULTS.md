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

## Preliminary real-data 2D checkpoint

The current bounded validation pass completed preprocessing on all 2,414
usable Yale B images. These are single-seed checkpoints, not the final
five-seed report:

| Experiment | Protocol | Arm | Rank-1 | EER | AUC |
|---|---|---|---:|---:|---:|
| E01 | P1 closed | 2D-PCA | 0.0375 | 0.4969 | 0.5059 |
| E01 | P1 closed | 2D-LBP | 0.0332 | 0.4792 | 0.5240 |
| E01 | P1 closed | 2D-HOG | 0.0434 | 0.4944 | 0.5110 |
| E05 | P2 disjoint | best-2D | 0.0653 | 0.4794 | 0.5330 |
| E08 | P2 disjoint, clean | best-2D | 0.0653 | 0.4794 | 0.5330 |

The E08 occlusion checkpoint produced Rank-1 values of 0.0625, 0.0618,
0.0625, and 0.0625 for sunglasses-20%, block-30%, block-40%, and block-20%,
respectively. E10 measured 1.898 ms per probe for the configured 2D-PCA
timing path. These values are engineering checkpoints only until all
configured seeds and arms have been run and reviewed.

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
