# Decisions

- This sprint compares 2D texture with pseudo-3D monocular geometry.
- Real 3D sensors, point-cloud IO, ICP, Torch, depth CNNs, and a Streamlit app
  are deferred until licensed Tier-A data is available.
- Verification scores are similarities: larger means more likely same person.
- EER uses linear interpolation at the FAR/FRR crossing.

## 2026-08-14 — Yale B pseudo-3D blocked in the current environment

The real-data MediaPipe path was tested against an actual Yale B image using
three configurations:

- MediaPipe 0.10.21 legacy FaceMesh failed while creating the native GPU
  service with `Could not create an NSOpenGLPixelFormat`.
- MediaPipe 0.10.21 Tasks FaceLandmarker with an explicit CPU delegate failed
  during native graph initialization with the same `kGpuService` failure.
- MediaPipe 0.10.9 was tested separately in an isolated CPython 3.11
  environment using the official macOS wheel and failed with the same native
  graph error.

Decision: real-data pseudo-3D preprocessing is environment-blocked for this
sprint. We will not continue version hunting or add a weaker replacement
reconstruction pipeline. The toy dataset remains the validation path for the
pseudo-3D methodology. Yale B work may continue for the 2D benchmark, with
the limitation reported explicitly in the results.

## 2026-08-17 — Yale B CLAHE cache invalidation bug corrected

The first real Yale B 2D checkpoint was invalid. The CLAHE, histogram
equalization, and Tan–Triggs conversion path cast normalized floats to
`uint8` before multiplying by 255, collapsing most pixels to zero. The
corrupted crops passed the existing shape and dtype tests and caused PCA to
fit only three components with same- and different-subject similarities near
1.0.

The conversion now rounds after scaling, the Yale 2D crop cache was deleted
and rebuilt, and a regression test requires meaningful pixel spread. The
corrected single-seed PCA checks produce Rank-1 39.56% on P1 and 48.82% on P2,
with condition-level performance separating moderate from extreme lighting.
The earlier E05/E08/E10 checkpoint numbers are superseded and must not be
used.

## 2026-08-18 — Gabor descriptor reworked to spatial maps + train-only PCA

The original Gabor implementation reduced each filter response to a single
scalar (5 frequencies x 8 orientations = 40 values, no spatial structure). On
Yale B P1 seed-0 it scored rank-1 17.21% with genuine/impostor cosine
separation of 0.0031 — effectively collapsed. Replacement: full magnitude
maps per filter, 4x downsample (~10,240 raw dims), PCA to 200 components fit
on the train split only. Validation (P1 seed-0): rank-1 43.43%, separation
0.1974, EER 38.0% (from 46.2%). A regression test asserts that two crops with
identical global brightness but different spatial layouts produce different
raw vectors, so the collapse cannot silently return.

Runtime cost is dominated by the Gabor filter banks (~0.2 s/image, ~58 min
per P1 run); accepted for a one-time benchmark since the batch runs
unattended.

## 2026-08-18 — Re-run idempotency bug fixed

`_new_run_dir` embeds a microsecond timestamp, so the ``metrics.json
is_file()`` skip check always compared a freshly generated path and re-ran
completed work. `_existing_run_dir` now globs the runs tree for the most
recent completed (exp, protocol, seed, arm) dir and skips it. This caused
duplicate run dirs for the runs re-executed before the fix; the dedupe pass
keeps the newest directory per combination.
