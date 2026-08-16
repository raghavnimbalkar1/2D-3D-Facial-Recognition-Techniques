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
