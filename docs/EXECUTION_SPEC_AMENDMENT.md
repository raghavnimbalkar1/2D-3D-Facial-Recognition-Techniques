# Execution Spec Amendment

**Date:** 2026-08-14

This amendment records the change in scope after the final isolated
MediaPipe compatibility test. The original execution specification remains
the design reference; these changes govern the current sprint status.

## Section 0 — Scope and objective

The toy dataset remains the complete validation path for the 2D and
pseudo-3D methodology. Yale B remains the real-data source for the 2D
benchmark. Yale B pseudo-3D is environment-blocked and cannot support a
reported real-data comparison in this execution environment.

## Section 2 — Dataset strategy

Yale B ingestion and manifest generation are supported. The planned MediaPipe
FaceMesh reconstruction is not considered operational on the current macOS
runtime after testing both the legacy API and Tasks API, including an explicit
CPU delegate and MediaPipe 0.10.9. No alternative pseudo-3D reconstruction
pipeline will be added during this sprint.

## Section 6 — Pipeline execution

The remaining real-data effort should focus on a rigorous Yale B 2D benchmark,
including subject-aware protocols and the planned illumination and occlusion
sweeps where the 2D path is available. Pseudo-3D, fusion, and comparative
real-data experiments remain blocked until a compatible MediaPipe execution
environment is available. The toy pipeline continues to validate the
pseudo-3D stages in isolation.
