# Datasets

## Toy

The toy adapter generates deterministic ellipsoid-plus-bump faces with a
Lambertian image, synthetic depth observation, and sidecar landmarks. Depth
contains deterministic 0.05-unit simulated sensor noise per observation so
lighting variants do not share identical depth files. It is the CI and
development dataset, not evidence about independently acquired real faces.

## Extended Yale Face Database B

Yale B is supported through `ivafr.datasets.yaleb`. Download the cropped
frontal subset and extract it to `data/raw/yaleb`. The adapter accepts `.pgm`,
`.png`, `.jpg`, `.jpeg`, and `.bmp` files, records native light-direction
identifiers, excludes the extra ambient capture, and never copies raw images
into `results/`.

The real Yale B 2D path uses the OpenCV ResNet-10 DNN detector. Supply its
model files through `IVAFR_FACE_DNN_PROTO` and `IVAFR_FACE_DNN_MODEL`; model
weights are not committed to this repository. The adapter rejects a mirror
whose files are pixel-identical duplicates.

Yale B pseudo-3D is blocked in the current execution environment. MediaPipe
remains available for the toy methodology path, but no real Yale B pseudo-3D
result is reported this sprint. Any synthetic 3D result must be labelled
`synthetic_toy`.

## Tufts Face Database (TD_3D + TD_RGB_E)

Decision (2026-08-18): the 3D data source for the real-data phase is Tufts.

- **TD_3D** — one SfM-reconstructed mesh per participant, stored as PLY
  (`TD_3D_<n>.ply`). Captured with a quad camera rig moved through 9
  equidistant positions on a semi-circle; meshes are built with open-source
  structure-from-motion, NOT a depth sensor. Units are arbitrary SfM scale,
  so the pipeline must normalise scale (e.g. by inter-ocular distance or
  bounding-box height) before range-image resampling, and results must be
  labelled `modality: sensor3d-reconstructed` (or similar) — a distinct
  comparison axis versus the toy depth maps.
- 113 participants originally; participant #47 withdrew (readme.txt), the
  published database is 112. 74F / 38M, 15+ countries, ages 4-70.
- **TD_RGB_E** — per participant 5 expressions (neutral, smile, eyes closed,
  exaggerated shocked, sunglasses) with a Nikon D3100; same protocol as
  TD_IR_E (thermal) and TD_CS (sketch), giving a ready expression/occlusion
  axis (sunglasses) and cross-modality pairings (RGB ↔ IR ↔ sketch ↔ 3D).
- **Direct download, no licence form**: `http://tdface.ece.tufts.edu/downloads/`
  (use https; listing is open). Sizes: TD_3D ~574 MB (Set1-4),
  TD_RGB_E ~1.9 GB (Set1-4). Fetch script: `scripts/fetch_tufts.sh`.
- **Terms**: non-commercial research/educational use only; redistribution to
  third parties prohibited; publications must cite the TD-FD paper
  (Panetta et al., TPAMI 2018) and the website. See `docs/ETHICS.md`.
- **Implemented adapter** (`ivafr.datasets.tufts3d`): discovers ASCII PLY
  meshes and photos by participant number, canonicalizes IDs to `Sxxx`,
  validates vertex properties and records actual point counts. No `plyfile`
  extra is required. Photo-only input is supported; if meshes are present,
  the historical mesh-matched photo cohort is retained. `ingestion.json`
  records the selected cohort and excluded photo subjects. Source content
  hashes and per-modality rejection reasons are recorded in the manifest.
- **Current protocol gate:** one mesh per subject is insufficient for
  independent gallery/probe 3D recognition. E12/E13 are blocked by design,
  even when downloading and projection succeed. Real mesh orientation and
  metric scale have not been certified. See [current status](IMPLEMENTATION_STATUS.md).
