# Datasets

## Toy

The toy adapter generates deterministic ellipsoid-plus-bump faces with a
Lambertian image, exact depth map, and sidecar landmarks. It is the CI and
development dataset.

## Extended Yale Face Database B

Yale B is supported through `ivafr.datasets.yaleb`. Download it directly from
its authoritative distribution and extract it to `data/raw/yaleb`. The adapter
records native lighting identifiers as manifest conditions and never copies
raw images into `results/`.

Its pseudo-3D path is MediaPipe FaceMesh (`468` landmarks, fixed topology),
not a physical depth sensor. Results must label this modality
`pseudo3d`.
