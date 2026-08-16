# Datasets

## Toy

The toy adapter generates deterministic ellipsoid-plus-bump faces with a
Lambertian image, exact depth map, and sidecar landmarks. It is the CI and
development dataset.

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
