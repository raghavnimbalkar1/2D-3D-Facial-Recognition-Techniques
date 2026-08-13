# ivafr

`ivafr` is a config-driven, dependency-light benchmark comparing 2D facial
recognition with pseudo-3D facial geometry. This sprint supports the procedural
toy dataset and an optional Extended Yale Face Database B adapter. Yale B is
not redistributed; place a licensed download under `data/raw/yaleb`.

The 3D axis is explicitly monocular reconstructed geometry: toy samples use
ground-truth depth, while Yale B uses MediaPipe FaceMesh landmarks and fixed
topology. The project intentionally does not include real sensor adapters,
point-cloud IO, ICP, Torch, or a depth CNN in this sprint.

## Reproduce the toy benchmark

```text
make setup
make all
```

The commands generate toy data, ingest and preprocess it, run the smoke
comparison, and aggregate `results/runs/*/metrics.json` into tables and
`results/RESULTS.md`. Direct commands are available through `ivafr`; run
`ivafr --help` for the full path.

Every experiment writes its resolved config, system fingerprint, metrics,
curves, confusion matrices, and verification score arrays under
`results/runs/<UTC>_<experiment>_.../`.
