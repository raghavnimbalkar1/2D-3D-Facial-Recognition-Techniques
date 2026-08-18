# Ethics and data handling

Raw face images are not redistributed by this repository. Users must obtain
Yale B under its own terms and comply with applicable institutional review,
privacy, and research-use requirements. `ivafr ingest --anonymize` hashes
subject identifiers in generated manifests.

The optional ArcFace path is disabled by default. If enabled, document the
downloaded model's research/non-commercial license and do not use it as a
load-bearing dependency for the core 2D-vs-pseudo3D claim.

This sprint processes real Yale B images only for the 2D benchmark. No real
3D sensor or real-data pseudo-3D biometric data is processed. Raw faces,
detector weights, and derived biometric templates must not be committed or
redistributed through this repository.

## Tufts Face Database terms (2026-08-18)

Tufts TD_3D / TD_RGB_E downloads are licensed for non-commercial research and
educational purposes only. Redistribution of the data to third parties is
prohibited (including research associates), profiles must not be published
without consent-irrelevant academic usage, and publications must cite the
TD-FD paper (Panetta et al., "A comprehensive database for benchmarking
imaging systems," IEEE TPAMI 2018) and http://tdface.ece.tufts.edu/. The
repository stores raw data only under the git-ignored `data/` tree and any
montage or visualisation of Tufts faces in `results/` is published only with
consent-appropriate anonymisation (e.g. eye-region blurring via the
`--anonymize` path when montages are committed).
