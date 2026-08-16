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
