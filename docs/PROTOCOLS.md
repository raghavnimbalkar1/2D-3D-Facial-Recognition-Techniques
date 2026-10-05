# Protocols

`P1_closed` uses frontal, neutral, normal-light samples as each subject's
gallery and the remaining captures as probes. For the cropped Yale B archive,
the native `A+000E+00` capture is mapped to `normal`; if a source has no
semantic normal label, the first deterministic native-light capture per
subject is used and recorded in the split metadata. `P2_disjoint` assigns 40% of
subjects to a background training pool and 60% to evaluation; no evaluation
subject may enter representation fitting or deployment threshold calibration.
The reported verification thresholds describe evaluation-set ROC operating
points; they are not deployment calibration. They use the `score >= threshold`
rule and accept whole tied-score blocks only if the requested empirical FAR
budget is respected. Reports include achieved FAR, pair-count resolution and
a warning flag when the requested FAR is below that resolution.

All seeds, splits, feature fitting, score direction, and verification pairs
are deterministic and traceable to the resolved experiment config.

Preprocess before generating modality-specific splits. Every accepted subject
must have a canonical gallery and at least one independent probe capture in
that modality. Split files under `splits/2d` or `splits/3d` freeze the eligible
manifest fingerprint and exact fitting IDs. The runner audits and consumes
these files rather than regenerating them silently. A changed manifest
requires new splits. Shared content/capture IDs across gallery/probe or P2
fitting/evaluation are rejected.

P1's canonical gallery is fixed across seeds. Seed variation affects sampled
verification pairs, not independent P1 identification splits. P2 seeds also
change subject partitions. EER intervals use a subject-cluster bootstrap with
shared sampled subject weights for genuine and impostor pairs.

Tufts TD_3D has one mesh per person, so it cannot supply this 3D recognition
protocol. E12/E13 fail eligibility until an appropriate repeated-capture
source and adapter are available. Reprojections of a single mesh are not
independent acquisitions. Cross-modality comparisons additionally require
matching subject/condition capture counts and subject partitions.
