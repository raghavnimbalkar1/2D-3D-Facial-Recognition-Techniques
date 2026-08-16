# Protocols

`P1_closed` uses frontal, neutral, normal-light samples as each subject's
gallery and the remaining captures as probes. For the cropped Yale B archive,
the native `A+000E+00` capture is mapped to `normal`; if a source has no
semantic normal label, the first deterministic native-light capture per
subject is used and recorded in the split metadata. `P2_disjoint` assigns 40% of
subjects to a background training pool and 60% to evaluation; no evaluation
subject may enter fitting, validation, or threshold selection.

All seeds, splits, feature fitting, score direction, and verification pairs
are deterministic and traceable to the resolved experiment config.
