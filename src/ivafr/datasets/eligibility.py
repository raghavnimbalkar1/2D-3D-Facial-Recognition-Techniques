"""Select usable modality-specific captures and enforce protocol eligibility."""

from __future__ import annotations

import pandas as pd


def eligible_samples(manifest: pd.DataFrame, modality: str) -> pd.DataFrame:
    if modality not in {"2d", "3d"}:
        raise ValueError("modality must be 2d or 3d")
    required = ["detect_ok", "align_ok"] if modality == "2d" else ["nosetip_ok"]
    mask = manifest[f"has_{modality}"].eq(True)
    for key in required:
        mask &= manifest[key].eq(True)
    out = manifest.loc[mask].sort_values("sample_id").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No accepted {modality} captures; preprocess this modality first")
    return out


def assert_eligible(manifest: pd.DataFrame) -> None:
    counts = manifest.groupby("subject_id")[
        "capture_id" if "capture_id" in manifest else "sample_id"
    ].nunique()
    bad = counts[counts < 2].index.tolist()
    if bad:
        raise ValueError(
            f"Independent gallery/probe captures required; fewer than two captures for {bad[:8]}. "
            "The Tufts single-mesh cohort cannot support real 3D recognition. "
            "Augmentations of one capture are not independent captures."
        )
    if len(counts) < 2:
        raise ValueError("At least two subjects required for verification")
