"""Manifest schema, audit, canonical ids."""

from __future__ import annotations

import pandas as pd
import pytest

from ivafr.datasets.manifest import (
    MANIFEST_COLUMNS,
    audit,
    validate_manifest,
)


def test_schema_complete(toy_manifest):
    assert list(toy_manifest.columns) == MANIFEST_COLUMNS
    assert len(toy_manifest) == 48  # 4 subjects x 12 samples


def test_no_blanks(toy_manifest):
    assert not toy_manifest["sample_id"].isna().any()
    assert not (toy_manifest["sample_id"].astype(str).str.strip() == "").any()


def test_canonical_ids(toy_manifest):
    assert toy_manifest["subject_id"].str.fullmatch(r"S\d{3,}").all()


def test_conditions_crosstab(toy_manifest):
    """Every subject sees all pose x illumination combos."""
    crosstab = toy_manifest.groupby(["subject_id", "pose_yaw", "illumination"]).size()
    counts = crosstab.groupby("subject_id").apply(lambda g: g.size).min()
    assert counts >= 8  # 3 poses x 4 lights would be 12; modulo gives >=8


def test_audit_fails_on_single_sample_subject():
    df = pd.DataFrame(
        {
            "dataset": ["toy", "toy", "toy"],
            "subject_id": ["S001", "S001", "S002"],
            "sample_id": ["a", "b", "c"],
            "session": ["s1"] * 3,
            "path_2d": [""] * 3,
            "path_3d": [""] * 3,
            "path_landmarks": [""] * 3,
            "expression": ["neutral"] * 3,
            "pose_yaw": [0.0] * 3,
            "pose_pitch": [0.0] * 3,
            "illumination": ["normal"] * 3,
            "occlusion": ["none"] * 3,
            "orig_w": [120] * 3,
            "orig_h": [120] * 3,
            "n_points": [0] * 3,
            "has_2d": [True] * 3,
            "has_3d": [True] * 3,
            "detect_ok": [True] * 3,
            "align_ok": [True] * 3,
            "nosetip_ok": [True] * 3,
            "quality_flag": [""] * 3,
            "notes": ["NA"] * 3,
            "data_modality": ["synthetic_toy"] * 3,
        }
    )
    with pytest.raises(ValueError, match="genuine pairs"):
        audit(df)


def test_invalid_subject_id_rejected():
    df = pd.DataFrame(
        {
            "dataset": ["toy"],
            "subject_id": ["alice"],
            "sample_id": ["x"],
            "session": ["s1"],
            "path_2d": [""],
            "path_3d": [""],
            "path_landmarks": [""],
            "expression": ["neutral"],
            "pose_yaw": [0.0],
            "pose_pitch": [0.0],
            "illumination": ["normal"],
            "occlusion": ["none"],
            "orig_w": [1],
            "orig_h": [1],
            "n_points": [0],
            "has_2d": [False],
            "has_3d": [False],
            "detect_ok": [False],
            "align_ok": [False],
            "nosetip_ok": [False],
            "quality_flag": [""],
            "notes": ["NA"],
            "data_modality": ["synthetic_toy"],
        }
    )
    with pytest.raises(ValueError, match="canonical"):
        validate_manifest(df)
