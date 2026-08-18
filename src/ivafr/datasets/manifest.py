"""Manifest handling: schema, writing, reading, audit.

``manifest.csv`` is the single source of truth for a dataset once ingested.
Every downstream stage filters on the manifest, never on the filesystem.

Schema (one row = one capture)::

    dataset, subject_id, sample_id, session,
    path_2d, path_3d, path_landmarks,
    expression, pose_yaw, pose_pitch, illumination, occlusion,
    orig_w, orig_h, n_points, has_2d, has_3d,
    detect_ok, align_ok, nosetip_ok, quality_flag, notes, data_modality

Rules: ``subject_id`` is a zero-padded canonical string (``S001``);
``pose_yaw/pose_pitch`` in degrees (0 = frontal); ``illumination`` in
{normal, strong, dark, side, synthetic:<param>}; unknown -> ``NA``, never blank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ivafr.datasets.base import Sample
from ivafr.logging_utils import get_logger

log = get_logger("datasets.manifest")

MANIFEST_COLUMNS = [
    "dataset",
    "subject_id",
    "sample_id",
    "session",
    "path_2d",
    "path_3d",
    "path_landmarks",
    "expression",
    "pose_yaw",
    "pose_pitch",
    "illumination",
    "occlusion",
    "orig_w",
    "orig_h",
    "n_points",
    "has_2d",
    "has_3d",
    "detect_ok",
    "align_ok",
    "nosetip_ok",
    "quality_flag",
    "notes",
    "data_modality",
]

ILLUM_VALUES = {"normal", "strong", "dark", "side"}


def _na(v: object) -> object:
    return "NA" if v is None else v


def _int_na(v: object) -> int | str:
    """Convert optional numeric metadata while preserving the NA sentinel."""
    if v is None or (isinstance(v, str) and v.strip().upper() == "NA"):
        return "NA"
    return int(v)


@dataclass
class ManifestStats:
    """Summary of a manifest for the ingest audit report."""

    n_subjects: int
    n_samples: int
    per_subject_min: int
    per_subject_max: int
    n_missing_2d: int
    n_missing_3d: int
    conditions: pd.DataFrame


def samples_to_manifest(samples: list[Sample]) -> pd.DataFrame:
    """Convert adapter output to a manifest DataFrame (schema enforced)."""
    rows = []
    for s in samples:
        meta = s.meta
        rows.append(
            {
                "dataset": s.dataset,
                "subject_id": s.subject_id,
                "sample_id": s.sample_id,
                "session": _na(meta.get("session")),
                "path_2d": str(s.path_2d) if s.path_2d else "",
                "path_3d": str(s.path_3d) if s.path_3d else "",
                "path_landmarks": str(s.path_landmarks) if s.path_landmarks else "",
                "expression": str(_na(meta.get("expression"))),
                "pose_yaw": float(_na(meta.get("pose_yaw"))),
                "pose_pitch": float(_na(meta.get("pose_pitch"))),
                "illumination": str(_na(meta.get("illumination"))),
                "occlusion": str(_na(meta.get("occlusion"))),
                "orig_w": _int_na(meta.get("orig_w")),
                "orig_h": _int_na(meta.get("orig_h")),
                "n_points": _int_na(meta.get("n_points")),
                "has_2d": bool(s.path_2d is not None),
                "has_3d": bool(s.path_3d is not None),
                "detect_ok": False,
                "align_ok": False,
                "nosetip_ok": False,
                "quality_flag": "",
                "notes": str(_na(meta.get("notes"))),
                "data_modality": "synthetic_toy" if s.dataset == "toy" else "real",
            }
        )
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    validate_manifest(df)
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    """Validate schema invariants; raises ValueError on violation."""
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {missing_cols}")
    if df["sample_id"].isna().any() or (df["sample_id"].astype(str).str.strip() == "").any():
        raise ValueError("sample_id must never be blank")
    bad_ids = df.loc[~df["subject_id"].str.fullmatch(r"S\d{3,}"), "subject_id"].unique()
    if len(bad_ids):
        raise ValueError(f"subject_id must be zero-padded canonical (S001...): {bad_ids}")
    unknown_illum = {
        value
        for value in set(df["illumination"].unique()) - ILLUM_VALUES - {"NA"}
        if not str(value).startswith("yale:") and not str(value).startswith("synthetic:")
    }
    if unknown_illum:
        raise ValueError(f"Unknown illumination values: {unknown_illum}")
    modalities = set(df["data_modality"].astype(str))
    if not modalities <= {"real", "synthetic_toy"}:
        raise ValueError(f"Unknown data_modality values: {modalities - {'real', 'synthetic_toy'}}")
    for col in ("pose_yaw", "pose_pitch", "orig_w", "orig_h", "n_points"):
        nonnum = pd.to_numeric(df[col], errors="coerce")
        is_na = (df[col] == "NA") | df[col].isna() | (df[col].astype(str).str.strip() == "")
        if (nonnum.isna() & ~is_na).any():
            raise ValueError(f"Column {col} contains non-numeric or blank values")


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a validated manifest to CSV."""
    validate_manifest(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Manifest written: %s (%d rows)", path, len(df))


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a manifest CSV."""
    df = pd.read_csv(path, dtype={"subject_id": str, "sample_id": str}, keep_default_na=False)
    # Upgrade manifests written before v2 without changing their row data.
    if "data_modality" not in df.columns:
        df["data_modality"] = np.where(df["dataset"].eq("toy"), "synthetic_toy", "real")
        df = df[[*MANIFEST_COLUMNS]]
    validate_manifest(df)
    return df


def audit(df: pd.DataFrame) -> ManifestStats:
    """Compute the ingest audit summary.

    Fails loudly if any subject has fewer than two samples (genuine pairs
    cannot be formed otherwise).
    """
    per_subject = df.groupby("subject_id").size()
    if per_subject.min() < 2:
        offenders = per_subject[per_subject < 2].index.tolist()
        raise ValueError(
            f"Subjects with <2 samples (cannot form genuine pairs): {offenders}"
        )
    cond = (
        df.groupby(["pose_yaw", "illumination"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return ManifestStats(
        n_subjects=len(per_subject),
        n_samples=len(df),
        per_subject_min=int(per_subject.min()),
        per_subject_max=int(per_subject.max()),
        n_missing_2d=int((~df["has_2d"]).sum()),
        n_missing_3d=int((~df["has_3d"]).sum()),
        conditions=cond,
    )


def audit_report(stats: ManifestStats) -> str:
    """Render the audit as a printable text block."""
    lines = [
        f"Subjects: {stats.n_subjects}  Samples: {stats.n_samples}",
        f"Samples/subject: [{stats.per_subject_min}, {stats.per_subject_max}]",
        f"Missing 2D: {stats.n_missing_2d}  Missing 3D: {stats.n_missing_3d}",
        "",
        "Condition cross-tab (pose_yaw x illumination):",
    ]
    tab = stats.conditions.pivot_table(
        index="pose_yaw", columns="illumination", values="n", fill_value=0, aggfunc="sum"
    )
    lines.append(tab.to_string())
    return "\n".join(lines)
