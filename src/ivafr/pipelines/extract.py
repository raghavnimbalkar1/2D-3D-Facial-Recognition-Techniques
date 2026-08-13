"""Stage 3 — Feature extraction for one (arm, split) combination.

Fitting happens on the TRAIN pool only (hard anti-leakage rule); gallery and
probe are transformed with the fitted extractor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ivafr.logging_utils import get_logger
from ivafr.registry import get_feature

log = get_logger("pipelines.extract")

_MODALITY_DIR = {"2d": "crops2d", "pseudo3d": "crops2d", "3d": "range"}


def _load_arrays(interim: Path, manifest: pd.DataFrame, modality: str, ids: list[str], feature_name: str = "") -> dict[str, np.ndarray]:
    """Load preprocessed arrays for the given sample ids."""
    subdir = _MODALITY_DIR[modality]
    ext = ".npy"
    if modality == "2d":
        prefix = "g64"
    elif feature_name == "normal_hog":
        prefix = "n64"
    elif feature_name == "curv_hist":
        prefix = "c64"
    elif feature_name == "lmk3d":
        subdir = "landmarks3d"
        prefix = "lmk3d"
    else:
        prefix = "r64"
    out: dict[str, np.ndarray] = {}
    id_set = set(ids)
    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        if sid not in id_set:
            continue
        path = interim / subdir / str(row["subject_id"]) / f"{sid}_{prefix}{ext}"
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed array missing: {path} (run `ivafr preprocess` first)")
        out[sid] = np.load(path)
    missing = id_set - set(out)
    if missing:
        raise FileNotFoundError(f"Missing preprocessed samples: {sorted(missing)[:10]}")
    return out


def extract_features(
    feature_name: str,
    feature_params: dict,
    train_ids: list[str],
    gallery_ids: list[str],
    probe_ids: list[str],
    manifest: pd.DataFrame,
    interim: Path,
    modality: str,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Extract features for a split; returns (X_train, X_gallery, X_probe, ...).

    The extractor is FIT ON TRAIN IDS ONLY, then applied to gallery/probe.
    """
    from ivafr.seeding import set_all_seeds

    set_all_seeds(seed)
    arrays = _load_arrays(interim, manifest, modality, list(set(train_ids) | set(gallery_ids) | set(probe_ids)), feature_name)

    cls = get_feature(feature_name)
    extractor = cls(feature_params)

    subj_of = dict(zip(manifest["sample_id"].astype(str), manifest["subject_id"].astype(str)))
    y_train = np.asarray([subj_of[i] for i in train_ids])
    X_train = extractor.fit([arrays[i] for i in train_ids], y_train).transform([arrays[i] for i in train_ids])
    X_gallery = extractor.transform([arrays[i] for i in gallery_ids])
    X_probe = extractor.transform([arrays[i] for i in probe_ids])
    log.info(
        "%s: fit on %d train, dim=%s, gallery=%d probe=%d",
        feature_name,
        len(train_ids),
        extractor.feature_dim(),
        len(gallery_ids),
        len(probe_ids),
    )
    return X_train, X_gallery, X_probe, train_ids, gallery_ids, probe_ids
