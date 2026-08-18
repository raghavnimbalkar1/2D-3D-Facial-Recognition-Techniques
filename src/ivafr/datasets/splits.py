"""Subject-aware data splits and verification pairs.

Protocols:

* **P1_closed** — classic closed-set identification. Per subject, the
  frontal-neutral samples form the gallery/train set; every remaining
  (variation) sample is a probe. Subspaces (PCA/LDA/SVM) are fit on the
  gallery only.
* **P2_disjoint** — subject-disjoint generalisation. Subjects are split into
  a background pool (40%, used to learn subspaces / fusion weights /
  thresholds) and an evaluation pool (60%, gallery + probe). No evaluation
  subject ever appears in training. This is the headline protocol.

Verification pairs are deterministic per seed: all genuine pairs within a
subject (self-pairs excluded); impostor pairs subsampled to
``ratio`` x the genuine count. For P2, pairs never span the background pool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ivafr.logging_utils import get_logger

log = get_logger("datasets.splits")


def _frontal_neutral(df: pd.DataFrame, strict: dict[str, Any]) -> pd.Series:
    yaw = pd.to_numeric(df["pose_yaw"], errors="coerce")
    pitch = pd.to_numeric(df["pose_pitch"], errors="coerce")
    illum = df["illumination"]
    mask = (yaw == 0.0) & (pitch == 0.0)
    if strict.get("illumination") is not None:
        mask &= illum.isin(strict["illumination"])
    if "expression" in df.columns and (df["expression"] == "neutral").any():
        mask &= df["expression"].eq("neutral")

    # The cropped Yale B distribution has no semantic ``normal`` label; its
    # native lighting identifiers are yale:<index>. Use one deterministic
    # canonical capture per subject for P1 instead of silently producing an
    # empty gallery.
    if not mask.any() and df["dataset"].eq("yaleb").all():
        order = pd.to_numeric(illum.astype(str).str.extract(r"yale:(\d+)")[0], errors="coerce")
        mask = pd.Series(False, index=df.index)
        for _, group in df.assign(_illum_order=order).sort_values("_illum_order").groupby("subject_id"):
            mask.loc[group.index[0]] = True
    elif mask.any():
        # Ensure exactly one canonical gallery sample per subject
        first_per_subj = pd.Series(False, index=df.index)
        # Prioritize 2D neutral captures if available
        has_2d_prio = np.where(df.get("has_2d", pd.Series(True, index=df.index)), 0, 1)
        sorted_df = df.assign(_prio=has_2d_prio)
        for _, group in sorted_df.loc[mask].sort_values("_prio").groupby("subject_id"):
            first_per_subj.loc[group.index[0]] = True
        mask = first_per_subj
    return mask


def _verify_pairs(
    sample_ids: np.ndarray,
    subject_of: dict[str, str],
    rng: np.random.Generator,
    ratio: int = 20,
) -> dict[str, list[list[str]]]:
    """Genuine (same-subject, no self) + impostor pairs, deterministic."""
    ids = list(sample_ids)
    subj_of = {i: subject_of[i] for i in ids}
    subjects: dict[str, list[str]] = {}
    for i in ids:
        subjects.setdefault(subj_of[i], []).append(i)

    genuine: list[list[str]] = []
    for members in subjects.values():
        members = sorted(members)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                genuine.append([members[a], members[b]])

    impostor_pool = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if subj_of[ids[a]] != subj_of[ids[b]]:
                impostor_pool.append([ids[a], ids[b]])
    rng.shuffle(impostor_pool)
    impostors = impostor_pool[: len(genuine) * ratio]

    return {"genuine": genuine, "impostor": impostors}


def make_split(
    manifest: pd.DataFrame,
    protocol: str,
    seed: int,
    p2_train_frac: float = 0.4,
    imp_ratio: int = 20,
    gallery_strict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one split dict for a protocol + seed (deterministic).

    Args:
        manifest: validated manifest DataFrame.
        protocol: ``P1_closed`` or ``P2_disjoint``.
        seed: random seed controlling subject assignment and pair sampling.
        p2_train_frac: fraction of subjects in the background pool (P2 only).
        imp_ratio: impostor pairs = ``imp_ratio`` x genuine pairs.
        gallery_strict: extra gallery constraints, e.g. ``{"illumination": ["normal"]}``.
    """
    rng = np.random.default_rng(seed)
    strict = gallery_strict or {"illumination": ["normal"]}
    all_ids = manifest["sample_id"].astype(str).to_numpy()
    subject_of = dict(zip(manifest["sample_id"].astype(str), manifest["subject_id"]))
    subjects = sorted(manifest["subject_id"].unique())
    frontal = _frontal_neutral(manifest, strict)

    split: dict[str, Any] = {
        "protocol": protocol,
        "seed": seed,
        "train_subjects": [],
        "eval_subjects": [],
        "gallery_ids": [],
        "probe_ids": [],
        "gallery_selection": "normal_frontal" if protocol == "P1_closed" else "normal_frontal_eval_subjects",
    }

    if protocol == "P1_closed":
        split["train_subjects"] = subjects
        split["eval_subjects"] = subjects
        split["gallery_ids"] = manifest.loc[frontal, "sample_id"].astype(str).tolist()
        split["probe_ids"] = manifest.loc[~frontal, "sample_id"].astype(str).tolist()
        pool = all_ids
    elif protocol == "P2_disjoint":
        rng.shuffle(subjects)
        n_train = max(1, int(np.floor(p2_train_frac * len(subjects))))
        train_subjects = set(subjects[:n_train])
        eval_subjects = set(subjects[n_train:])
        split["train_subjects"] = sorted(train_subjects)
        split["eval_subjects"] = sorted(eval_subjects)
        is_eval = manifest["subject_id"].isin(eval_subjects)
        split["gallery_ids"] = manifest.loc[is_eval & frontal, "sample_id"].astype(str).tolist()
        split["probe_ids"] = manifest.loc[is_eval & ~frontal, "sample_id"].astype(str).tolist()
        pool = manifest.loc[is_eval, "sample_id"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown protocol {protocol!r}")

    if not split["gallery_ids"] or not split["probe_ids"]:
        raise ValueError(f"Split {protocol} seed {seed}: empty gallery or probe set")

    split["verification"] = _verify_pairs(pool, subject_of, rng, ratio=imp_ratio)
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: dict[str, Any]) -> None:
    """Raise AssertionError if any leakage rule is violated."""
    train_sub = set(split["train_subjects"])
    eval_sub = set(split["eval_subjects"])
    if split["protocol"] == "P2_disjoint":
        if train_sub & eval_sub:
            raise AssertionError("P2: train and eval subjects overlap")
        if not train_sub or not eval_sub:
            raise AssertionError("P2: empty train or eval pool")

    gallery = set(split["gallery_ids"])
    probe = set(split["probe_ids"])
    if gallery & probe:
        raise AssertionError("gallery and probe overlap")
    for pair_type in ("genuine", "impostor"):
        for a, b in split["verification"][pair_type]:
            if a == b:
                raise AssertionError("self-pair in verification")
    if split["protocol"] == "P2_disjoint":
        eval_ids = set(split["gallery_ids"]) | set(split["probe_ids"])
        for pair_type in ("genuine", "impostor"):
            for a, b in split["verification"][pair_type]:
                if a not in eval_ids or b not in eval_ids:
                    raise AssertionError("verification pair spans the background pool")


def write_split(split: dict[str, Any], out_dir: str | Path) -> Path:
    """Persist a split to ``<out_dir>/<protocol>_seed<seed>.json``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{split['protocol']}_seed{split['seed']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2, sort_keys=True)
    return path


def read_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    assert_no_leakage(split)
    return split


def summary(split: dict[str, Any]) -> str:
    """One-line human-readable summary of a split."""
    v = split["verification"]
    return (
        f"{split['protocol']} seed={split['seed']}: "
        f"train_subjects={len(split['train_subjects'])} eval_subjects={len(split['eval_subjects'])} "
        f"gallery={len(split['gallery_ids'])} probe={len(split['probe_ids'])} "
        f"genuine={len(v['genuine'])} impostor={len(v['impostor'])}"
    )
