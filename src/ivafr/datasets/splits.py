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
from ivafr.integrity import atomic_json, manifest_fingerprint
from ivafr.datasets.eligibility import assert_eligible, eligible_samples

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
        for _, group in (
            df.assign(_illum_order=order).sort_values("_illum_order").groupby("subject_id")
        ):
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
    modality: str | None = None,
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
    if not 0 < p2_train_frac < 1 or imp_ratio < 1:
        raise ValueError("Invalid train fraction or impostor ratio")
    assert_eligible(manifest)
    strict = gallery_strict or {"illumination": ["normal"]}
    all_ids = manifest["sample_id"].astype(str).to_numpy()
    subject_of = dict(zip(manifest["sample_id"].astype(str), manifest["subject_id"]))
    subjects = sorted(manifest["subject_id"].unique())
    frontal = _frontal_neutral(manifest, strict)

    split: dict[str, Any] = {
        "modality": modality,
        "protocol": protocol,
        "seed": seed,
        "train_subjects": [],
        "eval_subjects": [],
        "gallery_ids": [],
        "probe_ids": [],
        "gallery_selection": (
            "normal_frontal" if protocol == "P1_closed" else "normal_frontal_eval_subjects"
        ),
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

    expected = set(split["eval_subjects"])
    if {subject_of[i] for i in split["gallery_ids"]} != expected or {
        subject_of[i] for i in split["probe_ids"]
    } != expected:
        raise ValueError(
            "Every evaluation subject needs a canonical gallery and an independent probe"
        )
    split["train_ids"] = (
        list(split["gallery_ids"])
        if protocol == "P1_closed"
        else manifest.loc[manifest.subject_id.isin(split["train_subjects"]), "sample_id"]
        .astype(str)
        .tolist()
    )
    split["manifest_hash"] = manifest_fingerprint(manifest)
    split["schema_version"] = 2

    split["verification"] = _verify_pairs(pool, subject_of, rng, ratio=imp_ratio)
    assert_no_leakage(split, manifest)
    return split


def assert_no_leakage(split: dict[str, Any], manifest: pd.DataFrame | None = None) -> None:
    """Raise AssertionError if any leakage rule is violated."""
    if split["protocol"] not in {"P1_closed", "P2_disjoint"}:
        raise AssertionError("Unknown protocol")
    if set(split["verification"]) != {"genuine", "impostor"}:
        raise AssertionError("Unknown verification pair type")
    train_sub = set(split["train_subjects"])
    eval_sub = set(split["eval_subjects"])
    if split["protocol"] == "P2_disjoint":
        if train_sub & eval_sub:
            raise AssertionError("P2: train and eval subjects overlap")
        if not train_sub or not eval_sub:
            raise AssertionError("P2: empty train or eval pool")

    gallery = set(split["gallery_ids"])
    probe = set(split["probe_ids"])
    if not gallery or not probe:
        raise AssertionError("Empty gallery or probe")
    for key in ("gallery_ids", "probe_ids", "train_ids"):
        if key in split and len(set(split[key])) != len(split[key]):
            raise AssertionError(f"Duplicate IDs in {key}")
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
    if manifest is not None:
        if split.get("manifest_hash") != manifest_fingerprint(manifest):
            raise AssertionError("Stale split: manifest fingerprint differs; regenerate splits")
        subject_of = dict(zip(manifest.sample_id.astype(str), manifest.subject_id.astype(str)))
        if train_sub | eval_sub != set(subject_of.values()):
            raise AssertionError("Declared subjects do not cover the eligible cohort")
        train = set(split.get("train_ids", []))
        if not train or not (train | gallery | probe) <= set(subject_of):
            raise AssertionError("Missing training pool or unknown sample IDs")
        if {subject_of[i] for i in gallery} != eval_sub or {
            subject_of[i] for i in probe
        } != eval_sub:
            raise AssertionError("Evaluation IDs do not match declared subjects")
        if split["protocol"] == "P1_closed":
            if train != gallery or train_sub != eval_sub:
                raise AssertionError("P1 fitting pool must equal gallery")
        else:
            expected_train = {i for i, s in subject_of.items() if s in train_sub}
            if train != expected_train or train & (gallery | probe):
                raise AssertionError("P2 fitting IDs cross subject boundary")
        if gallery | probe != {i for i, s in subject_of.items() if s in eval_sub}:
            raise AssertionError("Evaluation IDs do not cover the eligible samples")
        for kind, pairs in split["verification"].items():
            if not pairs:
                raise AssertionError(f"No {kind} verification pairs")
            seen = set()
            for a, b in pairs:
                if a not in gallery | probe or b not in gallery | probe:
                    raise AssertionError("Verification ID outside evaluation pool")
                if (subject_of[a] == subject_of[b]) != (kind == "genuine"):
                    raise AssertionError(f"Incorrect {kind} pair labels")
                pair = tuple(sorted((a, b)))
                if pair in seen:
                    raise AssertionError("Duplicate verification pair")
                seen.add(pair)
        content_columns = (
            [f"content_hash_{split['modality']}"]
            if split.get("modality")
            else ["content_hash_2d", "content_hash_3d"]
        )
        columns = ["capture_id"] + [c for c in content_columns if c in manifest]
        for col in columns:
            if col not in manifest:
                continue
            by_id = dict(zip(manifest.sample_id.astype(str), manifest[col].astype(str)))

            def values(ids):
                return {by_id[i] for i in ids if by_id[i] not in {"", "NA", "nan"}}

            if values(gallery) & values(probe):
                raise AssertionError(f"Shared source content across gallery/probe: {col}")
            if split["protocol"] == "P2_disjoint" and values(train) & values(gallery | probe):
                raise AssertionError(f"Shared source content across fitting/evaluation: {col}")


def write_split(split: dict[str, Any], out_dir: str | Path) -> Path:
    """Persist a split to ``<out_dir>/<protocol>_seed<seed>.json``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{split['protocol']}_seed{split['seed']}.json"
    atomic_json(path, split)
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


def prepare_splits(
    manifest: pd.DataFrame, out_dir: Path, modalities, protocols, seeds
) -> list[Path]:
    paths = []
    prepared = []
    for modality in modalities:
        frame = eligible_samples(manifest, modality)
        assert_eligible(frame)
        for protocol in protocols:
            for seed in seeds:
                split = make_split(frame, protocol, seed, modality=modality)
                prepared.append((split, out_dir / modality))
    for split, directory in prepared:
        paths.append(write_split(split, directory))
    return paths


def load_experiment_split(
    data_root: Path, dataset: str, modality: str, protocol: str, seed: int, frame: pd.DataFrame
) -> dict:
    assert_eligible(frame)
    path = data_root / "processed" / dataset / "splits" / modality / f"{protocol}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved split missing: {path}; run ivafr splits --modality {modality}"
        )
    split = read_split(path)
    if (split.get("modality"), split["protocol"], split["seed"]) != (modality, protocol, seed):
        raise ValueError(f"Split metadata disagrees with path: {path}")
    assert_no_leakage(split, frame)
    return split
