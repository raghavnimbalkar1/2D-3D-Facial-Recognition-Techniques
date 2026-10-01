"""Dataset/cache provenance and verifiable run completion."""

from __future__ import annotations

import json
import platform
import math
from pathlib import Path

from ivafr.datasets.eligibility import eligible_samples, assert_eligible
from ivafr.datasets.manifest import read_manifest
from ivafr.integrity import atomic_json, digest, file_hash, source_fingerprint, dependency_versions
from ivafr.preprocess import cache

ARM_MODALITY = {
    "pca": "2d",
    "lbp": "2d",
    "hog": "2d",
    "gabor": "2d",
    "depth_pca": "3d",
    "depth_lbp": "3d",
    "normal_hog": "3d",
    "curv_hist": "3d",
    "lmk3d": "3d",
}


def experiment_inputs(exp, data_root: Path) -> dict:
    from ivafr.pipelines.preprocess_run import preprocessing_key

    manifest = read_manifest(data_root / "processed" / exp.dataset / "manifest.csv")
    frames, files = {}, {}
    for modality in sorted({ARM_MODALITY[a.feature] for a in exp.arms}):
        frame = eligible_samples(manifest, modality)
        assert_eligible(frame)
        cfg = exp.preprocess_2d if modality == "2d" else exp.preprocess_3d
        cfg_key = preprocessing_key(modality, cfg)
        expected = cache.cfg_hash(cfg_key)
        if (
            f"preprocess_{modality}_hash" not in frame
            or not frame[f"preprocess_{modality}_hash"].eq(expected).all()
        ):
            raise ValueError(f"Stale {modality} preprocessing configuration; preprocess again")
        for _, row in frame.iterrows():
            source = Path(row[f"path_{modality}"])
            source_hash = file_hash(source)
            if source_hash != row.get(f"content_hash_{modality}"):
                raise ValueError(f"Raw source changed: {source}; ingest/preprocess again")
            lmk = row.get("path_landmarks", "")
            input_hash = digest([source_hash, file_hash(lmk) if lmk else ""])
            directory = (
                data_root
                / "interim"
                / exp.dataset
                / ("crops2d" if modality == "2d" else "range")
                / row.subject_id
            )
            suffixes = (
                ["a112.png", "g64.npy"]
                if modality == "2d"
                else ["r64.npy", "n64.npy", "c64.npy", "valid64.npy"]
            )
            for suffix in suffixes:
                path = directory / f"{row.sample_id}_{suffix}"
                if not cache.is_cached(path, cfg_key, input_hash):
                    raise ValueError(f"Incomplete or corrupt preprocessing bundle: {path}")
                files[path.relative_to(data_root).as_posix()] = file_hash(path)
            if any(a.feature == "lmk3d" for a in exp.arms) and modality == "3d":
                path = (
                    data_root
                    / "interim"
                    / exp.dataset
                    / "landmarks3d"
                    / row.subject_id
                    / f"{row.sample_id}_lmk3d.npy"
                )
                if not cache.is_cached(path, cfg_key, input_hash):
                    raise ValueError(f"Missing/corrupt landmarks: {path}")
                files[path.relative_to(data_root).as_posix()] = file_hash(path)
        frames[modality] = frame
    if len(frames) > 1:
        # Exact cohort/condition matching; do not silently shrink one arm's pool.
        signatures = []
        for frame in frames.values():
            columns = [
                c
                for c in (
                    "subject_id",
                    "expression",
                    "occlusion",
                    "pose_yaw",
                    "pose_pitch",
                    "illumination",
                )
                if c in frame
            ]
            signatures.append(frame[columns].astype(str).value_counts().sort_index().to_dict())
        if any(s != signatures[0] for s in signatures[1:]):
            raise ValueError(
                "Multimodal comparisons require matching subject/condition capture counts"
            )
    modalities = set(manifest.data_modality.astype(str))
    if len(modalities) != 1:
        raise ValueError("Cannot mix real and synthetic samples")
    return {
        "frames": frames,
        "files": files,
        "source_hash": source_fingerprint(),
        "dependencies": dependency_versions(),
        "python": platform.python_version(),
        "data_modality": next(iter(modalities)),
    }


def run_identity(exp, arm, split, inputs: dict) -> dict:
    return {
        "schema_version": 2,
        "config_hash": digest(exp.to_dict()),
        "arm": arm.key,
        "protocol": split["protocol"],
        "seed": split["seed"],
        "split_hash": digest(split),
        "manifest_hash": split["manifest_hash"],
        "input_hash": digest(inputs["files"]),
        "source_hash": inputs["source_hash"],
        "dependencies": inputs["dependencies"],
        "python": inputs["python"],
    }


def complete_run(directory: Path, run_id: str) -> None:
    artifacts = {
        p.relative_to(directory).as_posix(): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "completed.json"
    }
    atomic_json(directory / "completed.json", {"run_id": run_id, "artifacts": artifacts})


def validate_run(directory: Path) -> dict:
    marker = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    required = {"metrics.json", "config_resolved.json", "sysinfo.json", "split.json", "inputs.json"}
    if not required <= set(marker["artifacts"]):
        raise ValueError(f"Missing required artifacts: {directory}")
    for rel, expected in marker["artifacts"].items():
        path = (directory / rel).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or not path.is_file()
            or file_hash(path) != expected
        ):
            raise ValueError(f"Artifact integrity failure: {directory}/{rel}")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics["run_id"] != marker["run_id"] or digest(metrics["provenance"]) != metrics["run_id"]:
        raise ValueError("Run identity mismatch")
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if digest(split) != metrics["provenance"]["split_hash"]:
        raise ValueError("Consumed split does not match run identity")
    config = json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
    if digest(config) != metrics["provenance"]["config_hash"]:
        raise ValueError("Resolved config does not match run identity")
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    if digest(inputs) != metrics["provenance"]["input_hash"]:
        raise ValueError("Input bundle does not match run identity")
    if (
        metrics["exp_id"] != config["id"]
        or metrics["dataset"]["name"] != config["dataset"]
        or metrics["seed"] not in config["seeds"]
        or metrics["protocol"] not in config["protocols"]
        or metrics["arm"] not in {a["key"] for a in config["arms"]}
    ):
        raise ValueError("Metrics disagree with experiment configuration")
    if (metrics["protocol"], metrics["seed"]) != (split["protocol"], split["seed"]):
        raise ValueError("Metric identity disagrees with split")
    if (metrics["arm"], metrics["protocol"], metrics["seed"]) != tuple(
        metrics["provenance"][k] for k in ("arm", "protocol", "seed")
    ):
        raise ValueError("Metric identity disagrees with provenance")
    arm = next(a for a in config["arms"] if a["key"] == metrics["arm"])
    for component in ("feature", "matcher"):
        if metrics[component] != {"name": arm[component], "params": arm[f"{component}_params"]}:
            raise ValueError(f"{component} disagrees with resolved configuration")
    for enabled, section in (
        ("evaluate_identification", "identification"),
        ("evaluate_verification", "verification"),
        ("evaluate_timing", "timing"),
    ):
        if config.get(enabled, False) and section not in metrics:
            raise ValueError(f"Missing enabled metric section: {section}")
    for section, fields in (
        ("identification", ("rank1", "rank5", "accuracy")),
        ("verification", ("eer", "auc")),
    ):
        if section in metrics:
            for field in fields:
                value = metrics[section][field]
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {section}.{field}")
    artifacts = set(marker["artifacts"])
    if "identification" in metrics:
        ident = metrics["identification"]
        if ident["n_gallery"] != len(split["gallery_ids"]) or ident["n_probe"] != len(
            split["probe_ids"]
        ):
            raise ValueError("Identification sample counts disagree with split")
        expected_artifacts = {"tables/cm.csv", "tables/per_class.csv", "figures/fig_cmc_all.png"}
        if not expected_artifacts <= artifacts:
            raise ValueError("Missing identification artifacts")
    if "verification" in metrics:
        import numpy as np

        for kind in ("genuine", "impostor"):
            path = f"{kind}_scores.npy"
            if path not in artifacts:
                raise ValueError(f"Missing score artifact: {path}")
            scores = np.load(directory / path, allow_pickle=False)
            count = len(split["verification"][kind])
            if (
                scores.shape != (count,)
                or not np.isfinite(scores).all()
                or metrics["verification"][f"n_{kind}"] != count
            ):
                raise ValueError("Verification scores disagree with split")
    if "timing" in metrics:
        timing = metrics["timing"]
        if not math.isfinite(timing["ms_per_probe"]) or timing["ms_per_probe"] < 0:
            raise ValueError("Invalid probe timing")
        for stage in ("fit", "transform", "match"):
            if (
                timing[stage]["repeats"] != config["timing_repeats"]
                or timing[stage]["warmups"] != config["timing_warmups"]
            ):
                raise ValueError("Timing repeat counts disagree with configuration")
    conditions = config.get("robustness", {}).get("conditions", [])
    expected_conditions = {c["name"] for c in conditions}
    if set(metrics.get("robustness", {}).get("conditions", {})) != expected_conditions:
        raise ValueError("Robustness conditions disagree with configuration")
    for name in expected_conditions:
        rel = f"conditions/{name}/metrics.json"
        if rel not in artifacts:
            raise ValueError(f"Missing robustness artifacts: {name}")
        condition = json.loads((directory / rel).read_text(encoding="utf-8"))
        summary = metrics["robustness"]["conditions"][name]
        if summary != {
            "rank1": condition.get("identification", {}).get("rank1"),
            "n": condition.get("identification", {}).get("n_probe", 0),
        }:
            raise ValueError("Robustness summary disagrees with condition metrics")
    return metrics


def verify_experiment(exp, data_root: Path, results_root: Path) -> dict:
    from ivafr.datasets.splits import load_experiment_split

    inputs = experiment_inputs(exp, data_root)
    expected = {}
    for protocol in exp.protocols:
        for seed in exp.seeds:
            for arm in exp.arms:
                modality = ARM_MODALITY[arm.feature]
                split = load_experiment_split(
                    data_root, exp.dataset, modality, protocol, seed, inputs["frames"][modality]
                )
                expected[(arm.key, protocol, seed)] = digest(run_identity(exp, arm, split, inputs))
    actual = {}
    for path in results_root.glob("runs/*/config_resolved.json"):
        preview = json.loads(path.read_text(encoding="utf-8"))
        if preview.get("id") != exp.id:
            continue
        m = validate_run(path.parent)
        key = (m["arm"], m["protocol"], m["seed"])
        if key in actual:
            raise ValueError(
                f"Duplicate logical run {key}; use a clean output root or reviewed dedupe"
            )
        if expected.get(key) != m["run_id"]:
            raise ValueError(f"Stale or unexpected run {key}")
        actual[key] = m["run_id"]
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"Incomplete {exp.id}: missing {len(missing)} runs: {missing[:8]}")
    return {
        "exp_id": exp.id,
        "complete": True,
        "data_modality": inputs["data_modality"],
        "validated_runs": len(actual),
        "run_ids": sorted(actual.values()),
        "source_hash": inputs["source_hash"],
        "input_hash": digest(inputs["files"]),
    }
