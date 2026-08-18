"""Stage 4/5 — Full experiment run: extract -> match -> evaluate -> artifacts.

One run directory per (experiment, arm, protocol, seed):

    results/runs/<UTC-ts>_<exp>_<protocol>_s<seed>_<arm>/
        metrics.json          — every number, traceable
        config_resolved.json  — frozen resolved config
        sysinfo.json          — machine fingerprint + git SHA
        figures/*.png|pdf     — CMC, ROC, DET, FAR/FRR, score hist, CM
        tables/*.csv          — confusion matrix, per-class report
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ivafr.config import ArmConfig, ExperimentConfig
from ivafr.datasets.manifest import read_manifest
from ivafr.datasets.splits import make_split
from ivafr.datasets.splits import summary as split_summary
from ivafr.evaluation.identification import evaluate_identification
from ivafr.evaluation.verification import evaluate_verification
from ivafr.logging_utils import get_logger
from ivafr.pipelines.extract import extract_features
from ivafr.registry import get_matcher
from ivafr.seeding import set_all_seeds
from ivafr.sysinfo import sysinfo
from ivafr.viz import cm as viz_cm
from ivafr.viz import plots as viz_plots
from ivafr.viz import style as viz_style

log = get_logger("pipelines.run_experiment")

_ARM_MODALITY = {
    "pca": "2d", "lda": "2d", "lbp": "2d", "hog": "2d", "gabor": "2d", "arcface": "2d",
    "depth_pca": "3d", "depth_lbp": "3d", "normal_hog": "3d", "curv_hist": "3d",
    "lmk3d": "3d", "icp": "3d",
}


def _pool_ids(split: dict[str, Any], manifest: pd.DataFrame) -> list[str]:
    """Verification pool: P1 = all samples; P2 = evaluation subjects only."""
    if split["protocol"] == "P1_closed":
        return manifest["sample_id"].astype(str).tolist()
    eval_subj = set(split["eval_subjects"])
    return manifest.loc[manifest["subject_id"].isin(eval_subj), "sample_id"].astype(str).tolist()


def _train_ids(split: dict[str, Any], manifest: pd.DataFrame) -> list[str]:
    """Extractor fit pool: P1 = gallery; P2 = all background-subject samples."""
    if split["protocol"] == "P1_closed":
        return split["gallery_ids"]
    train_subj = set(split["train_subjects"])
    return manifest.loc[manifest["subject_id"].isin(train_subj), "sample_id"].astype(str).tolist()


def _subject_of(manifest: pd.DataFrame) -> dict[str, str]:
    return dict(zip(manifest["sample_id"].astype(str), manifest["subject_id"], strict=True))


def _condition_of(manifest: pd.DataFrame) -> dict[str, str]:
    def cond(row: pd.Series) -> str:
        illum = str(row["illumination"])
        yaw = row["pose_yaw"]
        yaw_s = f"{float(yaw):+.0f}" if str(yaw) != "NA" else "NA"
        return f"yaw{yaw_s}_{illum}"

    return {str(row["sample_id"]): cond(row) for _, row in manifest.iterrows()}


def run_experiment(
    exp: ExperimentConfig,
    data_root: str | Path,
    results_root: str | Path,
    force: bool = False,
    seeds: list[int] | None = None,
    protocols: list[str] | None = None,
    arms: list[str] | None = None,
) -> list[Path]:
    """Execute one experiment config; returns the created run directories."""
    data_root = Path(data_root)
    results_root = Path(results_root)
    manifest = read_manifest(data_root / "processed" / exp.dataset / "manifest.csv")
    modalities = set(manifest["data_modality"].astype(str))
    if len(modalities) != 1:
        raise ValueError(f"One experiment cannot mix data modalities: {sorted(modalities)}")
    data_modality = next(iter(modalities))
    subject_of = _subject_of(manifest)
    condition_of = _condition_of(manifest)
    interim = data_root / "interim" / exp.dataset

    run_dirs: list[Path] = []
    for protocol in protocols or exp.protocols:
        for seed in seeds or exp.seeds:
            split = make_split(manifest, protocol, seed=seed)
            log.info("Split: %s", split_summary(split))
            pool_ids = _pool_ids(split, manifest)
            train_ids = _train_ids(split, manifest)
            for arm in exp.arms:
                if arms and arm.key not in arms:
                    continue
                run_dir = _existing_run_dir(results_root, exp.id, protocol, seed, arm.key)
                if run_dir is not None and not force:
                    log.info("Skipping existing %s", run_dir)
                    run_dirs.append(run_dir)
                    continue
                run_dir = _new_run_dir(results_root, exp.id, protocol, seed, arm.key)
                metrics_path = run_dir / "metrics.json"
                robustness_conditions = (
                    exp.robustness.get("conditions", []) if exp.robustness else []
                )
                evaluations = [("clean", None)] + [
                    (str(c["name"]), c) for c in robustness_conditions
                ]
                condition_metrics = {}
                metrics = None
                for condition_name, augmentation in evaluations:
                    current = _evaluate_arm(
                        run_dir=run_dir,
                        arm=arm,
                        split=split,
                        manifest=manifest,
                        interim=interim,
                        pool_ids=pool_ids,
                        train_ids=train_ids,
                        subject_of=subject_of,
                        condition_of=condition_of,
                        do_identification=exp.evaluate_identification,
                        do_verification=exp.evaluate_verification,
                        do_timing=exp.evaluate_timing,
                        probe_augmentation=augmentation,
                    )
                    if metrics is None:
                        metrics = current
                    if condition_name != "clean":
                        condition_metrics[condition_name] = {
                            "rank1": current.get("identification", {}).get("rank1"),
                            "n": current.get("identification", {}).get("n_probe", 0),
                        }
                assert metrics is not None
                if condition_metrics:
                    metrics["robustness"] = {
                        "type": exp.robustness.get("type", "unknown"),
                        "conditions": condition_metrics,
                    }
                    if "identification" in metrics:
                        metrics["identification"]["per_condition"] = condition_metrics
                _write_meta(run_dir, exp)
                metrics["exp_id"] = exp.id
                metrics["arm"] = arm.key
                metrics["protocol"] = protocol
                metrics["seed"] = seed
                metrics["dataset"] = {"name": exp.dataset, "data_modality": data_modality}
                metrics["data_modality"] = data_modality
                metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
                run_dirs.append(run_dir)
                log.info("Wrote %s", metrics_path)
    return run_dirs


def _evaluate_arm(
    run_dir: Path,
    arm: ArmConfig,
    split: dict[str, Any],
    manifest: pd.DataFrame,
    interim: Path,
    pool_ids: list[str],
    train_ids: list[str],
    subject_of: dict[str, str],
    condition_of: dict[str, str],
    do_identification: bool,
    do_verification: bool,
    probe_augmentation: dict | None = None,
    do_timing: bool = False,
) -> dict[str, Any]:
    """Extract -> match -> evaluate for one arm; returns metrics dict."""
    seed = int(split["seed"])
    started = time.perf_counter()
    set_all_seeds(seed)
    modality = _ARM_MODALITY.get(arm.feature, "2d")
    dataset_name = manifest["dataset"].iloc[0] if "dataset" in manifest.columns else ""
    if modality == "3d" and dataset_name == "yaleb":
        raise ValueError("Pseudo-3D arms are restricted on Yale B due to environment block")
    gallery_ids, probe_ids = split["gallery_ids"], split["probe_ids"]

    X_train, X_gallery, X_probe, *_ = extract_features(
        feature_name=arm.feature,
        feature_params=arm.feature_params,
        train_ids=train_ids,
        gallery_ids=gallery_ids,
        probe_ids=probe_ids,
        manifest=manifest,
        interim=interim,
        modality=modality,
        seed=seed,
        probe_augmentation=probe_augmentation,
    )

    matcher = get_matcher(arm.matcher)(arm.matcher_params).fit(X_train, np.asarray(train_ids))
    metrics: dict[str, Any] = {"feature": {"name": arm.feature, "params": arm.feature_params},
                               "matcher": {"name": arm.matcher, "params": arm.matcher_params}}

    if do_identification:
        scores = matcher.score_matrix(X_probe, X_gallery)
        y_probe = np.asarray([subject_of[i] for i in probe_ids])
        y_gallery = np.asarray([subject_of[i] for i in gallery_ids])
        cond_probe = [condition_of[i] for i in probe_ids]
        res = evaluate_identification(scores, y_probe, y_gallery, conditions=cond_probe)
        metrics["identification"] = res.as_dict()
        _artifacts_identification(run_dir, arm.key, res)

    if do_verification:
        id_of = {sid: i for i, sid in enumerate(pool_ids)}
        pairs = split["verification"]
        gen = [(id_of[a], id_of[b]) for a, b in pairs["genuine"] if a in id_of and b in id_of]
        imp = [(id_of[a], id_of[b]) for a, b in pairs["impostor"] if a in id_of and b in id_of]
        X_pool = _pool_features(pool_ids, X_gallery, X_probe, gallery_ids, probe_ids)
        g_scores = matcher.scores_for_pairs(X_pool, gen)
        i_scores = matcher.scores_for_pairs(X_pool, imp)
        ver = evaluate_verification(g_scores, i_scores, seed=seed)
        metrics["verification"] = ver.as_dict()
        np.save(run_dir / "genuine_scores.npy", g_scores)
        np.save(run_dir / "impostor_scores.npy", i_scores)
        _artifacts_verification(run_dir, arm.key, ver, g_scores, i_scores)
    if do_timing:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics["timing"] = {
            "total_ms": float(elapsed_ms),
            "train_samples": len(train_ids),
            "gallery_samples": len(gallery_ids),
            "probe_samples": len(probe_ids),
            "ms_per_probe": float(elapsed_ms / max(len(probe_ids), 1)),
        }
    return metrics


def _pool_features(
    pool_ids: list[str],
    X_gallery: np.ndarray,
    X_probe: np.ndarray,
    gallery_ids: list[str],
    probe_ids: list[str],
) -> np.ndarray:
    """Assemble the pool feature matrix in ``pool_ids`` order."""
    g_idx = {sid: i for i, sid in enumerate(gallery_ids)}
    p_idx = {sid: i for i, sid in enumerate(probe_ids)}
    rows = [
        X_gallery[g_idx[sid]] if sid in g_idx else X_probe[p_idx[sid]] for sid in pool_ids
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def _artifacts_identification(run_dir: Path, arm_key: str, res) -> None:
    fig_dir = run_dir / "figures"
    tbl_dir = run_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    viz_style.set_output_dir(fig_dir)
    viz_cm.plot_cm(res.confusion, res.labels, f"fig_cm_{arm_key}")
    viz_cm.cm_csv(res.confusion, res.labels, tbl_dir / "cm.csv")
    per_class = pd.DataFrame(res.per_class).T.reset_index().rename(columns={"index": "subject"})
    per_class.to_csv(tbl_dir / "per_class.csv", index=False)


def _artifacts_verification(run_dir: Path, arm_key: str, ver, g_scores, i_scores) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    viz_style.set_output_dir(fig_dir)
    viz_plots.plot_roc({arm_key: (ver.roc_fpr, ver.roc_tpr, ver.auc)})
    viz_plots.plot_det({arm_key: (ver.det_far, ver.det_frr)})
    viz_plots.plot_far_frr({arm_key: (ver.far_frr_thr, ver.far_frr[0], ver.far_frr[1])})
    viz_plots.plot_score_hists({arm_key: (g_scores, i_scores)})


def _existing_run_dir(
    results_root: Path, exp_id: str, protocol: str, seed: int, arm: str
) -> Path | None:
    """Most recent completed run dir for (exp, protocol, seed, arm), or None.

    Run dir names embed a UTC timestamp, so a naive ``is_file`` check on a
    freshly generated name can never hit. Scan the runs tree instead, so
    re-runs are idempotent: an existing ``metrics.json`` means skip.
    """
    best: Path | None = None
    best_ts = ""
    for d in (results_root / "runs").glob(f"*_{exp_id}_{protocol}_s{seed}_{arm}"):
        if not (d / "metrics.json").is_file():
            continue
        ts = d.name.split("_", 1)[0]
        if best is None or ts > best_ts:
            best, best_ts = d, ts
    return best


def _new_run_dir(results_root: Path, exp_id: str, protocol: str, seed: int, arm: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    d = results_root / "runs" / f"{ts}_{exp_id}_{protocol}_s{seed}_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_meta(run_dir: Path, exp: ExperimentConfig) -> None:
    (run_dir / "config_resolved.json").write_text(
        json.dumps(exp.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "sysinfo.json").write_text(json.dumps(sysinfo(), indent=2), encoding="utf-8")
