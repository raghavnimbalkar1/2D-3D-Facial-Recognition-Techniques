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
    "pca": "2d",
    "lda": "2d",
    "lbp": "2d",
    "hog": "2d",
    "gabor": "2d",
    "arcface": "2d",
    "depth_pca": "3d",
    "depth_lbp": "3d",
    "normal_hog": "3d",
    "curv_hist": "3d",
    "lmk3d": "3d",
    "icp": "3d",
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
        return f"yaw{yaw_s}_{illum}_expression:{row.get('expression', 'NA')}_occlusion:{row.get('occlusion', 'NA')}"

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
    """Execute only audited, persisted splits; publish complete runs atomically."""
    from ivafr.datasets.splits import load_experiment_split
    from ivafr.integrity import atomic_json, digest
    from ivafr.pipelines.provenance import experiment_inputs, run_identity, complete_run

    data_root, results_root = Path(data_root), Path(results_root)
    selected_seeds = exp.seeds if seeds is None else seeds
    selected_protocols = exp.protocols if protocols is None else protocols
    selected_arms = [a for a in exp.arms if arms is None or a.key in arms]
    if not selected_seeds or not set(selected_seeds) <= set(exp.seeds):
        raise ValueError("Seeds must be a nonempty subset of the configured seeds")
    if not selected_protocols or not set(selected_protocols) <= set(exp.protocols):
        raise ValueError("Protocols must be a nonempty subset of the experiment")
    if not selected_arms or (arms is not None and not set(arms) <= {a.key for a in exp.arms}):
        raise ValueError("Unknown or empty arm selection")
    inputs = experiment_inputs(exp, data_root)
    work = []
    # Preflight the entire requested matrix before creating any run artifacts.
    for protocol in selected_protocols:
        for seed in selected_seeds:
            subject_partition = None
            for arm in selected_arms:
                modality = _ARM_MODALITY[arm.feature]
                frame = inputs["frames"][modality]
                split = load_experiment_split(
                    data_root, exp.dataset, modality, protocol, seed, frame
                )
                partition = (split["train_subjects"], split["eval_subjects"])
                if subject_partition is not None and partition != subject_partition:
                    raise ValueError("Multimodal subject partitions disagree")
                subject_partition = partition
                work.append((arm, frame, split))
    run_dirs = []
    for arm, frame, split in work:
        protocol, seed = split["protocol"], split["seed"]
        provenance = run_identity(exp, arm, split, inputs)
        run_id = digest(provenance)
        directory = _existing_run_dir(results_root, exp.id, protocol, seed, arm.key, run_id)
        if directory is not None and not force:
            run_dirs.append(directory)
            continue
        directory = _new_run_dir(results_root, exp.id, protocol, seed, arm.key)
        _write_meta(directory, exp)
        atomic_json(directory / "split.json", split)
        atomic_json(directory / "inputs.json", inputs["files"])
        subject_of = _subject_of(frame)
        metrics = None
        robustness = {}
        for name, augmentation in [("clean", None)] + [
            (c["name"], c) for c in exp.robustness.get("conditions", [])
        ]:
            target = directory if name == "clean" else directory / "conditions" / name
            target.mkdir(parents=True, exist_ok=True)
            current = _evaluate_arm(
                run_dir=target,
                arm=arm,
                split=split,
                manifest=frame,
                interim=data_root / "interim" / exp.dataset,
                pool_ids=_pool_ids(split, frame),
                train_ids=split["train_ids"],
                subject_of=subject_of,
                condition_of=_condition_of(frame),
                do_identification=exp.evaluate_identification,
                do_verification=exp.evaluate_verification,
                do_timing=exp.evaluate_timing,
                probe_augmentation=augmentation,
                timing_repeats=exp.timing_repeats,
                timing_warmups=exp.timing_warmups,
                bootstrap_repeats=exp.bootstrap_repeats,
            )
            if name == "clean":
                metrics = current
            else:
                atomic_json(target / "metrics.json", current)
                robustness[name] = {
                    "rank1": current.get("identification", {}).get("rank1"),
                    "n": current.get("identification", {}).get("n_probe", 0),
                }
        if robustness:
            metrics["robustness"] = {
                "type": exp.robustness.get("type", "occlusion"),
                "conditions": robustness,
            }
        metrics.update(
            exp_id=exp.id,
            arm=arm.key,
            protocol=protocol,
            seed=seed,
            dataset={"name": exp.dataset, "data_modality": inputs["data_modality"]},
            data_modality=inputs["data_modality"],
            run_id=run_id,
            provenance=provenance,
        )
        atomic_json(directory / "metrics.json", metrics)
        complete_run(directory, run_id)
        run_dirs.append(directory)
        log.info("Completed %s", directory)
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
    timing_repeats: int = 5,
    timing_warmups: int = 1,
    bootstrap_repeats: int = 1000,
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
    timing = {} if do_timing else None

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
        timing=timing,
        timing_repeats=timing_repeats,
        timing_warmups=timing_warmups,
    )

    matcher = get_matcher(arm.matcher)(arm.matcher_params).fit(
        X_train, np.asarray([subject_of[i] for i in train_ids])
    )
    if do_timing:
        from ivafr.evaluation.timing import time_callable

        timing["match"] = time_callable(
            lambda: matcher.score_matrix(X_probe, X_gallery), timing_repeats, timing_warmups
        )
    metrics: dict[str, Any] = {
        "feature": {"name": arm.feature, "params": arm.feature_params},
        "matcher": {"name": arm.matcher, "params": arm.matcher_params},
    }

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
        ver = evaluate_verification(
            g_scores,
            i_scores,
            seed=seed,
            n_boot=bootstrap_repeats,
            genuine_subjects=[(subject_of[a], subject_of[b]) for a, b in pairs["genuine"]],
            impostor_subjects=[(subject_of[a], subject_of[b]) for a, b in pairs["impostor"]],
        )
        metrics["verification"] = ver.as_dict()
        np.save(run_dir / "genuine_scores.npy", g_scores)
        np.save(run_dir / "impostor_scores.npy", i_scores)
        _artifacts_verification(run_dir, arm.key, ver, g_scores, i_scores)
    if do_timing:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics["timing"] = {
            **timing,
            "pipeline_total_ms": float(elapsed_ms),
            "train_samples": len(train_ids),
            "gallery_samples": len(gallery_ids),
            "probe_samples": len(probe_ids),
            "ms_per_probe": float(
                (timing["transform"]["median_ms"] + timing["match"]["median_ms"]) / len(probe_ids)
            ),
            "definition": "median probe transform + gallery matching; excludes fitting, IO, metrics and plots",
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
    rows = [X_gallery[g_idx[sid]] if sid in g_idx else X_probe[p_idx[sid]] for sid in pool_ids]
    return np.stack(rows, axis=0).astype(np.float32)


def _artifacts_identification(run_dir: Path, arm_key: str, res) -> None:
    fig_dir = run_dir / "figures"
    tbl_dir = run_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    viz_style.set_output_dir(fig_dir)
    viz_plots.plot_cmc({arm_key: res.cmc})
    viz_cm.plot_cm(res.confusion, res.labels, f"fig_cm_{arm_key}")
    viz_cm.cm_csv(res.confusion, res.labels, tbl_dir / "cm.csv")
    per_class = pd.DataFrame(res.per_class).T.reset_index().rename(columns={"index": "subject"})
    per_class.to_csv(tbl_dir / "per_class.csv", index=False)


def _artifacts_verification(run_dir: Path, arm_key: str, ver, g_scores, i_scores) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    viz_style.set_output_dir(fig_dir)
    viz_plots.plot_roc({arm_key: (ver.roc_fpr, ver.roc_tpr, ver.auc)})
    viz_plots.plot_det({arm_key: (ver.det_far, ver.det_frr)})
    viz_plots.plot_far_frr({arm_key: (ver.far_frr_thr, ver.far_frr[0], ver.far_frr[1])})
    viz_plots.plot_score_hists({arm_key: (g_scores, i_scores)})
    labels = np.r_[np.ones_like(g_scores), np.zeros_like(i_scores)]
    scores = np.r_[g_scores, i_scores]
    precision, recall, _ = precision_recall_curve(labels, scores)
    viz_plots.plot_pr(
        {arm_key: (precision, recall, float(average_precision_score(labels, scores)))}
    )


def _existing_run_dir(
    results_root: Path, exp_id: str, protocol: str, seed: int, arm: str, run_id: str | None = None
) -> Path | None:
    """Most recent completed run dir for (exp, protocol, seed, arm), or None.

    Run dir names embed a UTC timestamp, so a naive ``is_file`` check on a
    freshly generated name can never hit. Scan the runs tree and reuse only
    a checksummed completed bundle with the exact current run identity.
    """
    best: Path | None = None
    best_ts = ""
    for d in (results_root / "runs").glob(f"*_{exp_id}_{protocol}_s{seed}_{arm}"):
        from ivafr.pipelines.provenance import validate_run

        try:
            saved = validate_run(d)
        except (OSError, ValueError, KeyError):
            continue
        if run_id is None or saved["run_id"] != run_id:
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
