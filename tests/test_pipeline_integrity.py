"""Regression checks of the complete producer/consumer artifact contract."""

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from ivafr.config import ConfigResolver
from ivafr.datasets.manifest import read_manifest
from ivafr.integrity import atomic_json
from ivafr.pipelines.aggregate import collect_metrics, aggregate
from ivafr.pipelines.preprocess_run import preprocess_dataset
from ivafr.pipelines.provenance import experiment_inputs, validate_run, verify_experiment
from ivafr.pipelines.run_experiment import run_experiment

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture(scope="module")
def all_2d_runs(toy_pipeline_dir, tmp_path_factory):
    resolver = ConfigResolver(CONFIGS)
    exp = replace(
        resolver.experiment("E00"),
        id="TEST_2D",
        seeds=[0],
        arms=resolver.experiment("E11").arms,
        bootstrap_repeats=20,
        evaluate_timing=True,
        timing_repeats=2,
        timing_warmups=1,
    )
    out = tmp_path_factory.mktemp("all_2d")
    directories = run_experiment(exp, toy_pipeline_dir, out)
    return exp, out, directories


@pytest.mark.slow
def test_every_2d_method_both_protocols_and_completion(toy_pipeline_dir, all_2d_runs):
    exp, out, directories = all_2d_runs
    assert len(directories) == 8
    report = verify_experiment(exp, toy_pipeline_dir, out)
    assert report["validated_runs"] == 8
    assert report["data_modality"] == "synthetic_toy"
    for directory in directories:
        m = validate_run(directory)
        assert m["verification"]["ci_method"] == "subject_cluster_bootstrap"
        timing = m["timing"]
        assert timing["ms_per_probe"] == pytest.approx(
            (timing["transform"]["median_ms"] + timing["match"]["median_ms"])
            / timing["probe_samples"]
        )
        assert timing["pipeline_total_ms"] > timing["ms_per_probe"]
    # Complete artifacts are reused; no extra directories appear.
    assert run_experiment(exp, toy_pipeline_dir, out) == directories
    atomic_json(out / f"validation_{exp.id}.json", report)
    aggregate(out, out)
    assert "matrix complete" in (out / "RESULTS.md").read_text(encoding="utf-8")
    assert len(collect_metrics(out)) == 8
    assert (out / "tables/T3_verification.csv").is_file()


@pytest.mark.parametrize(
    "artifact", ["inputs.json", "split.json", "metrics.json", "genuine_scores.npy"]
)
def test_tampered_artifact_cannot_resume(all_2d_runs, tmp_path, artifact):
    _, _, directories = all_2d_runs
    run = tmp_path / "tampered"
    shutil.copytree(directories[0], run)
    with (run / artifact).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        validate_run(run)


def test_incomplete_and_duplicate_runs_fail(all_2d_runs, toy_pipeline_dir, tmp_path):
    exp, _, directories = all_2d_runs
    with pytest.raises(ValueError, match="Incomplete"):
        verify_experiment(exp, toy_pipeline_dir, tmp_path)
    shutil.copytree(directories[0], tmp_path / "runs/a")
    shutil.copytree(directories[0], tmp_path / "runs/b")
    with pytest.raises(ValueError, match="Duplicate"):
        collect_metrics(tmp_path)
    (tmp_path / "runs/b/completed.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        collect_metrics(tmp_path)


def test_cache_bundle_regeneration_and_reingest(toy_pipeline_dir, tmp_path):
    from ivafr.pipelines.ingest import ingest

    # Keep the session's benchmark fixture immutable.
    root = tmp_path / "data"
    shutil.copytree(toy_pipeline_dir / "raw", root / "raw")
    resolver = ConfigResolver(CONFIGS)
    exp = replace(resolver.experiment("E00"), arms=resolver.experiment("E11").arms)
    ingest("toy", root)
    preprocess_dataset("toy", root, exp.preprocess_2d, exp.preprocess_3d, modality="2d")
    crop = next((root / "interim/toy/crops2d").rglob("*_g64.npy"))
    expected = np.load(crop)
    crop.unlink()
    with pytest.raises(ValueError, match="Incomplete or corrupt"):
        experiment_inputs(exp, root)
    preprocess_dataset("toy", root, exp.preprocess_2d, exp.preprocess_3d, modality="2d")
    np.testing.assert_array_equal(expected, np.load(crop))
    ingest("toy", root)
    preprocess_dataset("toy", root, exp.preprocess_2d, exp.preprocess_3d, modality="2d")
    manifest = read_manifest(root / "processed/toy/manifest.csv")
    assert manifest.detect_ok.all() and manifest.align_ok.all()
    experiment_inputs(exp, root)


def test_mismatched_multimodal_cohorts_rejected(toy_pipeline_dir, tmp_path):
    from ivafr.datasets.manifest import write_manifest

    root = tmp_path / "cohort"
    shutil.copytree(toy_pipeline_dir / "interim", root / "interim")
    manifest = read_manifest(toy_pipeline_dir / "processed/toy/manifest.csv")
    manifest.loc[0, "nosetip_ok"] = False
    write_manifest(manifest, root / "processed/toy/manifest.csv")
    with pytest.raises(ValueError, match="matching subject/condition"):
        experiment_inputs(ConfigResolver(CONFIGS).experiment("E00"), root)


def test_changed_configuration_cannot_reuse_current_runs(all_2d_runs, toy_pipeline_dir):
    exp, out, _ = all_2d_runs
    changed = replace(exp, bootstrap_repeats=exp.bootstrap_repeats + 1)
    with pytest.raises(ValueError, match="Stale"):
        verify_experiment(changed, toy_pipeline_dir, out)


@pytest.mark.slow
def test_robustness_separate_artifacts_and_natural_conditions(toy_pipeline_dir, tmp_path):
    resolver = ConfigResolver(CONFIGS)
    base = resolver.experiment("E00")
    exp = replace(
        base,
        id="TEST_ROBUST",
        seeds=[0],
        protocols=["P2_disjoint"],
        arms=[resolver.experiment("E11").arms[2]],
        bootstrap_repeats=20,
        robustness={"conditions": [{"name": "block20", "kind": "block", "fraction": 0.2}]},
    )
    directory = run_experiment(exp, toy_pipeline_dir, tmp_path)[0]
    metrics = validate_run(directory)
    assert metrics["identification"]["per_condition"]
    assert (
        metrics["robustness"]["conditions"]["block20"]["n"] == metrics["identification"]["n_probe"]
    )
    assert (directory / "conditions/block20/genuine_scores.npy").is_file()


@pytest.mark.slow
def test_classical_3d_methods_on_synthetic_observations(toy_pipeline_dir, tmp_path):
    resolver = ConfigResolver(CONFIGS)
    exp = replace(
        resolver.experiment("E00"),
        id="TEST_3D",
        seeds=[0],
        protocols=["P1_closed"],
        arms=resolver.experiment("E12").arms,
        bootstrap_repeats=20,
    )
    runs = run_experiment(exp, toy_pipeline_dir, tmp_path)
    assert len(runs) == 4
    assert verify_experiment(exp, toy_pipeline_dir, tmp_path)["complete"]
