"""End-to-end: full E00 on the toy dataset -> valid metrics.json."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ivafr.config import ConfigResolver

CONFIGS = __import__("pathlib").Path(__file__).parents[1] / "configs"


@pytest.mark.slow
def test_e2e_e00_toy(toy_pipeline_dir):
    from ivafr.pipelines.run_experiment import run_experiment

    resolver = ConfigResolver(CONFIGS)
    exp = resolver.experiment("E00")
    run_dirs = run_experiment(exp, data_root=toy_pipeline_dir, results_root=toy_pipeline_dir / "results")

    assert len(run_dirs) == 2 * 2 * 5  # 2 arms x 2 protocols x 5 seeds

    for d in run_dirs:
        metrics = json.loads((d / "metrics.json").read_text())
        _assert_schema(metrics)
        assert (d / "config_resolved.json").is_file()
        assert (d / "sysinfo.json").is_file()
        assert (d / "figures").is_dir()


def _assert_schema(m):
    assert set(m) >= {"exp_id", "arm", "protocol", "seed"}
    ident = m["identification"]
    for key in ("rank1", "rank5", "rank10", "mrr", "f1_macro", "accuracy"):
        assert isinstance(ident[key], float)
    assert len(ident["cmc"]) == 20
    ver = m["verification"]
    for key in ("eer", "auc", "dprime"):
        assert isinstance(ver[key], float)
    assert len(ver["tar_at_far"]) == 3
    assert 0.0 <= ver["eer"] <= 1.0


@pytest.mark.slow
def test_e2e_determinism(toy_pipeline_dir):
    """Same config+seed -> identical metrics.json (excluding timing keys)."""
    from ivafr.pipelines.run_experiment import run_experiment

    resolver = ConfigResolver(CONFIGS)
    exp = resolver.experiment("E00")
    kwargs = dict(data_root=toy_pipeline_dir, seeds=[0], protocols=["P2_disjoint"], arms=["2D-PCA"])
    run_dirs = run_experiment(exp, results_root=toy_pipeline_dir / "results2", **kwargs)
    assert len(run_dirs) == 1
    first = json.loads((run_dirs[0] / "metrics.json").read_text())
    run_dirs2 = run_experiment(exp, results_root=toy_pipeline_dir / "results3", **kwargs)
    second = json.loads((run_dirs2[0] / "metrics.json").read_text())
    assert second["identification"]["rank1"] == first["identification"]["rank1"]
    assert second["verification"]["eer"] == first["verification"]["eer"]
    assert second["identification"]["cmc"] == first["identification"]["cmc"]


@pytest.mark.slow
def test_e2e_reasonable_accuracy(toy_pipeline_dir):
    """On clean frontal toy data, PCA should exceed chance (1/n_subjects)."""
    from ivafr.pipelines.run_experiment import run_experiment

    resolver = ConfigResolver(CONFIGS)
    exp = resolver.experiment("E00")
    run_dirs = run_experiment(exp, seeds=[0], protocols=["P1_closed"], data_root=toy_pipeline_dir,
                              results_root=toy_pipeline_dir / "results4")
    m = json.loads((run_dirs[0] / "metrics.json").read_text())
    assert m["identification"]["rank1"] > 0.5
    # 3D depth-PCA should also beat chance under illumination change.
    m3 = json.loads((run_dirs[1] / "metrics.json").read_text())
    assert m3["identification"]["rank1"] > 0.5