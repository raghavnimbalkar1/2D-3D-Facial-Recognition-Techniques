"""Aggregation preserves protocol-specific chance baselines."""

from __future__ import annotations

import json

import pytest

from ivafr.pipelines.aggregate import collect_metrics, render_extended, render_t1


def test_chance_normalized_rank1_is_gallery_aware(tmp_path):
    metrics_path = tmp_path / "runs" / "p2" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "exp_id": "E01",
                "arm": "2D-PCA",
                "protocol": "P2_disjoint",
                "seed": 0,
                "dataset": {"name": "yaleb", "data_modality": "real"},
                "data_modality": "real",
                "identification": {
                    "rank1": 0.5,
                    "n_gallery": 23,
                    "rank5": 0.7,
                    "accuracy": 0.5,
                    "precision_macro": 0.5,
                    "recall_macro": 0.5,
                    "f1_macro": 0.5,
                    "mrr": 0.6,
                },
                "verification": {"eer": 0.4, "auc": 0.6},
            }
        ),
        encoding="utf-8",
    )

    frame = collect_metrics(tmp_path)
    assert frame.loc[0, "chance_rank1"] == pytest.approx(1 / 23)
    assert frame.loc[0, "rank1_over_chance"] == pytest.approx(11.5)
    assert "Rank-1 / Chance" in render_t1(frame).columns
    assert render_extended(frame).loc[0, "rank1_over_chance"].startswith("11.5000x")
