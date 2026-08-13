"""Shared fixtures: a tiny toy dataset on tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from ivafr.datasets.toy import generate_toy


@pytest.fixture(scope="session")
def toy_raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small toy dataset (4 subjects x 12 samples) generated once per session."""
    root = tmp_path_factory.mktemp("toy")
    raw = root / "raw"
    generate_toy(raw, n_subjects=4, n_samples=12, size=120, seed=7)
    return root


@pytest.fixture(scope="session")
def toy_manifest(toy_raw: Path):
    from ivafr.datasets.manifest import audit, samples_to_manifest
    from ivafr.datasets.toy import ToyAdapter

    adapter = ToyAdapter(raw_root=toy_raw / "raw")
    df = samples_to_manifest(adapter.discover())
    audit(df)
    return df


@pytest.fixture(scope="session")
def toy_pipeline_dir(toy_raw: Path):
    """Fully ingested + preprocessed toy dataset."""
    from ivafr.config import ConfigResolver
    from ivafr.pipelines.ingest import ingest
    from ivafr.pipelines.preprocess_run import preprocess_dataset

    root = toy_raw
    ingest("toy", root)
    resolver = ConfigResolver(Path(__file__).parents[1] / "configs")
    preprocess_dataset(
        "toy", root, resolver.preprocess_config("p2d_default"), resolver.preprocess_config("p3d_default")
    )
    return root