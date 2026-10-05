"""Stage 0 — Ingest: adapter -> manifest.csv + audit report."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ivafr.datasets.manifest import audit, audit_report, samples_to_manifest, write_manifest
from ivafr.logging_utils import get_logger
from ivafr.registry import get_dataset
from ivafr.integrity import atomic_json, manifest_fingerprint

log = get_logger("pipelines.ingest")


def ingest(dataset: str, data_root: str | Path, anonymize: bool = False) -> Path:
    """Discover a dataset via its adapter and write the canonical manifest.

    Args:
        dataset: registered dataset name (``toy``, later ``texas3d``, ...).
        data_root: ``data/`` root; raw data expected at ``data/raw/<dataset>``.
        anonymize: pseudonymize subject IDs; source paths/images remain identifying.

    Returns:
        Path of the written ``manifest.csv``.
    """
    data_root = Path(data_root)
    adapter_cls = get_dataset(dataset)
    adapter = adapter_cls(raw_root=data_root / "raw", anonymize=anonymize)
    log.info("Discovering %s under %s", dataset, data_root / "raw")
    samples = adapter.discover()
    for sample in samples:
        for path in (sample.path_2d, sample.path_3d):
            if path is not None and (not path.is_file() or path.stat().st_size == 0):
                raise ValueError(f"Missing or empty source file: {path}")
    log.info("Discovered %d samples", len(samples))
    _guard_duplicate_yaleb(dataset, samples)
    if anonymize:
        samples = [_anonymize(s) for s in samples]

    manifest = samples_to_manifest(samples)
    stats = audit(manifest)
    print(audit_report(stats))

    out_dir = data_root / "processed" / dataset
    out_path = out_dir / "manifest.csv"
    write_manifest(manifest, out_path)
    atomic_json(
        out_dir / "ingestion.json",
        {
            "dataset": dataset,
            "manifest_hash": manifest_fingerprint(manifest),
            "subjects": stats.n_subjects,
            "samples": stats.n_samples,
            "discovery": getattr(adapter, "discovery_report", {}),
        },
    )
    return out_path


def _guard_duplicate_yaleb(dataset: str, samples: list) -> None:
    """Refuse a Yale B mirror whose files are pixel-identical duplicates."""
    if dataset != "yaleb" or len(samples) < 2:
        return
    hashes = {
        hashlib.sha256(s.path_2d.read_bytes()).hexdigest()
        for s in samples
        if s.path_2d is not None and s.path_2d.is_file()
    }
    if len(hashes) == 1:
        raise ValueError(
            "Yale B input contains one identical pixel file repeated across all "
            f"{len(samples)} samples; refusing to benchmark a corrupted mirror"
        )


def _anonymize(sample):
    """Pseudonymize a subject ID; this does not anonymize faces or source paths."""
    import hashlib

    from dataclasses import replace

    # Keep an S-prefixed label for downstream reports. This deterministic,
    # unsalted mapping is linkable and must not be presented as anonymization.
    h = hashlib.sha256(sample.subject_id.encode("utf-8")).hexdigest()
    digits = "".join(str(int(ch, 16) % 10) for ch in h[:8])
    return replace(sample, subject_id=f"S{digits}")
