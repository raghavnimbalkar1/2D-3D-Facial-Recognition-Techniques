"""Stage 0 — Ingest: adapter -> manifest.csv + audit report."""

from __future__ import annotations

from pathlib import Path

from ivafr.datasets.manifest import audit, audit_report, samples_to_manifest, write_manifest
from ivafr.logging_utils import get_logger
from ivafr.registry import get_dataset

log = get_logger("pipelines.ingest")


def ingest(dataset: str, data_root: str | Path, anonymize: bool = False) -> Path:
    """Discover a dataset via its adapter and write the canonical manifest.

    Args:
        dataset: registered dataset name (``toy``, later ``texas3d``, ...).
        data_root: ``data/`` root; raw data expected at ``data/raw/<dataset>``.
        anonymize: hash subject ids (publication mode).

    Returns:
        Path of the written ``manifest.csv``.
    """
    data_root = Path(data_root)
    adapter_cls = get_dataset(dataset)
    adapter = adapter_cls(raw_root=data_root / "raw", anonymize=anonymize)
    log.info("Discovering %s under %s", dataset, data_root / "raw")
    samples = adapter.discover()
    log.info("Discovered %d samples", len(samples))
    if anonymize:
        samples = [_anonymize(s) for s in samples]

    manifest = samples_to_manifest(samples)
    stats = audit(manifest)
    print(audit_report(stats))

    out_dir = data_root / "processed" / dataset
    out_path = out_dir / "manifest.csv"
    write_manifest(manifest, out_path)
    return out_path


def _anonymize(sample):
    """Hash the subject id so no raw identifier leaks into artifacts."""
    import hashlib

    from dataclasses import replace

    # Keep the canonical S### shape so all split/report code remains usable,
    # while making the identifier unlinkable to the original subject label.
    h = hashlib.sha256(sample.subject_id.encode("utf-8")).hexdigest()
    digits = "".join(str(int(ch, 16) % 10) for ch in h[:8])
    return replace(sample, subject_id=f"S{digits}")
