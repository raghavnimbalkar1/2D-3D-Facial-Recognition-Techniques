"""Small report helpers kept separate from metric computation."""

from __future__ import annotations

from pathlib import Path


def write_report_header(path: str | Path, dataset: str, modality_note: str) -> None:
    """Write the reproducibility/interpretation header for RESULTS.md."""
    Path(path).write_text(
        "# ivafr results\n\n"
        f"Dataset: `{dataset}`. 3D outputs are `{modality_note}`.\n\n"
        "All reported values are generated from resolved configs and subject-disjoint splits.\n",
        encoding="utf-8",
    )
