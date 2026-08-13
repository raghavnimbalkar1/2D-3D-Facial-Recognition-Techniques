"""Robustness sweep utilities and artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ivafr.preprocess.degrade2d import occlude


def occlusion_sweep(images: list[np.ndarray], fractions=(0.2, 0.3, 0.4), seed: int = 0) -> dict[str, list[np.ndarray]]:
    """Build probe variants; gallery inputs remain untouched by the caller."""
    return {
        f"{float(frac):.1f}": [occlude(img, "random", float(frac), seed + i) for i, img in enumerate(images)]
        for frac in fractions
    }


def write_sweep(path: str | Path, experiment: str, rows: list[dict[str, Any]]) -> Path:
    """Persist a JSON sweep artifact with explicit conditions."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"exp_id": experiment, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    return out
