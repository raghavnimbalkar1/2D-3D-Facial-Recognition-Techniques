"""Content-hash caching for preprocessing.

Each preprocessed output is accompanied by a sidecar JSON recording the
(config hash, input file digest). A re-run skips any output whose sidecar
matches — preprocessing re-runs are near-instant and cache-invalidating
changes (config edits, input edits) are detected automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from ivafr.integrity import file_hash, atomic_json

_SIDECAR = ".ivafr_cache.json"


def cfg_hash(cfg: dict[str, Any]) -> str:
    """SHA256 of a config subtree (canonical YAML dump)."""
    blob = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def file_digest(path: str | Path) -> str:
    """Digest of a file's identity: size + mtime_ns (cheap, stable)."""
    return file_hash(path)


def is_cached(out_path: str | Path, cfg: dict[str, Any], input_digest: str) -> bool:
    """True if ``out_path`` exists with a matching (cfg-hash, input-digest)."""
    out = Path(out_path)
    sidecar = out.with_name(out.name + _SIDECAR)
    if not out.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("cfg_hash") == cfg_hash(cfg)
        and meta.get("input_digest") == input_digest
        and meta.get("output_digest") == file_hash(out)
        and out.stat().st_size > 0
    )


def mark_cached(out_path: str | Path, cfg: dict[str, Any], input_digest: str) -> None:
    """Write the sidecar marking ``out_path`` as up-to-date."""
    out = Path(out_path)
    if not out.is_file() or out.stat().st_size == 0:
        raise ValueError(f"Cannot cache an absent or empty payload: {out}")
    sidecar = out.with_name(out.name + _SIDECAR)
    atomic_json(
        sidecar,
        {"cfg_hash": cfg_hash(cfg), "input_digest": input_digest, "output_digest": file_hash(out)},
    )
