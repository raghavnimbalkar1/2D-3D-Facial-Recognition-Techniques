"""Stable fingerprints and atomic publication of benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def source_fingerprint() -> str:
    root = Path(__file__).parent
    return digest(
        {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}
    )


def dependency_versions() -> dict[str, str]:
    result = {}
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "opencv-python",
        "scikit-image",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "typer",
        "joblib",
        "psutil",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def manifest_fingerprint(frame) -> str:
    return digest(frame.sort_values("sample_id").fillna("").astype(str).to_dict("records"))
