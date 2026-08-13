"""Deterministic seeding for the whole pipeline.

All randomness in the codebase must flow through :func:`set_all_seeds` so that
runs are reproducible: ``random``, ``numpy``, ``cv2``, and (if installed)
``torch`` / ``tensorflow`` RNGs are seeded together with ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Seed every RNG in the process.

    Args:
        seed: integer seed, e.g. from the experiment config ``seeds`` list.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import cv2  # noqa: PLC0415

        cv2.setRNGSeed(seed)
    except ImportError:  # pragma: no cover - cv2 is a hard dep, keep guard anyway
        pass

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf  # noqa: PLC0415

        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        from sklearn.utils import check_random_state  # noqa: PLC0415

        check_random_state(seed)
    except ImportError:  # pragma: no cover
        pass


def seed_int(sample_id: str) -> int:
    """Derive a stable per-sample seed from its canonical id.

    Args:
        sample_id: canonical sample identifier.

    Returns:
        Integer in ``[0, 2**31)``.
    """
    return int(hashlib_sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)


def hashlib_sha256(data: bytes) -> Any:
    """Return a sha256 hash object (import kept local for import speed)."""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data)
