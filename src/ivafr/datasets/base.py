"""Dataset adapter interfaces.

An adapter turns a raw download into a canonical list of :class:`Sample`
records plus loaders for each modality. Nothing else — all downstream stages
(manifest, splits, preprocessing, experiments) consume only these primitives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    """One capture (one image / one scan / paired 2D+3D)."""

    dataset: str
    subject_id: str
    sample_id: str
    path_2d: Path | None = None
    path_3d: Path | None = None
    path_landmarks: Path | None = None
    meta: dict = field(default_factory=dict)

    def get(self, key: str, default: object = None) -> object:
        """Read a metadata field (expression, pose_yaw, illumination, ...)."""
        return self.meta.get(key, default)


@dataclass(frozen=True)
class Cloud3D:
    """Canonical 3D representation: unorganized point cloud in millimetres.

    Attributes:
        points: (N, 3) float32 array, millimetres.
        rgb: optional (N, 3) uint8 per-point colour.
        valid: optional (N,) bool mask (False = interpolated/unknown points).
    """

    points: np.ndarray
    rgb: np.ndarray | None = None
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N,3), got {self.points.shape}")
        object.__setattr__(self, "points", np.asarray(self.points, dtype=np.float32))


class DatasetAdapter(ABC):
    """Turns a raw download into a canonical manifest. NOTHING else.

    Attributes:
        name: canonical dataset name, also used in ``data/raw/<name>``.
    """

    name: str = ""

    def __init__(self, raw_root: str | Path, anonymize: bool = False) -> None:
        self.raw_root = Path(raw_root)
        self.anonymize = anonymize

    @abstractmethod
    def discover(self) -> list[Sample]:
        """Enumerate every usable capture in the raw root.

        Fails loudly (or logs with count) when a subject has fewer than two
        samples — genuine pairs cannot be formed otherwise.
        """

    @abstractmethod
    def load_2d(self, s: Sample) -> np.ndarray:
        """Return the 2D image as HxWx3 uint8 BGR."""

    @abstractmethod
    def load_3d(self, s: Sample) -> Cloud3D:
        """Return the 3D capture as a millimetre point cloud."""

    def load_landmarks(self, s: Sample) -> np.ndarray | None:
        """Return (K,2) pixel or (K,3) mm landmarks, or None."""
        return None
