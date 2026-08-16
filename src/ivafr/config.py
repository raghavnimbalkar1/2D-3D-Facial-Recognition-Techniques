"""Configuration loading and resolution.

Configs live in ``configs/`` as YAML. The base file provides defaults for
every key; dataset / preprocess / feature / matcher / experiment files are
merged over it (deep merge). The resolved, frozen config is written verbatim
into every run directory so each result is traceable to a config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ivafr.logging_utils import get_logger

log = get_logger("config")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dict. Fails loudly with the path in the message."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a mapping at top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class ArmConfig:
    """One method arm of an experiment: extractor + matcher pair."""

    key: str
    feature: str
    matcher: str
    feature_params: dict[str, Any] = field(default_factory=dict)
    matcher_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully resolved description of one experiment."""

    id: str
    name: str
    dataset: str
    protocols: list[str]
    seeds: list[int]
    preprocess_2d: dict[str, Any]
    preprocess_3d: dict[str, Any]
    arms: list[ArmConfig]
    evaluate_identification: bool = True
    evaluate_verification: bool = True
    evaluate_timing: bool = False
    robustness: dict[str, Any] = field(default_factory=dict)
    tables: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigResolver:
    """Loads YAML configs from a root directory and resolves experiments."""

    def __init__(self, configs_root: str | Path) -> None:
        self.root = Path(configs_root)
        self._base = self._load("base.yaml")

    def _load(self, rel: str) -> dict[str, Any]:
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required config missing: {path}")
        return load_yaml(path)

    def dataset_config(self, name: str) -> dict[str, Any]:
        return deep_merge(self._base.get("datasets", {}), self._load(f"datasets/{name}.yaml"))

    def preprocess_config(self, name: str) -> dict[str, Any]:
        return deep_merge(self._base.get("preprocess", {}), self._load(f"preprocess/{name}.yaml"))

    def experiment(self, exp_id: str) -> ExperimentConfig:
        """Resolve an experiment file against base/dataset defaults."""
        raw = self._load(f"experiments/{exp_id}.yaml")
        return self._resolve(raw)

    def _resolve(self, raw: dict[str, Any]) -> ExperimentConfig:
        missing = [k for k in ("id", "dataset", "protocols", "seeds", "arms") if k not in raw]
        if missing:
            raise ValueError(f"Experiment config missing keys: {missing}")

        ds = self.dataset_config(raw["dataset"])
        pp = raw.get("preprocess", {})
        pp2d = self.preprocess_config(pp.get("2d", ds.get("preprocess_2d", "p2d_default")))
        pp3d = self.preprocess_config(pp.get("3d", ds.get("preprocess_3d", "p3d_default")))

        arms = [
            ArmConfig(
                key=a["key"],
                feature=a["feature"],
                matcher=a["matcher"],
                feature_params=a.get("feature_params", {}),
                matcher_params=a.get("matcher_params", {}),
            )
            for a in raw["arms"]
        ]
        return ExperimentConfig(
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            dataset=raw["dataset"],
            protocols=raw["protocols"],
            seeds=raw["seeds"],
            preprocess_2d=pp2d,
            preprocess_3d=pp3d,
            arms=arms,
            evaluate_identification=raw.get("evaluate", {}).get("identification", True),
            evaluate_verification=raw.get("evaluate", {}).get("verification", True),
            evaluate_timing=raw.get("evaluate", {}).get("timing", False),
            robustness=raw.get("robustness", {}),
            tables=raw.get("report", {}).get("tables", []),
            figures=raw.get("report", {}).get("figures", []),
        )
