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
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
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
    timing_repeats: int = 5
    timing_warmups: int = 1
    bootstrap_repeats: int = 1000

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

        for name in [raw["id"], raw["dataset"], *[a["key"] for a in raw["arms"]]]:
            if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Unsafe experiment, dataset or arm name")
        if int(raw.get("evaluate", {}).get("bootstrap_repeats", 1000)) < 2:
            raise ValueError("bootstrap_repeats must be at least 2")

        if raw["id"] == "E04" or any("fusion" in a.get("key", "").lower() for a in raw["arms"]):
            raise ValueError("Fusion is an unimplemented experimental capability")
        if not raw["arms"] or len({a["key"] for a in raw["arms"]}) != len(raw["arms"]):
            raise ValueError("Experiment requires nonempty, unique arm keys")
        if not raw["protocols"] or not set(raw["protocols"]) <= {"P1_closed", "P2_disjoint"}:
            raise ValueError("Unsupported or empty protocol list")
        if len(set(raw["protocols"])) != len(raw["protocols"]):
            raise ValueError("Duplicate protocols")
        if not raw["seeds"] or any(type(s) is not int or s < 0 for s in raw["seeds"]):
            raise ValueError("Seeds must be nonnegative integers")
        if len(set(raw["seeds"])) != len(raw["seeds"]):
            raise ValueError("Duplicate seeds")

        ds = self.dataset_config(raw["dataset"])
        pp = raw.get("preprocess", {})
        pp2d = self.preprocess_config(pp.get("2d", ds.get("preprocess_2d", "p2d_default")))
        pp3d = self.preprocess_config(pp.get("3d", ds.get("preprocess_3d", "p3d_default")))

        arms = [
            ArmConfig(
                key=a["key"],
                feature=a["feature"],
                matcher=a["matcher"],
                feature_params=self._component(
                    "features", a["feature"], a.get("feature_params", {})
                ),
                matcher_params=self._component(
                    "matchers", a["matcher"], a.get("matcher_params", {})
                ),
            )
            for a in raw["arms"]
        ]
        for arm in arms:
            if arm.feature in {"arcface", "lda"} or arm.matcher not in {
                "nn_cosine",
                "nn_l2",
                "nn_chi2",
            }:
                raise ValueError(f"Unsupported benchmark capability: {arm.feature}/{arm.matcher}")
        conditions = raw.get("robustness", {}).get("conditions", [])
        names = [c["name"] for c in conditions]
        if len(set(names)) != len(names) or "clean" in names:
            raise ValueError("Robustness condition names must be unique and cannot be clean")
        for c in conditions:
            if not str(c["name"]).replace("_", "").replace("-", "").isalnum():
                raise ValueError("Unsafe condition name")
            if (
                c.get("kind", "block") not in {"block", "random", "sunglasses"}
                or not 0 < c.get("fraction", 0.3) < 1
            ):
                raise ValueError("Invalid robustness condition")
        timing = raw.get("timing", {})
        if int(timing.get("repeats", 5)) < 1 or int(timing.get("warmups", 1)) < 0:
            raise ValueError("Invalid timing repeat counts")
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
            tables=raw.get("report", self._base.get("report", {})).get("tables", []),
            figures=raw.get("report", self._base.get("report", {})).get("figures", []),
            timing_repeats=int(timing.get("repeats", 5)),
            timing_warmups=int(timing.get("warmups", 1)),
            bootstrap_repeats=int(raw.get("evaluate", {}).get("bootstrap_repeats", 1000)),
        )

    def _component(self, directory: str, name: str, overrides: dict) -> dict:
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid component name")
        cfg = self._load(f"{directory}/{name}.yaml")
        params = deep_merge(cfg.get("params", {}), overrides)
        unknown = set(overrides) - set(cfg.get("params", {}))
        if unknown:
            raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
        for key in (
            "points",
            "radius",
            "blocks",
            "orientations",
            "downsample_factor",
            "pca_components",
            "max_components",
            "bins",
        ):
            if key in params and (not isinstance(params[key], (int, float)) or params[key] <= 0):
                raise ValueError(f"{name}.{key} must be positive")
        if "variance_keep" in params and not 0 < params["variance_keep"] <= 1:
            raise ValueError("variance_keep must be in (0,1]")
        if params.get("drop_first_k", 0) < 0:
            raise ValueError("drop_first_k cannot be negative")
        return params
