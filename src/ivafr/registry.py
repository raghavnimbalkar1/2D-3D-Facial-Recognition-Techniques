"""String-keyed registries for extractors, matchers and dataset adapters.

The registry pattern is what makes the whole experiment matrix declarative:
configs reference implementations by string, e.g. ``feature: pca`` resolves to
:class:`ivafr.features.pca.PCAFeature`. Adding a new method = register it, no
pipeline code changes.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_FEATURES: dict[str, type] = {}
_MATCHERS: dict[str, type] = {}
_DATASETS: dict[str, type] = {}


def register(name: str, store: dict[str, type]) -> Callable[[type], type]:
    """Return a decorator that inserts ``cls`` into ``store`` under ``name``."""

    def deco(cls: type) -> type:
        if name in store and store[name] is not cls:
            raise ValueError(f"Duplicate registry name {name!r} for {cls.__name__}")
        store[name] = cls
        return cls

    return deco


def register_feature(name: str) -> Callable[[type], type]:
    return register(name, _FEATURES)


def register_matcher(name: str) -> Callable[[type], type]:
    return register(name, _MATCHERS)


def register_dataset(name: str) -> Callable[[type], type]:
    return register(name, _DATASETS)


def get_feature(name: str) -> type:
    if name not in _FEATURES:
        raise KeyError(
            f"Unknown feature {name!r}. Registered: {sorted(_FEATURES)}"
        )
    return _FEATURES[name]


def get_matcher(name: str) -> type:
    if name not in _MATCHERS:
        raise KeyError(
            f"Unknown matcher {name!r}. Registered: {sorted(_MATCHERS)}"
        )
    return _MATCHERS[name]


def get_dataset(name: str) -> type:
    if name not in _DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {sorted(_DATASETS)}"
        )
    return _DATASETS[name]


def list_features() -> list[str]:
    return sorted(_FEATURES)


def list_matchers() -> list[str]:
    return sorted(_MATCHERS)


def list_datasets() -> list[str]:
    return sorted(_DATASETS)


def _import_all() -> None:
    """Import all modules that perform registration (side effect)."""
    from ivafr import datasets, features, matching  # noqa: F401  (registers)

    _ = datasets, features, matching


_import_all()

# Keep Any imported for typing convenience of consumers.
__all__ = [
    "register_feature",
    "register_matcher",
    "register_dataset",
    "get_feature",
    "get_matcher",
    "get_dataset",
    "list_features",
    "list_matchers",
    "list_datasets",
]
