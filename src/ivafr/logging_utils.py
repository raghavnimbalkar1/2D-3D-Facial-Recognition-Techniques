"""Structured logging utilities.

All modules log through :func:`get_logger`, which returns a child logger of the
``ivafr`` hierarchy. The root logger is configured once by :func:`setup_logging`
with console + optional file handler.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the ``ivafr`` root logger.

    Idempotent: re-configuration is a no-op after the first call with the same
    level, except the file handler which is appended if not present.

    Args:
        level: logging level name (``INFO``, ``DEBUG``, ...).
        log_file: optional path to append log output to.
    """
    root = logging.getLogger("ivafr")
    if root.handlers:
        return
    root.setLevel(level.upper())
    fmt = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger child of the ``ivafr`` hierarchy.

    Args:
        name: module name, e.g. ``ivafr.datasets.toy``.
    """
    return logging.getLogger(f"ivafr.{name}")
