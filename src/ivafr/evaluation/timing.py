"""Single-pass timing records for the trimmed E10 compute experiment."""

from __future__ import annotations

import platform
import statistics
import time
from typing import Any, Callable


def time_callable(
    fn: Callable[[], Any], repeats: int = 5, warmups: int = 1
) -> dict[str, float | int]:
    """Measure a callable and return median plus repeat statistics in ms."""
    values = []
    if repeats < 1 or warmups < 0:
        raise ValueError("Invalid timing repetitions")
    for _ in range(warmups):
        fn()
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "std_ms": float(statistics.pstdev(values)),
        "repeats": len(values),
        "warmups": warmups,
        "python": platform.python_version(),
    }
