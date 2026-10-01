import pytest
from ivafr.evaluation.timing import time_callable


def test_repeats_warmup_and_summary(monkeypatch):
    ticks = iter([0.0, 0.002, 1.0, 1.004, 2.0, 2.006])
    monkeypatch.setattr("ivafr.evaluation.timing.time.perf_counter", lambda: next(ticks))
    calls = []
    result = time_callable(lambda: calls.append(1), repeats=3, warmups=2)
    assert len(calls) == 5
    assert result["median_ms"] == pytest.approx(4)
    assert result["repeats"] == 3 and result["warmups"] == 2
