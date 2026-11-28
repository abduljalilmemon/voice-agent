"""Per-stage latency metrics: a rolling window of samples per stage, summarised as p50 / p95 / max.

In memory and per process (like the rate limiter) -- enough for a single instance. Failed attempts
are recorded too: a timeout is latency the caller actually waited through.
"""
import math
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

WINDOW = 500  # most recent samples kept per stage


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


class LatencyMetrics:
    def __init__(self, window: int = WINDOW):
        self._samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.Lock()  # recorded from the event loop and from worker threads

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._samples[stage].append(seconds)

    @contextmanager
    def timer(self, stage: str, into: dict | None = None):
        """Time a block as `stage`; optionally also store the milliseconds in `into[stage]`."""
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            self.record(stage, seconds)
            if into is not None:
                into[stage] = round(seconds * 1000)

    def summary(self) -> dict[str, dict]:
        with self._lock:
            snapshot = {stage: sorted(values) for stage, values in self._samples.items() if values}
        return {
            stage: {
                "count": len(values),
                "p50_ms": round(percentile(values, 50) * 1000),
                "p95_ms": round(percentile(values, 95) * 1000),
                "max_ms": round(values[-1] * 1000),
            }
            for stage, values in sorted(snapshot.items())
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


metrics = LatencyMetrics()
