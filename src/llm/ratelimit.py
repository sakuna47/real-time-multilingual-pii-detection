"""Requests-per-minute limiter shared by all threads that use one provider."""

from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    def __init__(self, rpm: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may be sent. Returns the seconds waited."""
        if not self.interval:
            return 0.0
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        wait = start - now
        if wait > 0:
            self._sleep(wait)
        return wait


_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def limiter_for(provider: str, rpm: float) -> RateLimiter:
    """One limiter per provider per process, so separate LLM objects share the budget."""
    with _limiters_lock:
        lim = _limiters.get(provider)
        if lim is None or lim.interval != (60.0 / rpm if rpm and rpm > 0 else 0.0):
            lim = _limiters[provider] = RateLimiter(rpm)
        return lim
