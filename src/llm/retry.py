"""Retry with exponential backoff and jitter for retryable provider errors."""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from .base import LLMError

T = TypeVar("T")


def call_with_retry(fn: Callable[[], T], *, max_attempts: int = 5, base_delay: float = 2.0,
                    max_delay: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                    on_retry: Callable[[int, LLMError, float], None] | None = None) -> T:
    """Call fn(); on a retryable LLMError wait and try again, up to max_attempts calls.
    Non-retryable errors (400, 401, 403, 404, ...) are raised at once."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except LLMError as e:
            if not e.retryable or attempt >= max_attempts:
                raise
            delay = e.retry_after if e.retry_after else min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= 1 + random.uniform(0, 0.25)
            if on_retry:
                on_retry(attempt, e, delay)
            sleep(delay)
