"""Run many requests concurrently, respecting the provider's rate limit.

Resumable for free: requests already answered come from the cache.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .base import LLMRequest, LLMResponse
from .client import LLM


@dataclass
class BatchResult:
    index: int
    response: LLMResponse | None = None
    data: Any = None          # parsed JSON when json=True
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_batch(llm: LLM, requests: Sequence[LLMRequest], *, workers: int = 4, json: bool = False,
              on_result: Callable[[BatchResult], None] | None = None,
              progress_every: int = 50) -> list[BatchResult]:
    """Results come back in the same order as `requests`. Failures are captured per item,
    so one bad request does not stop the batch."""

    def one(i: int, req: LLMRequest) -> BatchResult:
        try:
            if json:
                data, resp = llm.complete_json(req)
                return BatchResult(i, resp, data)
            return BatchResult(i, llm.complete(req))
        except Exception as e:  # keep going; caller decides
            return BatchResult(i, error=e)

    results: list[BatchResult | None] = [None] * len(requests)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, i, r) for i, r in enumerate(requests)]
        for fut in as_completed(futures):
            r = fut.result()
            results[r.index] = r
            done += 1
            failed += not r.ok
            if on_result:
                on_result(r)
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(requests)} done, {failed} failed", flush=True)
    return results  # type: ignore[return-value]
