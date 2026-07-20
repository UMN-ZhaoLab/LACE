"""Agent runner utilities with retry, backoff, and structured output handling."""

from __future__ import annotations

import random
import time
from typing import Any, Callable


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retryable_keywords = [
        "rate limit",
        "timeout",
        "connection",
        "temporarily unavailable",
        "503",
        "429",
        "premature close",
        "closed",
        "reset",
        "broken pipe",
    ]
    return any(kw in msg for kw in retryable_keywords)


def invoke_with_backoff(
    structured: Any,
    messages: list[Any],
    max_attempts: int,
    _sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Invoke a structured LLM with exponential backoff on retryable errors.

    Args:
        structured: LangChain structured output runnable.
        messages: List of LangChain message objects.
        max_attempts: Maximum invocation attempts.
        _sleep: Sleep function (injected for testing).

    Returns:
        The structured output response.

    Raises:
        RuntimeError: If all attempts are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return structured.invoke(messages)
        except Exception as exc:
            last_error = exc
            if _is_retryable_error(exc) and attempt < max_attempts - 1:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                _sleep(backoff)
            elif attempt < max_attempts - 1:
                # Non-retryable but still within budget; try once more without delay
                continue
            else:
                break
    raise RuntimeError(
        f"LLM invocation failed after {max_attempts} attempts: {last_error}"
    ) from last_error
