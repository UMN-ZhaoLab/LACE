"""Confidence utilities for gating model output."""

from __future__ import annotations

from typing import Any


_CONF_MAP = {
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}


def confidence_score(value: Any) -> float:
    """Normalize confidence values to a float in [0, 1]."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _CONF_MAP:
            return _CONF_MAP[key]
        try:
            return max(0.0, min(1.0, float(key)))
        except ValueError:
            return 0.0
    return 0.0
