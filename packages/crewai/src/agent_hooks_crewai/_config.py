"""Shared admission checks applied before loading application policy configuration."""

from __future__ import annotations

import math


def validate_run_options(timeout: float, emit_timeout: float, max_records: int) -> None:
    """Require positive finite deadlines and a bounded audit cap; raise on invalid input."""
    for duration in (timeout, emit_timeout):
        if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Timeouts must be positive finite seconds")
    if type(max_records) is not int or not 1 <= max_records <= 100_000:
        raise ValueError("max_records must be between 1 and 100000")
