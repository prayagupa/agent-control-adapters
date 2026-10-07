"""Common limits validated before native resources or policy files are opened."""

from __future__ import annotations

import math


def validate_run_options(timeout: float, emit_timeout: float, max_records: int) -> None:
    """Require finite positive policy deadlines and a bounded audit cap."""
    for duration in (timeout, emit_timeout):
        if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Timeouts must be positive finite seconds")
    if type(max_records) is not int or not 1 <= max_records <= 100_000:
        raise ValueError("max_records must be between 1 and 100000")
