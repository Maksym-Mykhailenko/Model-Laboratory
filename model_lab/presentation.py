"""Presentation helpers for Model Laboratory.

These functions alter only how numerical results are displayed. They do not modify
model values or analysis results.
"""

from __future__ import annotations

import math


def clean_display_number(
    value: float,
    *,
    zero_tolerance: float = 1e-10,
) -> float:
    """Return a numerical value cleaned only for presentation.

    Values sufficiently close to zero are returned as exactly ``0.0``.  This helper is
    intended for chart coordinates and hover data; it never mutates stored analysis
    results.
    """
    numeric = float(value)
    if math.isfinite(numeric) and abs(numeric) <= zero_tolerance:
        return 0.0
    return numeric


def format_number(
    value: float,
    *,
    zero_tolerance: float = 1e-10,
    significant_digits: int = 8,
) -> str:
    """Return a compact, stable display string for a finite numerical value.

    Values sufficiently close to zero are displayed as ``0``. Other values are
    rounded for presentation only; the underlying numerical result is unchanged.
    """
    numeric = float(value)
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "NaN"
        return "∞" if numeric > 0 else "−∞"

    if abs(numeric) <= zero_tolerance:
        return "0"

    nearest_integer = round(numeric)
    if abs(numeric - nearest_integer) <= zero_tolerance:
        return str(nearest_integer)

    return f"{numeric:.{significant_digits}g}"
