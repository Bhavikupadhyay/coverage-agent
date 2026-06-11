"""Basic statistics utilities."""
from __future__ import annotations


def clamp(value: float, lo: float, hi: float) -> float:
    """Returns value clamped to [lo, hi].

    The initial test suite only calls clamp(5, 0, 10) -> 5, so the
    lo-clamp and hi-clamp branches are not hit.
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def safe_divide(numerator: float, denominator: float) -> float:
    """Divides numerator by denominator, returning 0.0 on zero denominator.

    The initial test suite only calls safe_divide(10, 2), so the
    zero-denominator branch is not hit.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def letter_grade(score: float) -> str:
    """Maps a numeric score [0, 100] to a letter grade.

    The initial test suite only calls letter_grade(95), so only the
    >= 90 branch is hit.
    """
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "C"


def normalize(values: list[float]) -> list[float]:
    """Scales values to [0, 1] range. Returns empty list on empty input.

    The initial test suite only calls normalize([]), so the non-empty
    and uniform-values branches are not hit.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
