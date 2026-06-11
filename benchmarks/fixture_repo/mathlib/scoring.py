"""Scoring utilities — added in the diff branch."""
from __future__ import annotations


def percentage(correct: int, total: int) -> float:
    """Returns the percentage score (0-100). Returns 0.0 when total is zero."""
    if total == 0:
        return 0.0
    return correct / total * 100


def pass_fail(score: float, threshold: float = 60.0) -> str:
    """Returns 'pass' if score >= threshold, else 'fail'."""
    if score >= threshold:
        return "pass"
    return "fail"


def weighted_average(scores: list[float], weights: list[float]) -> float:
    """Computes a weighted average of scores. Returns 0.0 on empty input."""
    if not scores or not weights or len(scores) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight
