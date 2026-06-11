"""Initial test suite — deliberately leaves several branches uncovered."""
from mathlib.stats import clamp, safe_divide, letter_grade, normalize


def test_clamp_in_range():
    assert clamp(5, 0, 10) == 5


def test_safe_divide_normal():
    assert safe_divide(10, 2) == 5.0


def test_letter_grade_a():
    assert letter_grade(95) == "A"


def test_normalize_empty():
    assert normalize([]) == []
