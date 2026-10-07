#!/usr/bin/env python3
"""Regression checks for shared MATH/AIME answer grading."""

from eval_math500 import grade, normalize_for_compare


assert normalize_for_compare(42) == "42"
assert grade("42", 42)
assert grade(r"\frac{1}{2}", r"\frac{1}{2}")
assert not grade(None, 42)

print("MATH_GRADING PASS")
