"""Shared math_verify grading helper for the eval + re-grade scripts.

math_verify (HF) recovers algebraically/representationally-equivalent answers the
homegrown normalize+sympy grader drops: fraction<->decimal, unsimplified fractions,
LaTeX vs plain, tuples/intervals/sets, matrices. Used both live (eval_*.py grade())
and offline (regrade_math_verify.py over saved generations).

Every entry point is defensive: returns None when math_verify is unavailable or
errors, so callers fall back to the legacy grader and an eval never crashes on it.
"""
from __future__ import annotations

import os
import re

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


def mv_available() -> bool:
    try:
        import math_verify  # noqa: F401
        return True
    except Exception:
        return False


def use_mv() -> bool:
    """math_verify grading is ON by default; set USE_MATH_VERIFY=0 to disable."""
    return os.environ.get("USE_MATH_VERIFY", "1") == "1" and mv_available()


def _parse_gold(gold_str: str):
    """Parse a gold answer trying plain then LaTeX-anchored extraction.

    MATH-500 golds are raw (e.g. "5", "\\frac{3}{4}", "(2,3)", "[0,1)"). Plain
    parse handles scalars/fractions; wrapping in $...$ recovers tuples/intervals
    that need a LaTeX anchor. Return the first non-empty parse.
    """
    from math_verify import parse
    # If the gold carries LaTeX (backslash / ^ / _), anchor it first — otherwise the
    # anchorless Expr parser grabs a leading scalar (e.g. "2\sqrt{2}" -> just "2").
    has_latex = any(c in gold_str for c in ("\\", "^", "_"))
    cands = (f"${gold_str}$", gold_str) if has_latex else (gold_str, f"${gold_str}$")
    for cand in cands:
        try:
            g = parse(cand)
            if g:
                return g
        except Exception:
            continue
    return None


def mv_grade_text(text: str, gold, kind: str = "math"):
    """Grade a completion `text` against `gold` with math_verify.

    kind="math": gold is a MATH-500 answer string (latex/numeric/tuple/...).
    kind="aime": gold is an integer in [0,999].

    Returns True/False, or None if math_verify is unavailable or errors out
    (caller should fall back to the legacy grader).
    """
    if not mv_available():
        return None
    try:
        from math_verify import parse, verify
        gold_str = str(gold)
        g = _parse_gold(gold_str)
        if not g:
            return None
        # Parse the model side from the FULL text so math_verify's extraction
        # (last boxed, priority 50) drives correctness — that's where the boost is.
        p = parse(text)
        if not p:
            return None
        return bool(verify(g, p))
    except Exception:
        return None


def extract_last_boxed(text: str):
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None
