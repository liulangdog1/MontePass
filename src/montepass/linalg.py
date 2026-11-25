"""Small linear-algebra helpers."""
from __future__ import annotations

import numpy as np


def product(*factors: np.ndarray) -> np.ndarray:
    """Matrix product of finite arrays, from left to right.

    Some linear-algebra libraries raise floating-point warnings ("divide by zero encountered in
    matmul") in a matrix product whose result is correct. Such a warning would stop a program
    that turns warnings into errors. The warnings are switched off here and the result is
    checked instead: a value that is not a finite number raises FloatingPointError.
    """
    out = np.asarray(factors[0], dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for f in factors[1:]:
            out = np.matmul(out, np.asarray(f, dtype=float))
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("a matrix product gave a value that is not a finite number")
    return out
