"""Surrogate backed by an analytic energy function (model surfaces; tests and demonstrations)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .. import model
from ..units import HARTREE_TO_KCAL
from .base import Surrogate, SurrogateResult


class AnalyticSurrogate(Surrogate):
    """``surface(coords)`` returns the surrogate energy in kcal/mol. Results are ready at once.
    ``delay`` (in calls of :meth:`result`) can imitate a calculation that takes time."""

    def __init__(self, surface: Callable[[np.ndarray], float], delay: int = 0):
        self.surface = surface
        self.delay = int(delay)
        self._store: Dict[str, SurrogateResult] = {}
        self._wait: Dict[str, int] = {}
        self._zmin: Dict[str, np.ndarray] = {}
        self._scans: Dict[str, np.ndarray] = {}

    def request(self, tag: str, coords: np.ndarray, hessian: bool = True, gradient: bool = True) -> None:
        if tag in self._store:
            return
        X = np.asarray(coords, dtype=float).reshape(-1, 3)
        res = SurrogateResult(energy=self.surface(X) / HARTREE_TO_KCAL)
        if gradient:
            res.gradient = model.gradient(self.surface, X)
        if hessian:
            res.hessian = model.hessian(self.surface, X)
        self._store[tag] = res
        self._wait[tag] = self.delay

    def result(self, tag: str) -> Optional[SurrogateResult]:
        if tag not in self._store:
            return None
        if self._wait[tag] > 0:
            self._wait[tag] -= 1
            return None
        return self._store[tag]

    def request_scan(self, tag, geometries) -> None:
        if tag not in self._scans:
            self._scans[tag] = np.array([self.surface(np.asarray(x, dtype=float).reshape(-1, 3)) for x in geometries]) / HARTREE_TO_KCAL

    def scan(self, tag):
        return self._scans.get(tag)

    def request_corrected_minimum(self, tag, centre, vectors, g_delta, k_delta, bounds) -> bool:
        if tag in self._zmin:
            return True
        from scipy.optimize import minimize

        c = np.asarray(centre, dtype=float).reshape(-1, 3)
        V = np.asarray(vectors, dtype=float)
        gd = np.asarray(g_delta, dtype=float)
        kd = np.zeros(len(gd)) if k_delta is None else np.asarray(k_delta, dtype=float)
        e0 = self.surface(c)

        def f(z):
            return self.surface(c + (z @ V).reshape(-1, 3)) - e0 + gd @ z + 0.5 * np.sum(kd * z * z)

        b = np.asarray(bounds, dtype=float)
        res = minimize(f, np.zeros(len(gd)), method="L-BFGS-B", bounds=[(-x, x) for x in b],
                       options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 500})
        self._zmin[tag] = res.x
        return True

    def corrected_minimum(self, tag):
        return self._zmin.get(tag)
