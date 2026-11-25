"""The surrogate interface used by the geometry search.

The search asks the surrogate for its gradient and Hessian at a scan centre.
A real surrogate needs a batch job for that, so the interface is asynchronous:

    surrogate.request(tag, coords)      # start the calculation; does nothing if the tag exists
    res = surrogate.result(tag)         # None while it is still running

Units of the results: energy in hartree, gradient in kcal/mol/angstrom with
shape (n_atoms, 3), Hessian in kcal/mol/angstrom^2 with shape (3 n, 3 n).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SurrogateResult:
    energy: Optional[float] = None          # hartree
    gradient: Optional[np.ndarray] = None   # (n_atoms, 3), kcal/mol/angstrom
    hessian: Optional[np.ndarray] = None    # (3 n, 3 n), kcal/mol/angstrom^2 (full Cartesian Hessian), or None
    hessian_directions: Optional[np.ndarray] = None    # (d, d), kcal/mol/angstrom^2: Hessian in the basis of the
                                            # search directions. A surrogate may give this in place of ``hessian``.
    failed: bool = False
    message: str = ""


class Surrogate:
    def request(self, tag: str, coords: np.ndarray, hessian: bool = True, gradient: bool = True) -> None:
        raise NotImplementedError

    def result(self, tag: str) -> Optional[SurrogateResult]:
        """The finished result, or None if the calculation has not finished."""
        raise NotImplementedError

    # ---- surrogate energies of a list of geometries (used by the classic line search to size its windows)
    def request_scan(self, tag: str, geometries: Sequence[np.ndarray]) -> None:
        raise NotImplementedError

    def scan(self, tag: str) -> Optional[np.ndarray]:
        """Energies in hartree, one per geometry of the request, or None if not finished."""
        raise NotImplementedError

    # ---- optional service: minimum of the surrogate plus a correction measured by DMC
    def request_corrected_minimum(self, tag: str, centre: np.ndarray, vectors: np.ndarray,
                                  g_delta: np.ndarray, k_delta: Optional[np.ndarray], bounds: np.ndarray) -> bool:
        """Start the search for the minimum, inside the box ``|z_j| <= bounds[j]``, of

            F(z) = E_sur(centre + z @ vectors) + g_delta @ z + 0.5 * sum_j k_delta[j] * z_j**2

        ``g_delta`` (kcal/mol/A) and ``k_delta`` (kcal/mol/A^2) are the differences "DMC minus
        surrogate" of slope and curvature measured at the centre. Returns False if this surrogate
        does not offer the service; the search then takes a Newton step."""
        return False

    def corrected_minimum(self, tag: str) -> Optional[np.ndarray]:
        """Subspace coordinates z of the minimum (angstrom), or None while the search runs."""
        return None
