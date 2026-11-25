"""Point-group symmetry of a geometry and what it says about the search directions.

At a symmetric geometry the energy gradient is invariant under every symmetry
operation. So the slope along a direction that has no totally symmetric part is
exactly zero, and no DMC points are needed to measure it. This module finds the
symmetry operations of a geometry and sorts the search directions into

    "symmetric"  totally symmetric: the slope can be different from zero, scan both sides
    "odd"        some operation maps the direction to minus itself: the slope is zero and
                 E(+t) = E(-t), so one point is enough to measure the curvature
    "zero"       no totally symmetric part, but no operation reverses it (degenerate
                 pairs): the slope is zero; the curvature is taken from the surrogate
    "mixed"      neither: treated as an ordinary direction

A displacement along an "odd" or "zero" direction would break the symmetry. The
search does not need to move along them unless the measured curvature shows that
the symmetric geometry is unstable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import ATOMIC_NUMBER

Operation = Tuple[np.ndarray, np.ndarray]        # (3 x 3 orthogonal matrix, permutation: atom i goes to perm[i])


def _permutation(symbols: Sequence[str], X: np.ndarray, Xt: np.ndarray, tol: float) -> Optional[np.ndarray]:
    """Permutation p with Xt[i] = X[p[i]] (same element), or None if the transformed geometry is not the molecule."""
    n = len(X)
    perm = -np.ones(n, dtype=int)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        d = np.linalg.norm(X - Xt[i], axis=1)
        j = int(np.argmin(d + np.where(used, 1e9, 0.0)))
        if d[j] > tol or symbols[j] != symbols[i] or used[j]:
            return None
        perm[i] = j; used[j] = True
    return perm


def find_operations(symbols: Sequence[str], coords: np.ndarray, tol: float = 1e-3, max_order: int = 120) -> List[Operation]:
    """Symmetry operations of a molecule (the identity is not included).

    Candidates are the inversion, and reflections and two-fold rotations about axes built from the
    atom positions. The set is then closed under multiplication, which adds the rotations of higher
    order. Distances are compared with ``tol`` (angstrom).
    """
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = len(X)
    c = X.mean(axis=0)
    X0 = X - c
    z = np.array([ATOMIC_NUMBER[s] for s in symbols], dtype=float)
    axes: List[np.ndarray] = []

    def add(v):
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            axes.append(v / nv)

    _w, vec = np.linalg.eigh((X0 * z[:, None]).T @ X0)
    for k in range(3):
        add(vec[:, k])
    for i in range(n):
        add(X0[i])
    if n <= 40:
        for i in range(n):
            for j in range(i + 1, n):
                if symbols[i] == symbols[j]:
                    add(X0[i] - X0[j]); add(X0[i] + X0[j])
                add(np.cross(X0[i], X0[j]))
    # remove duplicate axes (a and -a are the same axis)
    unique: List[np.ndarray] = []
    for a in axes:
        if not any(abs(abs(a @ b) - 1.0) < 1e-6 for b in unique):
            unique.append(a)
    mats = [-np.eye(3)]
    for a in unique:
        mats.append(np.eye(3) - 2.0 * np.outer(a, a))          # reflection through the plane normal to a
        mats.append(2.0 * np.outer(a, a) - np.eye(3))          # rotation by 180 degrees about a
    # An operation is identified by what it does to the atoms (the permutation) and by whether it is a
    # rotation or a reflection (the sign of its determinant). Two matrices that agree in both are the same
    # operation of the molecule, also when their numbers differ a little.
    ops: List[Operation] = []
    keys = {(tuple(range(n)), 1)}

    def consider(Mx, tolerance) -> bool:
        p = _permutation(symbols, X0, X0 @ Mx.T, tolerance)
        if p is None:
            return False
        key = (tuple(int(v) for v in p), 1 if np.linalg.det(Mx) > 0 else -1)
        if key in keys:
            return False
        keys.add(key); ops.append((Mx, p))
        return True

    for Mx in mats:
        consider(Mx, tol)
    # closure: products of operations are operations (this adds rotations of order 3 and higher)
    grew = True
    while grew and len(ops) < max_order:
        grew = False
        for A, _pa in list(ops):
            for B, _pb in list(ops):
                if consider(A @ B, 3.0 * tol):
                    grew = True
                if len(ops) >= max_order:
                    break
            if len(ops) >= max_order:
                break
    return ops


def apply(op: Operation, vector: np.ndarray) -> np.ndarray:
    """Image of a flat Cartesian displacement vector (length 3 n) under a symmetry operation."""
    Mx, perm = op
    U = np.asarray(vector, dtype=float).reshape(-1, 3)
    out = np.zeros_like(U)
    out[perm] = U @ Mx.T
    return out.reshape(-1)


def symmetric_part(ops: Sequence[Operation], vector: np.ndarray) -> np.ndarray:
    """Totally symmetric part of a displacement: its average over the group."""
    v = np.asarray(vector, dtype=float).reshape(-1)
    acc = v.copy()
    for op in ops:
        acc = acc + apply(op, v)
    return acc / (len(ops) + 1.0)


def classify_directions(vectors: np.ndarray, symbols: Sequence[str], coords: np.ndarray, tol: float = 1e-3,
                        threshold: float = 1e-3) -> Tuple[List[str], int]:
    """Class of every direction ("symmetric", "odd", "zero" or "mixed") and the order of the group found."""
    ops = find_operations(symbols, coords, tol)
    V = np.atleast_2d(np.asarray(vectors, dtype=float))
    if not ops:
        return ["symmetric" for _ in V], 1        # no symmetry: every direction can have a slope
    out = []
    for v in V:
        ps = np.linalg.norm(symmetric_part(ops, v))
        if ps > 1.0 - threshold:
            out.append("symmetric")
        elif ps < threshold:
            if any(np.linalg.norm(apply(op, v) + v) < 10.0 * threshold for op in ops):
                out.append("odd")
            else:
                out.append("zero")
        else:
            out.append("mixed")
    return out, len(ops) + 1
