"""Search subspaces and search directions.

A *subspace* is a small set of orthonormal Cartesian displacement vectors
(rows of ``basis``, each of length 3 n_atoms). The geometry search moves the
molecule only inside this subspace:

    x(z) = x_centre + z @ basis          (z in angstrom)

Three ways to build one:

* :func:`pca_subspace`  principal components of the images of a reaction path;
* :func:`mode_subspace` the softest (or all) internal eigenvectors of a surrogate Hessian,
  for a single geometry with no path;
* :func:`span_subspace` an orthonormal basis of displacement vectors given by the user.

*Search directions* are the eigenvectors of the surrogate Hessian projected into
the subspace (:func:`search_directions`). At the geometry where the Hessian was
computed they are conjugate: the surrogate energy is a sum of independent
one-dimensional parabolas along them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .geometry import kabsch, remove_rigid_motion, n_internal
from .linalg import product
from .units import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL


@dataclass
class Subspace:
    """Orthonormal Cartesian basis (d, 3 n) with the facts about how it was built."""

    basis: np.ndarray                       # (d, 3 n_atoms), orthonormal rows
    reference: np.ndarray                   # (n_atoms, 3) geometry the path was aligned to
    kind: str = "pca"                       # "pca", "modes" or "span"
    explained_variance_ratio: Optional[np.ndarray] = None     # all components of the PCA (not only the kept d)
    singular_values: Optional[np.ndarray] = None
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.basis.shape[0])

    def displacement(self, z: Sequence[float]) -> np.ndarray:
        """Cartesian displacement (n_atoms, 3) for subspace coordinates ``z`` (angstrom)."""
        return (np.asarray(z, dtype=float) @ self.basis).reshape(-1, 3)

    def coordinates(self, displacement: np.ndarray) -> np.ndarray:
        """Subspace coordinates of a Cartesian displacement (its orthogonal projection)."""
        return self.basis @ np.asarray(displacement, dtype=float).reshape(-1)

    def captured_fraction(self, displacement: np.ndarray) -> float:
        """Fraction of the squared length of a displacement that lies inside the subspace (0 to 1)."""
        dx = np.asarray(displacement, dtype=float).reshape(-1)
        total = float(dx @ dx)
        if total == 0.0:
            return 1.0
        z = self.basis @ dx
        return float(z @ z) / total

    def save(self, path: str) -> None:
        np.savez(path, basis=self.basis, reference=self.reference, kind=self.kind,
                 explained_variance_ratio=(self.explained_variance_ratio
                                           if self.explained_variance_ratio is not None else np.array([])),
                 singular_values=(self.singular_values if self.singular_values is not None else np.array([])))

    @staticmethod
    def load(path: str) -> "Subspace":
        with np.load(path, allow_pickle=False) as z:
            evr = z["explained_variance_ratio"]; sv = z["singular_values"]
            return Subspace(basis=z["basis"], reference=z["reference"], kind=str(z["kind"]),
                            explained_variance_ratio=evr if evr.size else None,
                            singular_values=sv if sv.size else None)


def _orthonormalize(rows: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis of the row space, in the order of decreasing singular value."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    _u, s, vt = np.linalg.svd(rows, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((0, rows.shape[1]))
    return vt[s > tol * s[0]]


def _fix_sign(basis: np.ndarray) -> np.ndarray:
    """Make the largest component of every row positive, so that a basis is reproducible."""
    out = basis.copy()
    for i, row in enumerate(out):
        k = int(np.argmax(np.abs(row)))
        if row[k] < 0:
            out[i] = -row
    return out


def align_path(frames: Sequence[np.ndarray], reference_index: int = 0,
               allow_reflection: bool = False) -> List[np.ndarray]:
    """Rigidly superpose every image of a path on the image ``reference_index``."""
    ref = np.asarray(frames[reference_index], dtype=float).reshape(-1, 3)
    out = []
    for f in frames:
        moved, _rmsd, _refl = kabsch(f, ref, allow_reflection=allow_reflection)
        out.append(moved)
    return out


def pca_subspace(frames: Sequence[np.ndarray], d: Optional[int] = None, variance: float = 0.999,
                 reference_index: int = 0, align: bool = True, remove_rigid: bool = True,
                 max_dim: int = 9) -> Subspace:
    """Principal components of the images of a reaction path.

    1. Every image is superposed on image ``reference_index`` (Kabsch, proper rotation).
    2. The images are mean-centred in Cartesian coordinates. No per-coordinate scaling is applied.
    3. The right singular vectors of the centred matrix are the principal components.
    4. Optionally the rigid-body part (translation, rotation at the reference image) is projected
       out of every kept component and the set is orthonormalized again.

    ``d`` fixes the number of components. With ``d=None`` the smallest number is taken whose
    cumulative explained variance reaches ``variance``, limited by ``max_dim``.
    The number of components can never exceed ``n_images - 1`` or the number of internal
    degrees of freedom of the molecule.
    """
    if len(frames) < 2:
        raise ValueError("a path needs at least 2 images, got %d" % len(frames))
    work = align_path(frames, reference_index) if align else [np.asarray(f, float).reshape(-1, 3) for f in frames]
    ref = work[reference_index]
    X = np.array([f.reshape(-1) for f in work])
    Xc = X - X.mean(axis=0)
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    if var.sum() <= 0.0:
        raise ValueError("all images of the path are identical: no principal component exists")
    evr = var / var.sum()
    rank = int(np.sum(s > 1e-8 * s[0]))
    limit = min(rank, n_internal(ref), len(frames) - 1)
    if d is None:
        cum = np.cumsum(evr)
        d_use = int(np.searchsorted(cum, variance - 1e-12) + 1)
        d_use = max(1, min(d_use, limit, max_dim))
    else:
        if d < 1:
            raise ValueError("d must be at least 1")
        if d > limit:
            raise ValueError("d = %d was requested, but this path supports at most %d components "
                             "(%d images, %d internal degrees of freedom)" % (d, limit, len(frames), n_internal(ref)))
        d_use = d
    basis = vt[:d_use]
    info = {"n_images": len(frames), "rank": rank, "limit": limit, "rigid_removed": bool(remove_rigid)}
    if remove_rigid:
        cleaned = remove_rigid_motion(basis, ref)
        info["rigid_fraction"] = [float(1.0 - (c @ c)) for c in cleaned]     # squared length removed from each component
        # Gram-Schmidt in the PCA order keeps component 1 the leading path motion
        basis = _gram_schmidt(cleaned)
        if basis.shape[0] < d_use:
            raise ValueError("after removal of rigid-body motion only %d independent components remain "
                             "(asked for %d)" % (basis.shape[0], d_use))
    return Subspace(basis=_fix_sign(basis), reference=ref.copy(), kind="pca",
                    explained_variance_ratio=evr, singular_values=s, info=info)


def _gram_schmidt(rows: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    out = []
    for r in np.atleast_2d(rows):
        v = r.astype(float).copy()
        for q in out:
            v -= (v @ q) * q
        norm = np.linalg.norm(v)
        if norm > tol:
            out.append(v / norm)
    return np.array(out) if out else np.zeros((0, np.atleast_2d(rows).shape[1]))


def span_subspace(vectors: np.ndarray, reference: np.ndarray, remove_rigid: bool = True) -> Subspace:
    """Subspace spanned by displacement vectors given by the user (rows, flat Cartesian)."""
    V = np.atleast_2d(np.asarray(vectors, dtype=float))
    if remove_rigid:
        V = remove_rigid_motion(V, reference)
    basis = _gram_schmidt(V)
    if basis.shape[0] == 0:
        raise ValueError("the given vectors span no internal displacement")
    return Subspace(basis=_fix_sign(basis), reference=np.asarray(reference, float).reshape(-1, 3).copy(), kind="span")


def hessian_to_kcal_per_A2(hessian_hartree_bohr2: np.ndarray) -> np.ndarray:
    """Convert a Cartesian Hessian from hartree/bohr^2 to kcal/mol/angstrom^2."""
    return np.asarray(hessian_hartree_bohr2, dtype=float) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM ** 2


def mode_subspace(hessian: np.ndarray, coords: np.ndarray, k: Optional[int] = None) -> Subspace:
    """Subspace of the ``k`` softest internal eigenvectors of a Cartesian Hessian (plain Cartesian,
    not mass-weighted). With ``k=None`` all internal degrees of freedom are kept.

    The eigenvectors are ordered by SIGNED eigenvalue, so a negative-curvature direction
    (the reaction coordinate at a saddle) always comes first and is never dropped.
    """
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    H = np.asarray(hessian, dtype=float)
    n3 = X.size
    if H.shape != (n3, n3):
        raise ValueError("Hessian shape %s does not match %d atoms" % (H.shape, len(X)))
    internal = _orthonormalize(remove_rigid_motion(np.eye(n3), X))       # (3n - 6, 3n)
    Hi = product(internal, 0.5 * (H + H.T), internal.T)
    w, V = np.linalg.eigh(Hi)
    order = np.argsort(w)
    n_keep = len(w) if k is None else k
    if n_keep < 1 or n_keep > len(w):
        raise ValueError("k = %s is outside 1..%d" % (k, len(w)))
    basis = (V[:, order[:n_keep]].T) @ internal
    return Subspace(basis=_fix_sign(basis), reference=X.copy(), kind="modes",
                    info={"eigenvalues": [float(v) for v in w[order[:n_keep]]]})


@dataclass
class Directions:
    """Search directions: unit Cartesian vectors with their surrogate curvatures.

    ``vectors`` has shape (d, 3 n). ``curvature[j]`` is the SIGNED force constant
    ``k_j`` of the surrogate along direction j in kcal/mol/A^2, defined by
    ``E = E0 + 0.5 * k_j * t**2``. (Protocol v1 uses ``Lambda_j = k_j / 2``.)
    ``hessian`` is the surrogate Hessian in the basis of these directions; at the
    geometry where it was computed it is diagonal.
    """

    vectors: np.ndarray
    curvature: np.ndarray
    hessian: np.ndarray
    coefficients: np.ndarray        # (number of directions, d): direction j = coefficients[j] @ subspace.basis

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_negative(self) -> int:
        return int(np.sum(self.curvature < 0))

    def lam(self) -> np.ndarray:
        """Protocol-v1 curvatures Lambda_j = k_j / 2 (kcal/mol/A^2)."""
        return self.curvature / 2.0


def project_hessian(hessian_cart: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Hessian in a subspace: ``basis @ H @ basis.T`` (same units as the input)."""
    H = np.asarray(hessian_cart, dtype=float)
    return product(basis, 0.5 * (H + H.T), basis.T)


def highest_image(energies: Sequence[Optional[float]]) -> int:
    """Number of the image with the highest energy on a path: the saddle image of a climbing-image path.

    The image must lie between the two ends. A path without energies, or one whose highest image is an end
    (no barrier along the path), has no such image (ValueError).
    """
    if len(energies) < 3:
        raise ValueError("a path of %d image(s) has no image between its ends" % len(energies))
    if any(e is None for e in energies):
        raise ValueError("the path file gives no energy for every image, so the saddle image cannot be found; "
                         "name it with [geometry] saddle_image")
    top = int(np.argmax(np.asarray(energies, dtype=float)))
    if top in (0, len(energies) - 1):
        raise ValueError("the highest image of the path is an end of the path (image %d): the path has no saddle "
                         "between its ends" % top)
    return top


def path_targets(selection, n_images: int, saddle: Optional[int] = None) -> Dict[int, str]:
    """Which images of a path are optimized, and as what: ``{image number: kind}``.

    ``selection`` is ``"barrier"`` (the first image and the saddle image: 2 searches), ``"mep"`` (every image
    of the path), or a list of image numbers (in the order given). The kinds follow from the place on the
    path: the first and the last image are minima, the image ``saddle`` is a saddle, and every other image is
    a path image (kind ``"image"``: searched across the path only). ``saddle`` may be None for a list or for
    ``"mep"`` on a path that has no saddle; ``"barrier"`` needs it.
    """
    if n_images < 2:
        raise ValueError("a path needs at least 2 images, got %d" % n_images)
    if saddle is not None and not 0 < saddle < n_images - 1:
        raise ValueError("the saddle image must lie between the two ends of the path: image %d is not one of "
                         "the images 1 to %d" % (saddle, n_images - 2))
    if selection == "barrier":
        if saddle is None:
            raise ValueError("a barrier needs the saddle image of the path")
        numbers = [0, saddle]
    elif selection == "mep":
        numbers = list(range(n_images))
    else:
        numbers = [int(i) for i in selection]
        bad = [i for i in numbers if not 0 <= i < n_images]
        if bad:
            raise ValueError("image(s) %s do not exist; the path holds the images 0 to %d"
                             % (", ".join(str(i) for i in bad), n_images - 1))
        if len(set(numbers)) != len(numbers):
            raise ValueError("an image is named twice in the list of targets")
    out: Dict[int, str] = {}
    for i in numbers:
        out[i] = "minimum" if i in (0, n_images - 1) else ("saddle" if i == saddle else "image")
    return out


def path_tangent(frames: Sequence[np.ndarray], index: int, at: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit tangent of a path at its image ``index``: the difference of the two neighbour images.

    The two neighbours are first superposed (Kabsch, proper rotation) on the geometry ``at`` (default: the
    image itself), so that a rigid motion between the images is no part of the tangent. Returns a flat
    Cartesian vector of length 3 n. The first and the last image have only one neighbour: no tangent.
    """
    n = len(frames)
    if not 0 < index < n - 1:
        raise ValueError("image %d has no two neighbours on a path of %d images (the first image is 0): "
                         "a path tangent needs the image before and the image after" % (index, n))
    x = np.asarray(frames[index] if at is None else at, dtype=float).reshape(-1, 3)
    before = kabsch(np.asarray(frames[index - 1], dtype=float).reshape(-1, 3), x)[0]
    after = kabsch(np.asarray(frames[index + 1], dtype=float).reshape(-1, 3), x)[0]
    t = (after - before).reshape(-1)
    norm = float(np.linalg.norm(t))
    if norm == 0.0:
        raise ValueError("the images %d and %d of the path are identical: image %d has no tangent"
                         % (index - 1, index + 1, index))
    return t / norm


def search_directions(subspace_hessian: np.ndarray, subspace: Subspace,
                      exclude: Optional[np.ndarray] = None) -> Directions:
    """Diagonalize a surrogate Hessian given in the subspace basis (d x d, kcal/mol/A^2).

    Directions are ordered by SIGNED curvature, lowest first. A negative value
    marks a direction along which the surrogate has a maximum (a saddle).

    ``exclude`` is a Cartesian displacement, for example the tangent of a path at one of its images
    (:func:`path_tangent`). With it the search is restricted to the part of the subspace that is orthogonal
    to this displacement: the Hessian is projected there and d - 1 directions come back, each of them
    orthogonal to ``exclude``. ``coefficients`` then has the shape (d - 1, d). A displacement that has no
    part inside the subspace cannot be excluded (ValueError): the whole subspace is orthogonal to it already.
    """
    K = np.asarray(subspace_hessian, dtype=float)
    d = subspace.dim
    if K.shape != (d, d):
        raise ValueError("subspace Hessian has shape %s, expected (%d, %d)" % (K.shape, d, d))
    if exclude is None:
        w, V = np.linalg.eigh(0.5 * (K + K.T))
        order = np.argsort(w)
        coeff = V[:, order].T
    else:
        dx = np.asarray(exclude, dtype=float).reshape(-1)
        c = subspace.basis @ dx                         # the part of the displacement inside the subspace
        size = float(np.linalg.norm(c))
        if size <= 1e-6 * float(np.linalg.norm(dx)) or size == 0.0:
            raise ValueError("the displacement to exclude has no part inside the subspace")
        if d < 2:
            raise ValueError("the subspace has 1 direction: none is left after one is excluded")
        c = c / size
        # rows of Q: an orthonormal basis of the coordinates orthogonal to c (eigenvalue 1 of the projector)
        w0, V0 = np.linalg.eigh(np.eye(d) - np.outer(c, c))
        Q = V0[:, np.argsort(w0)[1:]].T
        w, V = np.linalg.eigh(Q @ (0.5 * (K + K.T)) @ Q.T)
        order = np.argsort(w)
        coeff = V[:, order].T @ Q
    vectors = coeff @ subspace.basis
    for i in range(vectors.shape[0]):              # reproducible signs
        k = int(np.argmax(np.abs(vectors[i])))
        if vectors[i, k] < 0:
            vectors[i] = -vectors[i]; coeff[i] = -coeff[i]
    return Directions(vectors=vectors, curvature=w[order].copy(), hessian=np.diag(w[order]), coefficients=coeff)


def hessian_in_directions(hessian_cart: np.ndarray, directions: Directions) -> np.ndarray:
    """A Cartesian Hessian (computed at any geometry) expressed in the basis of fixed search directions."""
    return project_hessian(hessian_cart, directions.vectors)


def max_coupling(K: np.ndarray) -> float:
    """Largest normalized off-diagonal element ``|K_ij| / sqrt(|K_ii K_jj|)`` of a Hessian in the
    direction basis. It is near 0 where the directions are conjugate."""
    K = np.asarray(K, dtype=float)
    d = K.shape[0]
    best = 0.0
    for i in range(d):
        for j in range(d):
            if i != j and K[i, i] != 0 and K[j, j] != 0:
                best = max(best, abs(K[i, j]) / np.sqrt(abs(K[i, i] * K[j, j])))
    return float(best)
