"""A surrogate that is moved rigidly in space: right shape, wrong place.

To test a search that starts from a wrong geometry, the surrogate (DFT) surface is
moved by a fixed displacement ``shift``:

    E_shifted(x) = E_surrogate(x - shift)

The shifted surrogate has its minimum at (true surrogate minimum + shift), with the
same gradient field and the same Hessian. The DMC energies are not changed. A search
that starts at the minimum of the shifted surrogate therefore sees what it would
see with a functional whose minimum is ``shift`` away from the right place.

Two wrappers do this for any surrogate and any engine.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from .engine.base import ABSENT, DONE, FAILED, RUNNING, STOPPED, Engine, PointData, PointSpec
from .surrogate.base import Surrogate, SurrogateResult


class ShiftedSurrogate(Surrogate):
    def __init__(self, inner: Surrogate, shift: np.ndarray):
        self.inner = inner
        self.shift = np.asarray(shift, dtype=float).reshape(-1, 3)

    def request(self, tag: str, coords: np.ndarray, hessian: bool = True, gradient: bool = True) -> None:
        self.inner.request(tag, np.asarray(coords, dtype=float).reshape(-1, 3) - self.shift, hessian=hessian, gradient=gradient)

    def result(self, tag: str) -> Optional[SurrogateResult]:
        return self.inner.result(tag)

    def request_scan(self, tag: str, geometries: Sequence[np.ndarray]) -> None:
        self.inner.request_scan(tag, [np.asarray(x, dtype=float).reshape(-1, 3) - self.shift for x in geometries])

    def scan(self, tag: str):
        return self.inner.scan(tag)

    def request_corrected_minimum(self, tag, centre, vectors, g_delta, k_delta, bounds) -> bool:
        return self.inner.request_corrected_minimum(tag, np.asarray(centre, dtype=float).reshape(-1, 3) - self.shift,
                                                    vectors, g_delta, k_delta, bounds)

    def corrected_minimum(self, tag):
        return self.inner.corrected_minimum(tag)


class ShiftedEngine(Engine):
    """Gives every point the energy of the moved surrogate. The DMC run of the point is not changed.

    Two ways, depending on the inner engine:

    * ``source = None``: the inner engine computes the surrogate energy itself at the geometry
      ``PointSpec.settings["surrogate_coords"]`` (the mock engine does this).
    * ``source`` is a surrogate object with ``request_scan`` / ``scan``: for every point a
      one-geometry scan with the tag ``<prefix>/<key>`` is requested at the moved geometry. The
      point is reported as RUNNING until that energy exists, and then with that energy.
    """

    def __init__(self, inner: Engine, shift: np.ndarray, source: Optional[Surrogate] = None, prefix: str = "moved"):
        self.inner = inner
        self.shift = np.asarray(shift, dtype=float).reshape(-1, 3)
        self.source = source
        self.prefix = prefix

    def _tag(self, key: str) -> str:
        return "%s/%s" % (self.prefix, key)

    def submit(self, spec: PointSpec) -> None:
        coords = np.asarray(spec.coords, dtype=float).reshape(-1, 3)
        moved = coords - self.shift
        if self.source is None:
            self.inner.submit(replace(spec, settings=dict(spec.settings, surrogate_coords=moved.tolist())))
            return
        self.source.request_scan(self._tag(str(spec.key)), [moved])
        self.inner.submit(spec)

    def read(self, key: str) -> PointData:
        data = self.inner.read(key)
        if self.source is None or data.state == ABSENT:
            return data
        if data.state == FAILED:
            # A failed point can keep blocks (a run that ended at its time limit). Its surrogate energy must be
            # that of the moved surrogate too, or no surrogate energy at all.
            try:
                energies = self.source.scan(self._tag(str(key)))
            except RuntimeError:
                energies = None
            e = None if energies is None else float(np.asarray(energies).reshape(-1)[0])
            return replace(data, e_surrogate=e)
        # The inner engine reports the surrogate energy of the true geometry. It must never reach the search,
        # in any state of the point: a search reads the blocks of a point while the point still runs.
        energies = self.source.scan(self._tag(str(key)))          # raises RuntimeError if the calculation failed
        if energies is None:
            # no blocks are shown before the energy of the moved surrogate exists
            state = RUNNING if data.state in (DONE, STOPPED) else data.state
            return replace(data, state=state, e_surrogate=None, energy=data.energy[:0], weight=data.weight[:0],
                           message="waiting for the energy of the moved surrogate")
        return replace(data, e_surrogate=float(np.asarray(energies).reshape(-1)[0]))

    def stop(self, key: str) -> None:
        self.inner.stop(key)

    def advance(self, *args, **kwargs) -> None:
        self.inner.advance(*args, **kwargs)

    def cost(self, key: str) -> float:
        return self.inner.cost(key)

    def __getattr__(self, name):
        return getattr(self.inner, name)
