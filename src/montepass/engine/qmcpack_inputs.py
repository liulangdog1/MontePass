"""Input files of the QMC chain for one geometry: PySCF, convert4qmc, QMCPACK.

The chain of one geometry (one "cell") is

    python scf.py                                  -> scf.h5, energy.dat, system.json
    convert4qmc -prefix c4q -orbitals scf.h5       -> c4q.structure.xml, c4q.wfj.xml (checks the orbital file;
                                                      QMCPACK reads the orbitals from scf.h5 itself)
    qmcpack opt.in.xml                             -> opt.sNNN.opt.xml, opt.sNNN.scalar.dat (Jastrow optimization)
    qmcpack dmc.in.xml                             -> dmc.s000.scalar.dat (VMC), dmc.s001.scalar.dat (DMC)

This module only writes and reads text. It can also be run as a script inside the
batch job of a cell (see :func:`main`), because the DMC input needs the result
of the optimization run.

Trial wave function: one Slater determinant of Kohn-Sham orbitals, a one-body
and a two-body B-spline Jastrow factor, pseudopotentials for all atoms.

Atom order. QMCPACK lists the ions grouped by element, while the orbital file
keeps the atom order of the SCF. The two orders agree only if the atoms given
to the SCF are already grouped by element. :func:`species_order` returns the
permutation that groups them, and :func:`scf_script` refuses any other order.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import ATOMIC_NUMBER, normalize_symbol
from ..units import ANGSTROM_TO_BOHR

# atomic masses in atomic mass units (ions are fixed, QMCPACK only records the value)
_MASS_AMU = {"H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811, "C": 12.0107, "N": 14.0067,
             "O": 15.9994, "F": 18.9984032, "Ne": 20.1797, "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815386,
             "Si": 28.0855, "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948}
_AMU_TO_ME = 1822.888486209


# ----------------------------------------------------------------------------- atom order
def species_order(symbols: Sequence[str]) -> List[int]:
    """Permutation that groups the atoms by element. Elements keep the order of their first
    appearance and atoms keep their order inside an element. ``[symbols[i] for i in perm]`` is grouped."""
    syms = [normalize_symbol(s) for s in symbols]
    first = []
    for s in syms:
        if s not in first:
            first.append(s)
    return [i for el in first for i, s in enumerate(syms) if s == el]


def is_grouped(symbols: Sequence[str]) -> bool:
    return species_order(symbols) == list(range(len(symbols)))


# ----------------------------------------------------------------------------- PySCF
def scf_script(symbols: Sequence[str], coords: np.ndarray, charge: int = 0, spin: int = 0,
               xc: str = "b3lyp", basis: str = "ccecpccpvtz", ecp: str = "ccecp", grid_level: int = 3,
               conv_tol: float = 1e-10, threads: int = 1, gradient: bool = False, rhf_energy: bool = False) -> str:
    """Text of the PySCF script of one geometry.

    ``spin`` is the number of unpaired electrons (PySCF convention, 2S). The script writes
    ``energy.dat`` (total energy in hartree), ``system.json`` (the facts the QMCPACK inputs need),
    ``scf.h5`` (orbitals for convert4qmc) and, with ``gradient=True``, ``gradient.dat``
    (nuclear gradient in hartree/bohr, one atom per line, in the atom order of the script).
    With ``rhf_energy=True`` (closed shells only) the script also computes the restricted Hartree-Fock
    energy with the same basis and pseudopotentials and stores it as ``e_rhf`` in ``system.json``:
    another program that claims to describe the same molecule must find the same number.
    """
    syms = [normalize_symbol(s) for s in symbols]
    if not is_grouped(syms):
        raise ValueError("the atoms must be grouped by element before the SCF script is written "
                         "(use species_order); got %s" % " ".join(syms))
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(X) != len(syms):
        raise ValueError("%d symbols and %d coordinate rows" % (len(syms), len(X)))
    atoms = "\n".join("    %-2s %18.10f %18.10f %18.10f" % (s, r[0], r[1], r[2]) for s, r in zip(syms, X))
    method = "RKS" if spin == 0 else "UKS"
    lines = [
        "#!/usr/bin/env python3",
        '"""Kohn-Sham calculation of one geometry: trial orbitals for QMC and the surrogate energy."""',
        "import os",
        "os.environ['OMP_NUM_THREADS'] = '%d'" % threads,
        "os.environ['MKL_NUM_THREADS'] = '%d'" % threads,
        "os.environ['OPENBLAS_NUM_THREADS'] = '%d'" % threads,
        "import json",
        "import numpy as np",
        "from pyscf import gto, dft, lib",
        "lib.num_threads(%d)" % threads,
        "",
        "mol = gto.Mole()",
        "mol.atom = '''",
        atoms,
        "'''",
        "mol.unit = 'A'",
        "mol.basis = %r" % basis,
        "mol.ecp = %r" % ecp,
        "mol.charge = %d" % charge,
        "mol.spin = %d" % spin,
        "mol.symmetry = False",
        "mol.build()",
        "",
        "mf = dft.%s(mol)" % method,
        "mf.xc = %r" % xc,
        "mf.grids.level = %d" % grid_level,
        "mf.conv_tol = %g" % conv_tol,
        "mf.max_cycle = 300",
        "e_scf = mf.kernel()",
        "assert mf.converged, 'SCF not converged'",
        "np.savetxt('energy.dat', [e_scf, 0.0])",
        "",
        "# Energy of the Kohn-Sham determinant with the true Hamiltonian (the Hartree-Fock energy expression).",
        "# A VMC run with no Jastrow factor must reproduce this number: it tests the whole conversion to QMCPACK.",
        "from pyscf import scf as _scf",
        "e_det = float(_scf.%s(mol).energy_tot(dm=mf.make_rdm1()))" % ("RHF" if spin == 0 else "UHF"),
        "nup, ndn = mol.nelec",
        "json.dump({'symbols': [mol.atom_symbol(i) for i in range(mol.natm)],",
        "           'coords_bohr': mol.atom_coords().tolist(),",
        "           'zeff': [int(z) for z in mol.atom_charges()],",
        "           'nup': int(nup), 'ndn': int(ndn), 'nao': int(mol.nao_nr()),",
        "           'energy': float(e_scf), 'e_determinant': e_det, 'xc': %r, 'basis': %r, 'ecp': %r," % (xc, basis, ecp),
        "           'restricted': %s}, open('system.json', 'w'), indent=1)" % ("True" if spin == 0 else "False"),
    ]
    if rhf_energy:
        if spin != 0:
            raise ValueError("rhf_energy needs a closed-shell molecule (spin = 0)")
        lines += ["",
                  "# Restricted Hartree-Fock energy with the same basis and pseudopotentials. The Hartree-Fock energy of",
                  "# Quantum Package for this geometry must agree with it.",
                  "mf_rhf = _scf.RHF(mol)",
                  "mf_rhf.conv_tol = 1e-10",
                  "mf_rhf.max_cycle = 300",
                  "e_rhf = float(mf_rhf.kernel())",
                  "assert mf_rhf.converged, 'RHF not converged'",
                  "record = json.load(open('system.json'))",
                  "record['e_rhf'] = e_rhf",
                  "json.dump(record, open('system.json', 'w'), indent=1)"]
    if gradient:
        lines += ["", "grad = mf.nuc_grad_method().kernel()        # hartree/bohr", "np.savetxt('gradient.dat', grad)"]
    lines += ["", "from PyscfToQmcpack import savetoqmcpack", "savetoqmcpack(mol, mf, 'scf')", "print('SCF_DONE')", ""]
    return "\n".join(lines)


CONVERT_COMMAND = "convert4qmc -prefix c4q -orbitals scf.h5"


# ----------------------------------------------------------------------------- QMCPACK input pieces
def load_system(path: str = "system.json") -> dict:
    with open(path) as fh:
        s = json.load(fh)
    s["symbols"] = [normalize_symbol(x) for x in s["symbols"]]
    if not is_grouped(s["symbols"]):
        raise ValueError("system.json: atoms are not grouped by element: %s" % " ".join(s["symbols"]))
    return s


def _species(system: dict) -> List[str]:
    out = []
    for s in system["symbols"]:
        if s not in out:
            out.append(s)
    return out


def _particlesets(system: dict) -> str:
    out = ['      <simulationcell>', '         <parameter name="bconds">', '            n n n',
           '         </parameter>', '      </simulationcell>',
           '      <particleset name="e" random="yes" randomsrc="ion0">',
           '         <group name="u" size="%d" mass="1.0">' % system["nup"],
           '            <parameter name="charge"              >    -1                    </parameter>',
           '            <parameter name="mass"                >    1.0                   </parameter>',
           '         </group>']
    if system["ndn"] > 0:
        out += ['         <group name="d" size="%d" mass="1.0">' % system["ndn"],
                '            <parameter name="charge"              >    -1                    </parameter>',
                '            <parameter name="mass"                >    1.0                   </parameter>',
                '         </group>']
    out += ['      </particleset>', '      <particleset name="ion0">']
    syms = system["symbols"]; X = np.asarray(system["coords_bohr"], dtype=float)
    for el in _species(system):
        idx = [i for i, s in enumerate(syms) if s == el]
        zeff = {system["zeff"][i] for i in idx}
        if len(zeff) != 1:
            raise ValueError("atoms of element %s have different effective charges %s" % (el, sorted(zeff)))
        z = zeff.pop()
        mass = _MASS_AMU.get(el, 2.0 * ATOMIC_NUMBER[el]) * _AMU_TO_ME
        out += ['         <group name="%s" size="%d" mass="%.10f">' % (el, len(idx), mass),
                '            <parameter name="charge"              >    %d                     </parameter>' % z,
                '            <parameter name="valence"             >    %d                     </parameter>' % z,
                '            <parameter name="atomicnumber"        >    %d                     </parameter>' % ATOMIC_NUMBER[el],
                '            <parameter name="mass"                >    %.10f            </parameter>' % mass,
                '            <attrib name="position" datatype="posArray" condition="0">']
        out += ['               %18.10f %18.10f %18.10f' % tuple(X[i]) for i in idx]
        out += ['            </attrib>', '         </group>']
    out += ['      </particleset>']
    return "\n".join(out)


def jastrow_ids(system: dict) -> List[str]:
    """Names of the coefficient sets of the Jastrow factor, in the order they are written."""
    ids = ["e" + el for el in _species(system)]
    ids.append("uu")
    if system["ndn"] > 0:
        ids.append("ud")
    return ids


def _wavefunction(system: dict, jastrow: Optional[Dict[str, object]], coefficients: Optional[Dict[str, Sequence[float]]],
                  orbitals: str, determinantset: Optional[str] = None) -> str:
    """Wave function block. ``jastrow=None`` writes the determinant alone, with no Jastrow element at all.
    ``determinantset`` is the complete text of a ``<determinantset>`` element (several determinants, see the
    module ``multidet``); it replaces the single Kohn-Sham determinant, and ``orbitals`` is then not used."""
    if jastrow is None:
        n1 = n2 = 0; r1 = r2 = 0.0
    else:
        n1, r1 = int(jastrow["j1_size"]), float(jastrow["j1_rcut"])
        n2, r2 = int(jastrow["j2_size"]), float(jastrow["j2_rcut"])

    def coeff(cid, n):
        if coefficients is None:
            return " ".join(["0"] * n)
        if cid not in coefficients:
            raise KeyError("no Jastrow coefficients for %r" % cid)
        vals = list(coefficients[cid])
        if len(vals) != n:
            raise ValueError("Jastrow %r: %d coefficients given, %d expected" % (cid, len(vals), n))
        return " ".join("%.10g" % v for v in vals)

    if determinantset is not None:
        if "<determinantset" not in determinantset or "</determinantset>" not in determinantset:
            raise ValueError("the text given as determinant set holds no determinantset element")
        out = ['      <wavefunction name="psi0" target="e">', determinantset.rstrip("\n")]
    else:
        out = ['      <wavefunction name="psi0" target="e">',
               '         <determinantset type="MolecularOrbital" href="%s" source="ion0" transform="yes" name="LCAOBSet">' % orbitals,
               '            <slaterdeterminant>',
               '               <determinant id="updet" size="%d">' % system["nup"],
               '                  <occupation mode="ground"> </occupation>',
               '                  <coefficient size="%d" spindataset="0"> </coefficient>' % system["nao"],
               '               </determinant>']
        if system["ndn"] > 0:
            out += ['               <determinant id="downdet" size="%d">' % system["ndn"],
                    '                  <occupation mode="ground"> </occupation>',
                    '                  <coefficient size="%d" spindataset="%d"> </coefficient>' % (
                        system["nao"], 0 if system.get("restricted", True) else 1),
                    '               </determinant>']
        out += ['            </slaterdeterminant>', '         </determinantset>']
    if jastrow is None:
        out += ['      </wavefunction>']
        return "\n".join(out)
    out += ['         <jastrow type="One-Body" name="J1" function="bspline" source="ion0" print="yes">']
    for el in _species(system):
        out += ['            <correlation elementType="%s" size="%d" rcut="%s" cusp="0.0">' % (el, n1, r1),
                '               <coefficients id="e%s" type="Array"> %s </coefficients>' % (el, coeff("e" + el, n1)),
                '            </correlation>']
    out += ['         </jastrow>', '         <jastrow type="Two-Body" name="J2" function="bspline" print="yes">']
    pairs = [("u", "u", "uu")] + ([("u", "d", "ud")] if system["ndn"] > 0 else [])
    for a, b, cid in pairs:
        out += ['            <correlation speciesA="%s" speciesB="%s" size="%d" rcut="%s">' % (a, b, n2, r2),
                '               <coefficients id="%s" type="Array"> %s </coefficients>' % (cid, coeff(cid, n2)),
                '            </correlation>']
    out += ['         </jastrow>', '      </wavefunction>']
    return "\n".join(out)


def _hamiltonian(system: dict, pseudo_dir: str = "") -> str:
    out = ['      <hamiltonian name="h0" type="generic" target="e">',
           '         <pairpot type="coulomb" name="ElecElec" source="e" target="e"/>',
           '         <pairpot type="coulomb" name="IonIon" source="ion0" target="ion0"/>',
           '         <pairpot type="pseudo" name="PseudoPot" source="ion0" wavefunction="psi0" format="xml">']
    for el in _species(system):
        out.append('            <pseudo elementType="%s" href="%s"/>' % (el, os.path.join(pseudo_dir, "%s.ccECP.xml" % el)))
    out += ['         </pairpot>', '      </hamiltonian>']
    return "\n".join(out)


def _head(project: str, seed: Optional[int]) -> str:
    out = ['<?xml version="1.0"?>', '<simulation>', '   <project id="%s" series="0">' % project,
           '      <parameter name="driver_version"      >    legacy          </parameter>',
           '      <application name="qmcpack" role="molecu" class="serial" version="1.0"/>', '   </project>']
    if seed is not None:
        out.append('   <random seed="%d"/>' % int(seed))
    return "\n".join(out)


DEFAULT_JASTROW = {"j1_size": 6, "j1_rcut": 6.0, "j2_size": 8, "j2_rcut": 8.0}


def opt_input(system: dict, jastrow: Optional[Dict[str, object]] = None, cycles: int = 8, samples: int = 100000,
              start: Optional[Dict[str, Sequence[float]]] = None, orbitals: str = "scf.h5",
              pseudo_dir: str = "", seed: Optional[int] = None, project: str = "opt",
              vmc_timestep: float = 0.3, warmup: int = 300, blocks: int = 200, substeps: int = 2,
              determinantset: Optional[str] = None) -> str:
    """Jastrow optimization input (linear method, one shift). ``start`` holds starting coefficients
    (for example those of a neighbouring geometry); without it all coefficients start at zero."""
    j = dict(DEFAULT_JASTROW); j.update(jastrow or {})
    parts = [_head(project, seed), '   <qmcsystem>', _particlesets(system),
             _wavefunction(system, j, start, orbitals, determinantset),
             _hamiltonian(system, pseudo_dir), '   </qmcsystem>',
             '   <loop max="%d">' % int(cycles),
             '      <qmc method="linear" move="pbyp" checkpoint="-1">',
             '         <parameter name="warmupSteps"         >    %d                </parameter>' % warmup,
             '         <parameter name="blocks"              >    %d                </parameter>' % blocks,
             '         <parameter name="steps"               >    1                  </parameter>',
             '         <parameter name="subSteps"            >    %d                  </parameter>' % substeps,
             '         <parameter name="timestep"            >    %s                </parameter>' % vmc_timestep,
             '         <parameter name="useDrift"            >    no                 </parameter>',
             '         <parameter name="samples"             >    %d             </parameter>' % int(samples),
             '         <parameter name="MinMethod"           >    OneShiftOnly         </parameter>',
             '         <parameter name="minwalkers"          >    0.1                </parameter>',
             '      </qmc>', '   </loop>', '</simulation>', '']
    return "\n".join(parts)


def dmc_input(system: dict, coefficients: Dict[str, Sequence[float]], jastrow: Optional[Dict[str, object]] = None,
              timestep: float = 0.005, warmup_steps: int = 2000, blocks: int = 400, steps: int = 500,
              walkers: int = 2048, orbitals: str = "scf.h5", pseudo_dir: str = "", seed: Optional[int] = None,
              project: str = "dmc", vmc_timestep: float = 0.3, vmc_blocks: int = 40, vmc_steps: int = 10,
              vmc_substeps: int = 3, vmc_warmup: int = 30, tmoves: bool = True, checkpoint: int = -1,
              determinantset: Optional[str] = None) -> str:
    """Input with a VMC section (series 000: makes the walkers) and a DMC section (series 001).
    ``walkers`` is the target population of the DMC run; it must be a multiple of the number of MPI ranks."""
    j = dict(DEFAULT_JASTROW); j.update(jastrow or {})
    parts = [_head(project, seed), '   <qmcsystem>', _particlesets(system),
             _wavefunction(system, j, coefficients, orbitals, determinantset),
             _hamiltonian(system, pseudo_dir), '   </qmcsystem>',
             '   <qmc method="vmc" move="pbyp" checkpoint="-1">',
             '      <parameter name="warmupSteps"         >    %d              </parameter>' % vmc_warmup,
             '      <parameter name="blocks"              >    %d              </parameter>' % vmc_blocks,
             '      <parameter name="steps"               >    %d              </parameter>' % vmc_steps,
             '      <parameter name="subSteps"            >    %d               </parameter>' % vmc_substeps,
             '      <parameter name="timestep"            >    %s             </parameter>' % vmc_timestep,
             '      <parameter name="samples"             >    %d            </parameter>' % int(walkers),
             '      <parameter name="walkers"             >    1               </parameter>',
             '   </qmc>',
             '   <qmc method="dmc" move="pbyp" checkpoint="%d">' % checkpoint,
             '      <parameter name="warmupSteps"         >    %d            </parameter>' % int(warmup_steps),
             '      <parameter name="blocks"              >    %d             </parameter>' % int(blocks),
             '      <parameter name="steps"               >    %d             </parameter>' % int(steps),
             '      <parameter name="timestep"            >    %s           </parameter>' % timestep,
             '      <parameter name="nonlocalmoves"       >    %s             </parameter>' % ("yes" if tmoves else "no"),
             '   </qmc>', '</simulation>', '']
    return "\n".join(parts)


def vmc_check_input(system: dict, orbitals: str = "scf.h5", pseudo_dir: str = "", seed: Optional[int] = None,
                    project: str = "vmc0", walkers: int = 2048, blocks: int = 100, steps: int = 10,
                    timestep: float = 0.3, substeps: int = 3, warmup: int = 100) -> str:
    """VMC run of the determinant alone. Its energy must agree with the energy of the determinant computed
    by PySCF (``e_determinant`` in ``system.json``) within the statistical error. A disagreement means that
    the orbitals, the atom order or the pseudopotentials of the QMCPACK input are not those of the SCF.

    The input holds NO Jastrow element. A two-body Jastrow factor whose coefficients are all zero is not
    "no Jastrow": QMCPACK still gives it the electron-electron cusp, and that lowers the energy
    (measured: 0.042 Ha for water)."""
    parts = [_head(project, seed), '   <qmcsystem>', _particlesets(system), _wavefunction(system, None, None, orbitals),
             _hamiltonian(system, pseudo_dir), '   </qmcsystem>',
             '   <qmc method="vmc" move="pbyp" checkpoint="-1">',
             '      <parameter name="warmupSteps"         >    %d              </parameter>' % warmup,
             '      <parameter name="blocks"              >    %d              </parameter>' % blocks,
             '      <parameter name="steps"               >    %d              </parameter>' % steps,
             '      <parameter name="subSteps"            >    %d               </parameter>' % substeps,
             '      <parameter name="timestep"            >    %s             </parameter>' % timestep,
             '      <parameter name="samples"             >    %d            </parameter>' % int(walkers),
             '      <parameter name="walkers"             >    1               </parameter>',
             '   </qmc>', '</simulation>', '']
    return "\n".join(parts)


_ZVAL_RE = re.compile(r'<header[^>]*\bzval\s*=\s*"([^"]+)"', re.S)


def pseudopotential_zval(path: str) -> float:
    """Valence charge written in the header of a QMCPACK pseudopotential XML file."""
    with open(path) as fh:
        head = fh.read(20000)
    m = _ZVAL_RE.search(head)
    if not m:
        raise ValueError("%s: no zval in the header" % path)
    return float(m.group(1))


def check_pseudopotentials(system: dict, pseudo_dir: str = "") -> None:
    """Every element needs its pseudopotential file, and the valence charge of the file must equal the
    effective nuclear charge that the SCF used. Raises ValueError otherwise."""
    for el in _species(system):
        path = os.path.join(pseudo_dir, "%s.ccECP.xml" % el)
        if not os.path.exists(path):
            raise ValueError("pseudopotential file missing for element %s: %s" % (el, path))
        z_file = pseudopotential_zval(path)
        z_scf = {system["zeff"][i] for i, s in enumerate(system["symbols"]) if s == el}
        if z_scf != {int(round(z_file))}:
            raise ValueError("element %s: the pseudopotential file has zval %g but the SCF used charge %s"
                             % (el, z_file, sorted(z_scf)))
    n_el = system["nup"] + system["ndn"]
    if "charge" in system and sum(system["zeff"]) - int(system["charge"]) != n_el:
        raise ValueError("electron count %d does not match the valence charges %d and the molecular charge %d"
                         % (n_el, sum(system["zeff"]), system["charge"]))


# ----------------------------------------------------------------------------- reading the optimization
_COEFF_RE = re.compile(r'<coefficients\s+id="([^"]+)"[^>]*>([^<]*)</coefficients>', re.S)


def read_jastrow(path: str) -> Dict[str, List[float]]:
    """Jastrow coefficients of a QMCPACK wavefunction file (``*.opt.xml``) or input file."""
    with open(path) as fh:
        text = fh.read()
    out = {}
    for cid, body in _COEFF_RE.findall(text):
        out[cid] = [float(v) for v in body.split()]
    if not out:
        raise ValueError("%s holds no Jastrow coefficients" % path)
    return out


def _series_stats(scalar_path: str, drop_fraction: float = 0.2) -> Tuple[float, float, float]:
    """Mean energy, its plain standard error, and the variance of the local energy of one optimization cycle."""
    from ..stats import read_scalar

    cols = read_scalar(scalar_path)
    e = cols["LocalEnergy"]; e2 = cols["LocalEnergy_sq"]
    n0 = int(len(e) * drop_fraction)
    e, e2 = e[n0:], e2[n0:]
    if len(e) < 4:
        raise ValueError("%s: too few blocks" % scalar_path)
    mean = float(e.mean())
    return mean, float(e.std(ddof=1) / np.sqrt(len(e))), float(e2.mean() - mean * mean)


def read_optimization(folder: str = ".", project: str = "opt") -> List[dict]:
    """One entry per finished optimization cycle: series number, energy, error, variance, coefficient file.

    The file ``<project>.sNNN.scalar.dat`` holds the VMC run made with the parameters that cycle
    NNN started from; ``<project>.sNNN.opt.xml`` holds the parameters that cycle NNN produced. So the
    energy of the parameters in ``sNNN.opt.xml`` is measured by the scalar file of series NNN + 1.
    """
    out = []
    files = sorted(glob.glob(os.path.join(folder, "%s.s[0-9][0-9][0-9].scalar.dat" % project)))
    for f in files:
        n = int(re.search(r"\.s(\d{3})\.scalar\.dat$", f).group(1))
        try:
            e, se, var = _series_stats(f)
        except (OSError, ValueError, KeyError):
            continue                # a file that cannot be read, has too few blocks or lacks a column: not a usable cycle
        prev = os.path.join(folder, "%s.s%03d.opt.xml" % (project, n - 1))
        out.append({"series": n, "energy": e, "error": se, "variance": var,
                    "parameters_from": prev if (n > 0 and os.path.exists(prev)) else None})
    return out


def select_jastrow(cycles: Sequence[dict], variance_slack: float = 1.5) -> dict:
    """Choose the parameter set to use for DMC among the measured ones.

    Only parameter sets whose energy was measured by a later cycle can be chosen. Among those whose
    variance is at most ``variance_slack`` times the smallest variance, the one with the lowest energy
    is taken. Raises ValueError when no parameter set was measured or a value is not finite.
    """
    usable = [c for c in cycles if c.get("parameters_from") and np.isfinite(c["energy"]) and np.isfinite(c["variance"])
              and c["variance"] > 0]
    if not usable:
        raise ValueError("no optimization cycle with measured parameters")
    vmin = min(c["variance"] for c in usable)
    good = [c for c in usable if c["variance"] <= variance_slack * vmin]
    return min(good, key=lambda c: c["energy"])


# ----------------------------------------------------------------------------- script entry (runs inside the cell job)
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write an input file inside a cell folder.

        python -m montepass.engine.qmcpack_inputs vmc0 [settings.json]     (zero-Jastrow check run)
        python -m montepass.engine.qmcpack_inputs opt  [settings.json]
        python -m montepass.engine.qmcpack_inputs dmc  [settings.json]

    ``settings.json`` (default ``qmc_settings.json`` in the current folder) holds the keyword
    arguments of :func:`opt_input` under ``"opt"`` and of :func:`dmc_input` under ``"dmc"``, and
    optionally ``"jastrow"`` and ``"pseudo_dir"``. The ``dmc`` step reads the optimization result
    from the current folder, or the file named by ``"jastrow_file"`` when no optimization was run.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("opt", "dmc", "vmc0"):
        print(main.__doc__)
        return 2
    cfg = {}
    cfg_path = args[1] if len(args) > 1 else "qmc_settings.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    system = load_system("system.json")
    jas = cfg.get("jastrow")
    pseudo_dir = cfg.get("pseudo_dir", "")
    check_pseudopotentials(system, pseudo_dir)
    extra = {}
    if cfg.get("determinantset_file"):          # a point with several determinants (module multidet)
        with open(cfg["determinantset_file"]) as fh:
            extra["determinantset"] = fh.read().rstrip("\n")
    if args[0] == "vmc0":
        with open("vmc0.in.xml", "w") as fh:
            fh.write(vmc_check_input(system, pseudo_dir=pseudo_dir, **dict(cfg.get("vmc0", {}))))
        print("wrote vmc0.in.xml")
        return 0
    if args[0] == "opt":
        kw = dict(cfg.get("opt", {}))
        start = None
        if cfg.get("jastrow_start") and os.path.exists(cfg["jastrow_start"]):
            start = read_jastrow(cfg["jastrow_start"])
        with open("opt.in.xml", "w") as fh:
            fh.write(opt_input(system, jastrow=jas, start=start, pseudo_dir=pseudo_dir, **kw, **extra))
        print("wrote opt.in.xml")
        return 0
    kw = dict(cfg.get("dmc", {}))
    project = kw.get("project", "dmc")
    record = {}
    if cfg.get("jastrow_file"):
        coeff = read_jastrow(cfg["jastrow_file"]); record["source"] = cfg["jastrow_file"]
    else:
        cycles = read_optimization(".")
        best = select_jastrow(cycles)
        coeff = read_jastrow(best["parameters_from"])
        record = {"source": best["parameters_from"], "energy": best["energy"], "error": best["error"],
                  "variance": best["variance"], "cycles": cycles}
    missing = [c for c in jastrow_ids(system) if c not in coeff]
    if missing:
        raise SystemExit("Jastrow coefficients missing for %s" % ", ".join(missing))
    with open("jastrow.json", "w") as fh:
        json.dump({"coefficients": coeff, "selection": record}, fh, indent=1)
    with open("%s.in.xml" % project, "w") as fh:
        fh.write(dmc_input(system, coeff, jastrow=jas, pseudo_dir=pseudo_dir, **kw, **extra))
    print("wrote %s.in.xml" % project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
