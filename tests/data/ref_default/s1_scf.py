#!/usr/bin/env python3
"""Kohn-Sham calculation of one geometry: trial orbitals for QMC and the surrogate energy."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import json
import numpy as np
from pyscf import gto, dft, lib
lib.num_threads(4)

mol = gto.Mole()
mol.atom = '''
    H        0.0000000000       0.7631265780      -0.5855676186
    H        0.0000000000      -0.7631265780      -0.5855676186
    O        0.0000000000       0.0000000000       0.0000000000
'''
mol.unit = 'A'
mol.basis = 'ccecpccpvtz'
mol.ecp = 'ccecp'
mol.charge = 0
mol.spin = 0
mol.symmetry = False
mol.build()

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.grids.level = 3
mf.conv_tol = 1e-10
mf.max_cycle = 300
e_scf = mf.kernel()
assert mf.converged, 'SCF not converged'
np.savetxt('energy.dat', [e_scf, 0.0])

# Energy of the Kohn-Sham determinant with the true Hamiltonian (the Hartree-Fock energy expression).
# A VMC run with no Jastrow factor must reproduce this number: it tests the whole conversion to QMCPACK.
from pyscf import scf as _scf
e_det = float(_scf.RHF(mol).energy_tot(dm=mf.make_rdm1()))
nup, ndn = mol.nelec
json.dump({'symbols': [mol.atom_symbol(i) for i in range(mol.natm)],
           'coords_bohr': mol.atom_coords().tolist(),
           'zeff': [int(z) for z in mol.atom_charges()],
           'nup': int(nup), 'ndn': int(ndn), 'nao': int(mol.nao_nr()),
           'energy': float(e_scf), 'e_determinant': e_det, 'xc': 'b3lyp', 'basis': 'ccecpccpvtz', 'ecp': 'ccecp',
           'restricted': True}, open('system.json', 'w'), indent=1)

from PyscfToQmcpack import savetoqmcpack
savetoqmcpack(mol, mf, 'scf')
print('SCF_DONE')
