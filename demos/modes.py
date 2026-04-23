# Copyright (C) 2026 Antonio Baiano Svizzero
#
# This file is part of FEniCSx_vib_Shells (https://github.com/bayswiss/fenicsx_vib_shells)
#
# SPDX-License-Identifier: GPL-3.0-or-later

from mpi4py import MPI
from dolfinx.io import gmsh, VTXWriter
from dolfinx import fem
from slepc4py import SLEPc
import numpy as np
from fenicsx_vib_shells import ShellMaterial, ShellModel


mesh_data = gmsh.read_from_msh("test/tbeam.msh", MPI.COMM_WORLD, 0, gdim=3)
msh = mesh_data[0]
facet_tags = mesh_data[2]

mat = ShellMaterial(
    E   = 2e9,
    nu  = 0.4,
    rho = 1e3,
    h   = 2e-3
)

shell = ShellModel(msh, mat)

# u = 0
dofs_u = fem.locate_dofs_topological(
    (shell.V.sub(0), shell.V_u), 1, facet_tags.find(2))
u0 = fem.Function(shell.V_u)
shell.bcs.append(fem.dirichletbc(u0, dofs_u, shell.V.sub(0)))

# theta = 0
dofs_theta = fem.locate_dofs_topological(
    (shell.V.sub(1), shell.V_theta), 1, facet_tags.find(2))
th0 = fem.Function(shell.V_theta)
shell.bcs.append(fem.dirichletbc(th0, dofs_theta, shell.V.sub(1)))

shell.assemble_matrices()

# SLEPc solver setup
N_modes = 6
solver = SLEPc.EPS().create()
solver.setDimensions(N_modes)
solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

st = SLEPc.ST().create()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.0)

ksp = st.getKSP()
ksp.setType('preonly')
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')

st.setFromOptions()
solver.setST(st)
solver.setOperators(shell.K, shell.M)
solver.solve()

tol, maxit = solver.getTolerances()
nconv = solver.getConverged()

if msh.comm.rank == 0:
    print("Number of iterations: %i" % solver.getIterationNumber())
    print("Solution method: %s"      % solver.getType())
    print("")
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

# Write modes to VTX
xr = fem.Function(shell.V)
V_u_lagrange = fem.functionspace(msh, ("Lagrange", 2, (3, )))
u_mode = fem.Function(V_u_lagrange)

vtx = VTXWriter(msh.comm, "fields/Modes.bp", [u_mode])

for i in range(nconv):
    k = solver.getEigenpair(i, xr.x.petsc_vec)
    fn = np.sqrt(np.real(k)) / (2 * np.pi)
    if msh.comm.rank == 0:
        print("%12f Hz" % fn)

    xr.x.scatter_forward()
    # u sub only
    u_mode.interpolate(xr.sub(0))
    vtx.write(float(fn))

vtx.close()
