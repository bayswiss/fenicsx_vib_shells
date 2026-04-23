# Copyright (C) 2026 Antonio Baiano Svizzero
#
# This file is part of FEniCSx_vib_Shells (https://github.com/bayswiss/fenicsx_vib_shells)
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Frequency response of a shell under a point force.

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, la
from dolfinx.fem.petsc import set_bc, apply_lifting
from dolfinx.io import gmsh, VTXWriter
from scifem import PointSource
from utils import FieldProbe, plot_complex_spectra
from fenicsx_vib_shells import ShellMaterial, ShellModel


# Frequencies [Hz]
frequencies = np.arange(1.0, 20.0, 0.5)

# Material
mat = ShellMaterial(
    E   = 2e9,
    nu  = 0.4,
    rho = 1e3,
    h   = 2e-3,
)

# Mesh
mesh_data = gmsh.read_from_msh("test/tbeam.msh", MPI.COMM_WORLD, 0, gdim=3)
msh = mesh_data[0]
facet_tags = mesh_data[2]

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

# Point force: F N in +z at x_S 
F = 1.0
if msh.comm.rank == 0:
    x_S = np.array([[0.11, 0.0, 1.5]])
else:
    x_S = np.zeros((0, 3))

# u sub-space, then scalar z component
V_u_coll, V_u_to_V   = shell.V.sub(0).collapse()
V_uz,     V_uz_to_Vu = V_u_coll.sub(2).collapse()
V_uz_to_V = np.asarray(V_u_to_V)[np.asarray(V_uz_to_Vu)]

# Point source on scalar V_uz
b_uz = fem.Function(V_uz)
ps = PointSource(V_uz, x_S, magnitude=F)
ps.apply_to_vector(b_uz)
b_uz.x.scatter_reverse(la.InsertMode.add)
b_uz.x.scatter_forward()

# Lift into mixed shell.V
b = fem.Function(shell.V)
b.x.array[V_uz_to_V] = b_uz.x.array

apply_lifting(b.x.petsc_vec, [shell.K_form], [shell.bcs])  
b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)
set_bc(b.x.petsc_vec, shell.bcs)
b.x.scatter_forward()

# Solution
uh = fem.Function(shell.V)

# KSP: direct LU with MUMPS
ksp = PETSc.KSP().create(msh.comm)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# A = K - omega^2 M, rebuilt per freq
# Hysteretic loss factor eta: A = (1 + 1j*eta)*K - omega^2 * M
A = shell.K.duplicate()

# VTX: u only, interp to Lagrange-1 vector
V_u_lagrange = fem.functionspace(msh, ("Lagrange", 1, (3,)))
u_out = fem.Function(V_u_lagrange)
u_out.name = "u"

vtx = VTXWriter(msh.comm, "fields/u_frf.bp", [u_out])

# Probe  
x_probe = np.array([-0.11, 0.0, 1.5])
probe = FieldProbe(msh, x_probe)
u_probe = np.zeros((len(frequencies), 3), dtype=complex)

for nf, f in enumerate(frequencies):
    if msh.comm.rank == 0:
        print(f"f = {f:.2f} Hz", flush=True)

    omega = 2 * np.pi * f

    # A = K - omega^2 M
    A.zeroEntries()
    A.axpy(1.0, shell.K)
    A.axpy(-omega**2, shell.M)

    ksp.setOperators(A)
    ksp.solve(b.x.petsc_vec, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Export u
    u_out.interpolate(uh.sub(0))
    vtx.write(f)

    # Probe u
    u_f = probe.sample(uh.sub(0))
    u_f_all = msh.comm.gather(u_f, root=0)
    if msh.comm.rank == 0:
        for arr in u_f_all:
            if np.asarray(arr).size > 0:
                u_probe[nf, :] = np.asarray(arr).reshape(-1, 3)[0, :]
                break

vtx.close()

# Plot: 3 components of u at probe
if msh.comm.rank == 0:
    plot_complex_spectra(
        x_axis=frequencies,
        p_spectra_list=u_probe.T,
        labels=["u_x", "u_y", "u_z"],
        title="Displacement at probe",
        plot_db=False,
    )