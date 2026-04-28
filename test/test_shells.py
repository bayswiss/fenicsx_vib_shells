# Copyright (C) 2026 Antonio Baiano Svizzero
#
# This file is part of FEniCSx_vib_Shells (https://github.com/bayswiss/fenicsx_vib_shells)
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh
from slepc4py import SLEPc

from fenicsx_vib_shells import ShellMaterial, ShellModel


N_MODES = 5


def _compute_modal_frequencies(
    mesh_file,
    material,
    clamp_u_tags,
    clamp_theta_tags,
    n_modes=N_MODES,
    comm=None,
):
    """Solve the shell GHEP and return natural frequencies in Hz.

    Parameters
    ----------
    mesh_file
        Path to a Gmsh ``.msh`` file.
    material
        Shell material parameters.
    clamp_u_tags
        Facet tags on which the displacement is clamped to zero.
    clamp_theta_tags
        Facet tags on which the rotation is clamped to zero.
    n_modes
        Number of eigenpairs to extract.
    comm
        MPI communicator (defaults to ``MPI.COMM_WORLD``).

    Returns
    -------
    np.ndarray
        Array of length ``n_modes`` with frequencies in Hz.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    mesh_data = gmsh.read_from_msh(mesh_file, comm, 0, gdim=3)
    msh = mesh_data[0]
    facet_tags = mesh_data[2]

    shell = ShellModel(msh, material)

    # u = 0
    if clamp_u_tags:
        u0 = fem.Function(shell.V_u)
        for tag in clamp_u_tags:
            facets = facet_tags.find(tag)
            dofs = fem.locate_dofs_topological(
                (shell.V.sub(0), shell.V_u), 1, facets
            )
            shell.bcs.append(fem.dirichletbc(u0, dofs, shell.V.sub(0)))

    # theta = 0
    if clamp_theta_tags:
        th0 = fem.Function(shell.V_theta)
        for tag in clamp_theta_tags:
            facets = facet_tags.find(tag)
            dofs = fem.locate_dofs_topological(
                (shell.V.sub(1), shell.V_theta), 1, facets
            )
            shell.bcs.append(fem.dirichletbc(th0, dofs, shell.V.sub(1)))

    shell.assemble_matrices()

    # SLEPc setup
    solver = SLEPc.EPS().create(comm)
    solver.setDimensions(n_modes)
    solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    st = SLEPc.ST().create(comm)
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(0.0)

    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    st.setFromOptions()
    solver.setST(st)

    solver.setOperators(shell.K, shell.M)
    solver.solve()

    nconv = solver.getConverged()
    assert nconv >= n_modes, (
        f"SLEPc converged only {nconv} eigenpairs, expected >= {n_modes}"
    )

    freqs = np.zeros(n_modes)
    for i in range(n_modes):
        k = solver.getEigenvalue(i)
        freqs[i] = np.sqrt(np.real(k)) / (2.0 * np.pi)

    solver.destroy()
    return freqs


def _assert_modes(freqs, ref, tol, label):
    """Print a comparison table and assert max relative error below ``tol``.

    Parameters
    ----------
    freqs
        Computed frequencies in Hz.
    ref
        Reference frequencies in Hz.
    tol
        Maximum allowed relative error.
    label
        Short label used in the printed report and assertion message.
    """
    freqs = np.asarray(freqs, dtype=float)
    ref = np.asarray(ref, dtype=float)
    rel = np.abs(freqs - ref) / np.abs(ref)
    max_rel = float(np.max(rel))

    if MPI.COMM_WORLD.rank == 0:
        print(f"\n[{label}] first {len(ref)} natural frequencies")
        print(f"{'idx':>3}  {'computed [Hz]':>14}  {'reference [Hz]':>14}  {'rel err':>9}")
        for i, (fc, fr, e) in enumerate(zip(freqs, ref, rel)):
            print(f"{i:>3}  {fc:>14.4f}  {fr:>14.4f}  {e:>9.2%}")
        print(f"[{label}] max relative error: {max_rel:.2%} (tol {tol:.2%})")

    assert max_rel < tol, (
        f"[{label}] max relative eigenfrequency error {max_rel:.2%} "
        f"exceeds tolerance {tol:.2%}"
    )


def test_plate_modes():
    """Square plate first modes vs Leissa analytical reference."""
    material = ShellMaterial(
        E=2e9,
        nu=0.4,
        rho=1e3,
        h=2e-3,
    )

    mesh_file        = "test/plate.msh"
    clamp_u_tags     = [2]
    clamp_theta_tags = [2]
    ref_freqs        = [  # Leissa [Hz]
        35.99,
        73.41,
        73.41,
        108.27,
        131.64,
    ]
    tol = 2e-2

    freqs = _compute_modal_frequencies(
        mesh_file, material, clamp_u_tags, clamp_theta_tags
    )
    # Leissa scaling
    a = 0.3  # square plate size
    D = material.E * material.h**3 / (12 * (1 - material.nu**2))
    f_factor = 1 / (2 * np.pi * a**2) * np.sqrt(D / (material.rho * material.h))
    freqs_scaled = freqs / f_factor

    _assert_modes(freqs_scaled, ref_freqs, tol, "plate")


def test_tbeam_modes():
    """T-beam first modes vs 3D FE reference."""
    material = ShellMaterial(
        E=2e9,
        nu=0.4,
        rho=1e3,
        h=2e-3,
    )

    mesh_file        = "test/tbeam.msh"
    clamp_u_tags     = [2]
    clamp_theta_tags = [2]
    ref_freqs        = [  # 3D FE [Hz]
        1.861502,
        5.679991,
        9.803593,
        14.437149,
        17.731813,
    ]
    tol = 5e-2

    freqs = _compute_modal_frequencies(
        mesh_file, material, clamp_u_tags, clamp_theta_tags
    )
    _assert_modes(freqs, ref_freqs, tol, "T-beam")


def test_cyl_shell_modes():
    """Cylindrical shell first modes vs 3D FE reference."""
    material = ShellMaterial(
        E=2e9,
        nu=0.4,
        rho=1e3,
        h=2e-3,
    )

    mesh_file        = "test/cyl.msh"
    clamp_u_tags     = [2]
    clamp_theta_tags = [2]
    ref_freqs        = [  # 3D FE [Hz]
        27.643969,
        60.785341,
        60.978816,
        101.611235,
        112.830119,
    ]
    tol = 5e-2

    freqs = _compute_modal_frequencies(
        mesh_file, material, clamp_u_tags, clamp_theta_tags
    )
    _assert_modes(freqs, ref_freqs, tol, "cylinder")
 