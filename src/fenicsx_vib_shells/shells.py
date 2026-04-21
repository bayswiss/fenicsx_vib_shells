from mpi4py import MPI
import numpy as np
import ufl
import basix
import basix.ufl
from dolfinx import fem
from dolfinx.fem.petsc import create_matrix
from petsc4py import PETSc


def plane_stress_hooke(eps_2d, lmbda_ps, mu):
    return lmbda_ps * ufl.tr(eps_2d) * ufl.Identity(2) + 2 * mu * eps_2d


class ShellMaterial:
    def __init__(self, E, nu, rho, h):
        self.E = E
        self.nu = nu
        self.rho = rho
        self.h = h
        self.mu = E / (2 * (1 + nu))
        # self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        # self.lmbda_ps = 2 * self.lmbda * self.mu / (self.lmbda + 2 * self.mu)
        self.lmbda_ps = E * nu / (1 - nu**2)



class ShellModel:
    def __init__(self, mesh, degree, material, bcs=None):
        self.mesh = mesh
        self.degree = degree
        self.mat = material
        self.dtype = PETSc.ScalarType
        self.bcs = bcs if bcs is not None else []

        self._build_function_space()
        self._build_geometry()
        self._build_forms()

    # ----- Function space -----
    def _build_function_space(self):
        cell_name = self.mesh.topology.cell_name()
        deg = self.degree

        if deg==2 and cell_name=="triangle":
            base = basix.ufl.element("Lagrange", cell_name, deg)
            bubble = basix.ufl.element("Bubble", cell_name, 3)
            enriched = basix.ufl.enriched_element([base, bubble])
            elem_u = basix.ufl.blocked_element(enriched, shape=(3,))
        else:
            elem_u = basix.ufl.element("Lagrange", cell_name, deg, shape=(3,))

        elem_theta = basix.ufl.element("Lagrange", cell_name, deg, shape=(3,))
        mixed_elem = basix.ufl.mixed_element([elem_u, elem_theta])

        self.V = fem.functionspace(self.mesh, mixed_elem)
        self.V_u, _ = self.V.sub(0).collapse()
        self.V_theta, _ = self.V.sub(1).collapse()

    # ----- Local geometry -----

    def _build_geometry(self):
        J = ufl.Jacobian(self.mesh)
        t1 = ufl.as_vector([J[0, 0], J[1, 0], J[2, 0]])
        t2 = ufl.as_vector([J[0, 1], J[1, 1], J[2, 1]])

        n_raw = ufl.cross(t1, t2)
        self.n = n_raw / ufl.sqrt(ufl.dot(n_raw, n_raw))

        ey = ufl.as_vector([0, 1, 0])
        ez = ufl.as_vector([0, 0, 1])
        e1_candidate = ufl.cross(ey, self.n)
        e1_fallback = ufl.cross(ez, self.n)
        norm_c = ufl.sqrt(ufl.real(ufl.dot(e1_candidate, e1_candidate)))
        e1_raw = ufl.conditional(ufl.lt(ufl.real(norm_c), 0.5), e1_fallback, e1_candidate)
        e1 = e1_raw / ufl.sqrt(ufl.real(ufl.dot(e1_raw, e1_raw)))
        e2 = ufl.cross(self.n, e1)
        e2 = e2 / ufl.sqrt(ufl.real(ufl.dot(e2, e2)))


        self.P_plane = ufl.as_matrix([
            [e1[0], e2[0]],
            [e1[1], e2[1]],
            [e1[2], e2[2]],
        ])

    # ----- Strains -----

    def _compute_strains(self, disp, rot):
        n = self.n
        P = self.P_plane

        def t_grad(v):
            return ufl.dot(ufl.grad(v), P)

        t_gu = ufl.dot(P.T, t_grad(disp))
        eps_m = ufl.sym(t_gu)

        beta = ufl.cross(n, rot)
        kappa = ufl.sym(ufl.dot(P.T, t_grad(beta)))

        gamma = t_grad(ufl.dot(disp, n)) - ufl.dot(P.T, beta)

        drill = ufl.dot(rot, n) - 0.5 * (t_gu[0, 1] - t_gu[1, 0])

        return eps_m, kappa, gamma, drill

    # ----- Variational forms -----

    def _build_forms(self):
        m = self.mat
        mesh = self.mesh
        deg = self.degree
        h = m.h
        n = self.n

        u_mixed = ufl.TrialFunction(self.V)
        v_mixed = ufl.TestFunction(self.V)
        u, theta = ufl.split(u_mixed)
        v_u, v_theta = ufl.split(v_mixed)

        eps_m_u, kappa_u, gamma_u, drill_u = self._compute_strains(u, theta)
        eps_m_v, kappa_v, gamma_v, drill_v = self._compute_strains(v_u, v_theta)

        # Stress resultants
        N_u = h * plane_stress_hooke(eps_m_u, m.lmbda_ps, m.mu)
        M_u = (h**3 / 12) * plane_stress_hooke(kappa_u, m.lmbda_ps, m.mu)
        Q_u = m.mu * (5 / 6) * h * gamma_u

        # PSRI
        full_degree = 2 * deg # + 2
        reduced_degree = max(1, deg)
        dx_f = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": full_degree})
        dx_r = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": reduced_degree})

        h_cell = ufl.CellDiameter(mesh)
        alpha_raw = h**2 / h_cell**2
        alpha = ufl.min_value(alpha_raw, 1)
        
        k_drill = m.E * h**3 / h_cell**2

        # Stiffness form
        a_K = (
            ufl.inner(M_u, kappa_v) * dx_f
            + k_drill * ufl.inner(drill_u, drill_v) * dx_f
            + alpha * ufl.inner(N_u, eps_m_v) * dx_f
            + (1 - alpha) * ufl.inner(N_u, eps_m_v) * dx_r
            + alpha * ufl.inner(Q_u, gamma_v) * dx_f
            + (1 - alpha) * ufl.inner(Q_u, gamma_v) * dx_r
        )

        # Mass form (tangential rotary inertia only)
        theta_tan = theta - ufl.dot(theta, n) * n
        v_theta_tan = v_theta - ufl.dot(v_theta, n) * n

        a_M = (
            m.rho * h * ufl.inner(u, v_u)
            + (m.rho * h**3 / 12) * ufl.inner(theta_tan, v_theta_tan)
        ) * dx_f

        self.K_form = fem.form(a_K)
        self.M_form = fem.form(a_M)

        self.K = fem.petsc.create_matrix(self.K_form)
        self.M = fem.petsc.create_matrix(self.M_form)

    # ----- Assembly -----

    def assemble_matrices(self):
        self.K.zeroEntries()
        fem.petsc.assemble_matrix(self.K, self.K_form, bcs=self.bcs)
        self.K.assemble()

        self.M.zeroEntries()
        fem.petsc.assemble_matrix(self.M, self.M_form, bcs=self.bcs, diag = 0)
        self.M.assemble()