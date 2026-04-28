from dataclasses import dataclass

import basix
import basix.ufl
import ufl
from dolfinx import fem, mesh as _mesh
from dolfinx.fem.petsc import create_matrix, assemble_matrix 
from petsc4py import PETSc


def plane_stress_hooke(
    eps_2d: ufl.core.expr.Expr,
    lmbda_ps: float,
    mu: float,
) -> ufl.core.expr.Expr:
    """Plane-stress isotropic Hooke law.

    Parameters
    ----------
    eps_2d
        2x2 in-plane strain tensor (UFL expression).
    lmbda_ps
        Plane-stress Lamé coefficient ``E*nu / (1 - nu**2)``.
    mu
        Shear modulus ``E / (2*(1 + nu))``.

    Returns
    -------
    ufl.core.expr.Expr
        2x2 stress tensor.
    """
    return lmbda_ps * ufl.tr(eps_2d) * ufl.Identity(2) + 2 * mu * eps_2d


@dataclass
class ShellMaterial:
    """Isotropic shell material parameters.

    Parameters
    ----------
    E
        Young's modulus.
    nu
        Poisson's ratio (must be in (-1, 0.5)).
    rho
        Mass density.
    h
        Shell thickness.

    Attributes
    ----------
    mu
        Shear modulus, derived as E / 2(1 + nu).
    lmbda_ps
        Plane-stress Lamé parameter, derived as E·nu / (1 - nu²).
    """

    E: float
    nu: float
    rho: float
    h: float

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def lmbda_ps(self) -> float:
        return self.E * self.nu / (1 - self.nu**2)


class ShellModel:
    """Reissner-Mindlin shell on a (possibly non-manifold) surface mesh.

    Builds the stiffness and mass forms with PSRI stabilization for
    membrane and shear locking, plus drilling-rotation stabilization.
    Supports triangle (P2 + bubble) and quadrilateral (P2) displacements;
    rotations are P2 in both cases.

    Parameters
    ----------
    mesh
        Surface mesh embedded in 3D (triangles or quadrilaterals).
    material
        Material parameters.
    bcs
        Optional list of Dirichlet boundary conditions applied to ``V``.
    """

    def __init__(
        self,
        mesh: _mesh.Mesh,
        material: ShellMaterial,
        bcs: list[fem.DirichletBC] | None = None,
    ) -> None:
        self.mesh = mesh
        self.mat = material
        self.dtype = PETSc.ScalarType
        self.bcs: list[fem.DirichletBC] = bcs if bcs is not None else []

        self._build_function_space()
        self._build_geometry()
        self._build_forms()

    # ----- Function space -----
    def _build_function_space(self) -> None:
        """Build the mixed (displacement, rotation) function space."""
        cell_name = self.mesh.topology.cell_name()

        if cell_name == "triangle":
            base = basix.ufl.element("Lagrange", cell_name, 2)
            bubble = basix.ufl.element("Bubble", cell_name, 3)
            enriched = basix.ufl.enriched_element([base, bubble])
            elem_u = basix.ufl.blocked_element(enriched, shape=(3,))
        else:
            elem_u = basix.ufl.element("Lagrange", cell_name, 2, shape=(3,))

        elem_theta = basix.ufl.element("Lagrange", cell_name, 2, shape=(3,))
        mixed_elem = basix.ufl.mixed_element([elem_u, elem_theta])

        self.V = fem.functionspace(self.mesh, mixed_elem)
        self.V_u, _ = self.V.sub(0).collapse()
        self.V_theta, _ = self.V.sub(1).collapse()

    # ----- Local geometry -----
    def _build_geometry(self) -> None:
        """Build shell normal ``n`` and tangent-plane projector ``P_plane``."""
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
    def _compute_strains(
        self,
        disp: ufl.core.expr.Expr,
        rot: ufl.core.expr.Expr,
    ) -> tuple[
        ufl.core.expr.Expr,
        ufl.core.expr.Expr,
        ufl.core.expr.Expr,
        ufl.core.expr.Expr,
    ]:
        """Compute membrane, bending, transverse-shear, and drilling strains.

        Parameters
        ----------
        disp
            Displacement field (3-vector).
        rot
            Rotation field (3-vector).

        Returns
        -------
        eps_m
            2x2 membrane strain.
        kappa
            2x2 bending curvature.
        gamma
            2-vector transverse shear strain.
        drill
            Scalar drilling strain.
        """
        n = self.n
        P = self.P_plane

        def t_grad(v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
            return ufl.dot(ufl.grad(v), P)

        t_gu = ufl.dot(P.T, t_grad(disp))
        eps_m = ufl.sym(t_gu)

        beta = ufl.cross(n, rot)
        kappa = ufl.sym(ufl.dot(P.T, t_grad(beta)))

        gamma = t_grad(ufl.dot(disp, n)) - ufl.dot(P.T, beta)

        drill = ufl.dot(rot, n) - 0.5 * (t_gu[0, 1] - t_gu[1, 0])

        return eps_m, kappa, gamma, drill

    # ----- Variational forms -----

    def _build_forms(self) -> None:
        """Build stiffness and mass forms and allocate PETSc matrices."""
        m = self.mat
        mesh = self.mesh
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
        dx_f = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
        dx_r = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})

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

        self.K = create_matrix(self.K_form)
        self.M = create_matrix(self.M_form)

    # ----- Assembly -----

    def assemble_matrices(self) -> None:
        """Assemble stiffness ``K`` and mass ``M`` PETSc matrices."""
        self.K.zeroEntries()
        assemble_matrix(self.K, self.K_form, bcs=self.bcs)
        self.K.assemble()

        self.M.zeroEntries()
        assemble_matrix(self.M, self.M_form, bcs=self.bcs, diag=0)
        self.M.assemble()