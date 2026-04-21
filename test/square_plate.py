from mpi4py import MPI
from dolfinx.io import gmsh, VTXWriter
from dolfinx import fem
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
import numpy as np
from fenicsx_vib_shells import ShellMaterial, ShellModel

mesh_data = gmsh.read_from_msh("test/square.msh", MPI.COMM_WORLD, 0, gdim=3)
msh = mesh_data[0]
cell_tags = mesh_data[1]
facet_tags = mesh_data[2]

mat = ShellMaterial(
    E   = 2e9,
    nu  = 0.4,
    rho = 1e3,
    h   = 2e-3
)

shell = ShellModel(msh, mat)

# Translation bc
dofs_u = fem.locate_dofs_topological(
    (shell.V.sub(0), shell.V_u), 1, facet_tags.find(2))
u0 = fem.Function(shell.V_u)
bc_u = fem.dirichletbc(u0, dofs_u, shell.V.sub(0))

shell.bcs.append(bc_u)

# Rotation bc
dofs_theta = fem.locate_dofs_topological(
    (shell.V.sub(1), shell.V_theta), 1, facet_tags.find(2))
u0 = fem.Function(shell.V_u)
bc_theta = fem.dirichletbc(u0, dofs_theta, shell.V.sub(1))

shell.bcs.append(bc_theta)

shell.assemble_matrices()

# Solver setup
N_modes = 6
solver = SLEPc.EPS().create()
solver.setDimensions(N_modes)
solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

st = SLEPc.ST().create()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.0)
st.setFromOptions()

solver.setST(st)
solver.setOperators(shell.K,shell.M)

solver.solve()

tol, maxit = solver.getTolerances()
nconv = solver.getConverged()

if msh.comm.rank==0:
    print("Number of iterations of the method: %i" % solver.getIterationNumber())
    print("Solution method: %s" % solver.getType())
    print("")
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))


# Save the k
vals = [(i, solver.getEigenvalue(i)) for i in range(nconv)]

xr = fem.Function(shell.V)
u_mode = fem.Function(shell.V_u)
ghosts=shell.V.dofmap.index_map.ghosts

eig_vector = []
eig_matrix = np.zeros((len(xr.x.array), nconv), dtype=complex)
eig_freq = []

vtx = VTXWriter(msh.comm, "fields/Modes.bp", [u_mode])
    

if nconv>0:
    for i, k in vals:
        solver.getEigenpair(i, xr.x.petsc_vec)
        fn = np.sqrt(k)/(2 * np.pi) 
        eig_freq.append(fn)
        if msh.comm.rank==0:
            print("%12f Hz" % fn)

        xr.x.scatter_forward()

        u_mode.interpolate(xr.sub(0))
        # vtx.write(np.round(fn, 2))

        vect = u_mode.x.petsc_vec.getArray()

        eig_vector.append(vect.copy())
        

        eig_matrix[:,i]=xr.x.array

D = mat.E * mat.h**3 / (12*(1 - mat.nu**2))

f_factor = 1/(2*np.pi*0.3**2)*np.sqrt(D/(mat.rho*mat.h))
print("Leissa first modes:")
print(35.99 * f_factor)
print(73.41 * f_factor)
print(73.41 * f_factor)
print(108.27 * f_factor)
print(131.64 * f_factor)
print(131.64 * f_factor)
