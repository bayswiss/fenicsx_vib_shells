# FEniCSx_vib_Shells

A FEniCSx shell library for modal analysis and linear frequency response.

**Compatible with dolfinx v0.10.0.**

<img width="604" height="847" alt="Screenshot from 2026-04-24 12-27-27" src="https://github.com/user-attachments/assets/cbc1f377-b9db-4747-8af4-2adf5b13a3bd" />

## Prerequisites
This tool requires **FEniCSx** with complex build of PETSc. We recommend installing it via ```conda```.

**Install FEniCSx via Conda:**
    
```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich petsc=*=complex* 
```
Alternative ways to install FEniCSx: [https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation)

## Installation

Inside your FEniCSx environment, clone the repository and install:

```bash
git clone https://github.com/bayswiss/fenicsx_vib_shells.git
cd fenicsx_vib_shells
pip install .
```

For development (editable install):

```bash
pip install -e .
```

## Usage

```python
from mpi4py import MPI
from dolfinx.io import gmsh
from fenicsx_vib_shells import ShellMaterial, ShellModel

msh, _, facet_tags = gmsh.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=3)

mat   = ShellMaterial(E=210e9, nu=0.3, rho=7800, h=0.01)
shell = ShellModel(msh, mat)

# Add Dirichlet BCs via shell.bcs.append(...)  (see demos)

shell.assemble_matrices()
# shell.K and shell.M are PETSc matrices - ready for SLEPc (modal) or PETSc (FRF)
```

## References

- J. Bleyer - [Linear Shells](https://bleyerj.github.io/comet-fenicsx/tours/shells/linear_shell/linear_shell.html)
- J. S. Hale, M. Brunetti, S. Bordas, C. Maurini - [FENICS-SHELLS: an open-source library for simulating thin structures](https://hal.sorbonne-universite.fr/hal-01763370v1/file/fenics-shells.pdf)
- T. Yang, M. Brunetti - [Clamped semi-cylindrical Naghdi shell under point load](https://fenics-shells.github.io/fenicsx-shells/demo/demo_nonlinear-naghdi-clamped-semicylinder.html)
