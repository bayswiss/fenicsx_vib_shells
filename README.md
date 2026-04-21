# fenicsx-linear-shell-dynamics

A FEniCSx shell tool to run modal analysis and linear frequency response.

Compatible with dolfinx 0.9.0 and 0.10.0
A Reissner–Mindlin / Naghdi shell element with:

- Mixed displacement–rotation formulation (6 DOFs per node)
- Drilling DOF stabilization
- Bubble enrichment (P2+B3 on triangles)
- Partial Selective Reduced Integration (PSRI) against membrane and shear locking

## Installation

Once you are inside your FEniCSx environment, download this repository and install it:

**Clone the repository:**
  
```bash
git clone https://github.com/bayswiss/fenicsx_vib_shells.git
cd fenicsx_vib_shells
```

**Install using pip:**
```bash
pip install .
```

**For developers (allows you to edit code without reinstalling):**
```bash
pip install -e .

## Quick start

```python
from fenicsx_linear_shell_dynamics import ShellMaterial, ShellModel

mat = ShellMaterial(E=210e9, nu=0.3, rho=7800, h=0.01)
model = ShellModel(mesh, degree=2, material=mat)
model.assemble_matrices()
# model.K and model.M are PETSc matrices ready for SLEPc
```

## References

- J. Bleyer, [Linear Shells](https://bleyerj.github.io/comet-fenicsx/tours/shells/linear_shell/linear_shell.html), Numerical Tours of Computational Mechanics with FEniCSx
- FEniCSx-Shells, [Nonlinear Naghdi clamped semi-cylinder](https://fenics-shells.github.io/fenicsx-shells/demo/demo_nonlinear-naghdi-clamped-semicylinder.html)

## License

GPL-3.0