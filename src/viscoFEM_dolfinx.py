# fixme: plotting code not provided. Wrap vedo or pyvista for jupyter notebook
from dolfinx.mesh import (
    locate_entities_boundary,
    create_unit_cube,
    CellType as ctype,
)
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver
from ufl import (
    TestFunction,
    TrialFunction,
    grad,
    inner,
    dot,
    Identity,
    avg,
    FacetArea,
    ln,
    det,
    Measure,
    derivative,
    variable,
    diff,
    SpatialCoordinate,
    TensorElement,
    inv,
    sqrt,
    FacetArea,
    jump,
    tr,
)
import dolfinx.fem as fem, ufl
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType as st
from petsc4py import PETSc
import math
from matplotlib import pyplot as plt
from dolfinx.log import set_log_level, LogLevel, set_output_file
from dolfiny.interpolation import interpolate as dl_interp

set_log_level(LogLevel.INFO)

# setup quadrature degree
metadata = {"quadrature_degree": 5}

# mark the boundaries
left = lambda x: np.isclose(x[0], 0)
right = lambda x: np.isclose(x[0], 1.0)
bottom = lambda x: np.isclose(x[1], 0)
back = lambda x: np.isclose(x[2], 0)

# helper function to project
def project(v, target_func, degree, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh, degree=degree)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(Pv, w) * dx)
    L = fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = fem.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.assemble_vector(L)
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)


# # define the material energy
def freeEnergy(C, Cv):
    J = sqrt(det(C))
    I1 = tr(C)
    Ce = C * inv(Cv)
    Ie1 = tr(Ce)
    Je = J / sqrt(det(Cv))

    psiEq = (
        3 ** (1 - alph1) / (2.0 * alph1) * mu1 * (I1 ** alph1 - 3 ** alph1)
        + 3 ** (1 - alph2) / (2.0 * alph2) * mu2 * (I1 ** alph2 - 3 ** alph2)
        - (mu1 + mu2) * ln(J)
        + mu_pr / 2 * (J - 1) ** 2
    )

    psiNeq = (
        3 ** (1 - a1) / (2.0 * a1) * m1 * (Ie1 ** a1 - 3 ** a1)
        + 3 ** (1 - a2) / (2.0 * a2) * m2 * (Ie1 ** a2 - 3 ** a2)
        - (m1 + m2) * ln(Je)
    )

    return psiEq + psiNeq


def evolEqG(C, Cv, Cv_step):
    Iv1 = tr(Cv)
    Ce = C * inv(Cv)
    CvinvC = inv(Cv) * C
    Ie1 = tr(Ce)

    # define etaK
    A2 = m1 * (Ie1 / 3) ** (a1 - 1) + m2 * (Ie1 / 3) ** (a2 - 1)

    G = (
        A2
        / (
            etaInf
            + (eta0 - etaInf + K1 * (Iv1 ** bta1 - 3 ** bta1))
            / (
                1
                + (K2 * ((-(Ie1 ** 2) / 6.0 + 1.0 / 2 * inner(CvinvC, Ce)) * A2 ** 2))
                ** bta2
            )
        )
        * (C - Ie1 / 3.0 * Cv)
    )
    dl_interp(G, Cv_step)


def k_terms(dt, C, Cn, Cvn, Cn_quart, Cn_half, Cn_thr_quart, Cv, k_cache):

    k1, k2, k3, k4, k5, k6 = k_cache

    with Cn_quart.vector.localForm() as loc:
        loc.set(0.0)
    with Cn_half.vector.localForm() as loc:
        loc.set(0.0)
    with Cn_thr_quart.vector.localForm() as loc:
        loc.set(0.0)

    Cn_quart.vector.axpy(1.0, Cn.vector + 0.25 * (C.vector - Cn.vector))
    Cn_half.vector.axpy(1.0, Cn.vector + 0.5 * (C.vector - Cn.vector))
    Cn_thr_quart.vector.axpy(1.0, Cn.vector + 0.75 * (C.vector - Cn.vector))
    # Cn_quart = Cn + 0.25 * (C - Cn)
    # Cn_half = Cn + 0.5 * (C - Cn)
    # Cn_thr_quart = Cn + 0.75 * (C - Cn)
    Cn_thr_quart.x.scatter_forward()
    Cn_quart.x.scatter_forward()
    Cn_half.x.scatter_forward()

    evolEqG(Cn, Cvn, k1)
    k1.x.scatter_forward()
    evolEqG(Cn_half, Cvn + k1 * dt / 2, k2)
    k2.x.scatter_forward()
    evolEqG(Cn_quart, Cvn + 1.0 / 16 * dt * (3 * k1 + k2), k3)
    k3.x.scatter_forward()
    evolEqG(Cn_half, Cvn + dt / 2.0 * k3, k4)
    k4.x.scatter_forward()
    evolEqG(Cn_thr_quart, Cvn + 3.0 / 16 * dt * (-k2 + 2.0 * k3 + 3.0 * k4), k5)
    k5.x.scatter_forward()
    evolEqG(C, Cvn + (k1 + 4.0 * k2 + 6.0 * k3 - 12.0 * k4 + 8.0 * k5) * dt / 7.0, k6)
    k6.x.scatter_forward()
    # print(type(k1.vector))

    # copy Cvn into Cv to start the update
    with Cv.vector.localForm() as cv, Cvn.vector.localForm() as cvn:
        cvn.copy(cv)
    # Cv.vector.axpy(1.0, Cvn.vector)
    # Cv.x.scatter_forward()
    # print(type(Cv.vector))
    Cv.vector.axpby(
        dt / 90.0,
        1.0,
        (
            7.0 * k1.vector
            + 32.0 * k3.vector
            + 12.0 * k4.vector
            + 32.0 * k5.vector
            + 7.0 * k6.vector
        ),
    )
    Cv.x.scatter_forward()
    # project(kfinal, Cv, degree=metadata["quadrature_degree"])
    return 1


def calculate_norm_tensor_function(C):
    norm: float = 0.0
    for i in range(3):
        for j in range(3):
            norm += fem.assemble_scalar(fem.form(C[i, j] * C[i, j] * dx))
    return np.sqrt(norm)


# mesh and inputs
comm = MPI.COMM_WORLD
mesh = create_unit_cube(comm, 2, 2, 2, ctype.tetrahedron)
mesh.topology.create_connectivity(2, 3)

V = fem.VectorFunctionSpace(mesh, ("CR", 1))
Vp = fem.VectorFunctionSpace(mesh, ("CG", 1))
u = fem.Function(V, name="u")
Cn = fem.Function(V, name="Cn")
uplot = fem.Function(Vp, name="disp")
v = TestFunction(V)
du = TrialFunction(V)

WCv = TensorElement(
    "Quadrature",
    mesh.ufl_cell(),
    degree=metadata["quadrature_degree"],
    quad_scheme="default",
)
Ws = TensorElement(
    "Quadrature",
    mesh.ufl_cell(),
    degree=metadata["quadrature_degree"],
    quad_scheme="default",
)
VCv = fem.FunctionSpace(mesh, WCv)
VS = fem.FunctionSpace(mesh, Ws)
CCv = fem.Function(VCv, name="Cv")
Cvn = fem.Function(VCv, name="Cvn")
C_quart = fem.Function(VCv, name="C_quarter")
C_thr_quart = fem.Function(VCv, name="C_thr_quarter")
C_half = fem.Function(VCv, name="C_half")

C = fem.Function(VCv, name="C")
Cn = fem.Function(VCv, name="Cn")
Cv_iter = fem.Function(VCv, name="Cv_iter")
del_Cv = fem.Function(VCv, name="delCv")
S = fem.Function(VS, name="S")


# cache to hold intermediate states
k_cache = [fem.Function(VCv, name=f"k{i:d}") for i in range(1, 7)]

# material parameters
kap_by_mu = fem.Constant(mesh, st(10.0 ** 3))
mu1 = fem.Constant(mesh, 13.54 * 10 ** 3)
mu2 = fem.Constant(mesh, 1.08 * 10 ** 3)
mu_pr = kap_by_mu * (mu1 + mu2)  # make this value very high
alph1 = fem.Constant(mesh, 1.0)
alph2 = fem.Constant(mesh, -2.474)
m1 = fem.Constant(mesh, 5.42 * 10 ** 3)
m2 = fem.Constant(mesh, 20.78 * 10 ** 3)
a1 = fem.Constant(mesh, -10.0)
a2 = fem.Constant(mesh, 1.948)
K1 = fem.Constant(mesh, 3507.0 * 10 ** 3)
K2 = fem.Constant(mesh, 10 ** (-6))
bta1 = fem.Constant(mesh, 1.852)
bta2 = fem.Constant(mesh, 0.26)
eta0 = fem.Constant(mesh, 7014.0 * 10 ** 3)
etaInf = fem.Constant(mesh, 0.1 * 10 ** 3)  # 0.1

# integration measures
dx = Measure("dx", metadata=metadata)
dS = Measure("dS", metadata=metadata)

# stabilization constant
qvals = 5.0 * (mu1 + mu2 + m1 + m2)

# loading rate
ldot = 0.05

# array of `time` like variable for stepping in loading
timeVals = np.linspace(0, 4.0 / ldot, 203)

# delta t values (will change when doing adaptive computation)
dt = timeVals[1] - timeVals[0]

# array of displacement values to apply at right boundary
stretchVals = np.hstack(
    (
        ldot * timeVals[: len(timeVals) // 2],
        ldot * (-timeVals[len(timeVals) // 2 :] + 2 * timeVals[len(timeVals) // 2]),
    )
)

svals = np.zeros_like(stretchVals)

plt.plot(timeVals, stretchVals)
plt.savefig("stretchesVisco.png")
plt.close()

# stabilization parameters
h = FacetArea(mesh)
h_avg = avg(h)  # can also use avg(h)


# new variable name to take derivatives
FF = Identity(3) + grad(u)
CC = FF.T * FF
Fv = variable(FF)
S = diff(freeEnergy(Fv.T * Fv, CCv), Fv)  # first PK stress
dl_interp(CC, C)
dl_interp(CC, Cn)

my_identity = grad(SpatialCoordinate(mesh))

dl_interp(my_identity, CCv)
dl_interp(my_identity, Cvn)
dl_interp(my_identity, C_quart)
dl_interp(my_identity, C_thr_quart)
dl_interp(my_identity, C_half)

a_uv = (
    derivative(freeEnergy(CC, CCv), u, v) * dx
    + qvals / h_avg * dot(jump(u), jump(v)) * dS
)
jac = derivative(a_uv, u, du)

# assign DirichletBC
left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
bottom_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, bottom)
back_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, back)

left_dofs = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)
right_dofs = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, right_facets)
back_dofs = fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, back_facets)
bottom_dofs = fem.locate_dofs_topological(
    V.sub(1), mesh.topology.dim - 1, bottom_facets
)

right_disp = fem.Constant(mesh, 0.0)
ul = fem.dirichletbc(st(0), left_dofs, V.sub(0))
ub = fem.dirichletbc(st(0), bottom_dofs, V.sub(1))
ubak = fem.dirichletbc(st(0), back_dofs, V.sub(2))
ur = fem.dirichletbc(right_disp, right_dofs, V.sub(0))

bcs = [ul, ub, ubak, ur]
problem = fem.NonlinearProblem(a_uv, u, bcs=bcs, J=jac)
solver = NewtonSolver(comm, problem)
solver.rtol = 1.0e-5
solver.atol = 1.0e-7
solver.convergence_criterion = "residual"
solver.max_it = 200
solver.report = True
ksp = solver.krylov_solver
opts = PETSc.Options()
opts.getAll()
opts.view()
option_prefix = ksp.getOptionsPrefix()

opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

ksp.setFromOptions()

wfil = XDMFFile(comm, "disp_Visco_0_05_realEta_Fixed.xdmf", "w")
wfil.write_mesh(mesh)

with open("sfiles.txt", "w") as fil:
    fil.write(f"0.0, 0.0\n")

for i, t in enumerate(timeVals):
    right_disp.value = stretchVals[i]
    print(f"Load step: {i+1}")
    set_output_file("viscoTrialrunW.log")
    num_its, converged = solver.solve(u)

    print(f"Newton iteration: {num_its}")
    dl_interp(CC, C)
    k_terms(dt, C, Cn, Cvn, C_quart, C_half, C_thr_quart, CCv, k_cache)
    CCv.vector.copy(result=Cv_iter.vector)
    CCv.x.scatter_forward()
    with del_Cv.vector.localForm() as loc:
        loc.set(0.0)
    del_Cv.vector.axpy(1.0, Cv_iter.vector - Cvn.vector)
    del_Cv.x.scatter_forward()
    norm_delCv = del_Cv.x.norm()
    print(f"norm(delta_Cv): {norm_delCv:.5e}")
    itr: int = 0
    max_iters: int = 10
    while norm_delCv > 1.0e-5 and itr < max_iters:
        Cv_iter.vector.copy(result=CCv.vector)
        set_output_file("viscoTrialrunW.log")
        num_its, converged = solver.solve(u)  # already scatters `u` to ghost processes
        u.x.scatter_forward()

        dl_interp(CC, C)
        k_terms(dt, C, Cn, Cvn, C_quart, C_half, C_thr_quart, CCv, k_cache)
        with del_Cv.vector.localForm() as loc:
            loc.set(0.0)
        del_Cv.vector.axpy(1.0, CCv.vector - Cv_iter.vector)
        norm_delCv = del_Cv.x.norm()

        print(f"Staggered Iteration: {itr+1}, norm_deCv: {norm_delCv:.5e}")
        print(f"Newton iterations: {num_its}, Converged: {converged}")
        itr += 1

    C.vector.copy(result=Cn.vector)  # assign Cn <-- C
    CCv.vector.copy(result=Cvn.vector)  # assign Cvn <-- Cv (converged)
    norm_Cv = calculate_norm_tensor_function(CCv)
    print(f"Cv norm is: {norm_Cv:.8e}")
    svals[i] = fem.assemble_scalar(fem.form(S[0, 0] * dx))  # average stress
    print(f"stress: {svals[i]:.6e}")
    with open("sfiles.txt", "a") as fil:
        fil.write(f"{t:.8f}, {svals[i]:.8f}\n")
    dl_interp(u, uplot)
    wfil.write_function(uplot, t=t)

plt.plot(stretchVals, svals)
plt.savefig("stresses.png")
plt.close()
