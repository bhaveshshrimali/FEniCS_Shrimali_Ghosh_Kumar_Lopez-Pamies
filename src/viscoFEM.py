# Rubber viscoelastic model for isotropic elastomers:
# ref: http://pamies.cee.illinois.edu/Publications_files/CRM_2016.pdf
# Material parameters: 14,
# * Elasticity: mu1, mu2, alph1, alph2,
# * Viscoelasticity: m1, m2, a1, a2, K1, K2, beta1, beta2, eta_0, etaInf
# Each of the functions has corresponding docstring
import os
from dolfin import *
import numpy as np
from time import time
set_log_level(30)
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["representation"] = "uflacs"
parameters["ghost_mode"] = "shared_facet"

ffc_options = {"optimize": True,
            "eliminate_zeros": True,
            "precompute_basis_const": True,
            "precompute_ip_const": True}
metadata = {"quadrature_degree": 3}
snes_solver_parameters = {"nonlinear_solver": "snes",
                        "snes_solver": {"linear_solver": "lu",
                                        "preconditioner": "petsc_amg",
                                        "maximum_iterations": 20,
                                        "report": True,
                                        "line_search": 'basic',
                                        "error_on_nonconvergence": False,
                                        "relative_tolerance": 1e-7,
                                        "absolute_tolerance": 1e-8}}

comm = MPI.comm_world


# Mesh and material properties
msh = UnitCubeMesh(2, 2, 2)

# x
left = CompiledSubDomain("near(x[0],0.) && on_boundary")
back = CompiledSubDomain("near(x[2],0.) && on_boundary")
bottom = CompiledSubDomain("near(x[1],0.) && on_boundary")
frontFace = CompiledSubDomain("near(x[2],1.) && on_boundary")


def freeEnergy(u, Cv):
    """[summary]
        Given, `u` and `Cv` this function
        calculates the sum of equilibrium and non-equilibrium
        free energies
    Args:
        u ([dolfin.Function]): [FE displacement field]
        Cv ([dolfin.Function]): [FE internal variable]

    Returns:
        [psiEq + psiNEq]: [Sum of energies of type `ufl.algebra.Sum`]
    """
    F = Identity(len(u)) + grad(u)
    C = F.T * F
    J = det(F)
    I1 = tr(C)
    Ce = C * inv(Cv)
    Ie1 = tr(Ce)
    Je = J / sqrt(det(Cv))

    psiEq = 3**(1 - alph1) / (2. * alph1) * mu1 * (I1**alph1 - 3**alph1) + 3**(1 - alph2) / \
        (2. * alph2) * mu2 * (I1**alph2 - 3**alph2) - \
        (mu1 + mu2) * ln(J) + mu_pr / 2 * (J - 1)**2

    psiNeq = 3**(1 - a1) / (2. * a1) * m1 * (Ie1**a1 - 3**a1) + \
        3**(1 - a2) / (2. * a2) * m2 * (Ie1**a2 - 3**a2) - (m1 + m2) * ln(Je)

    return psiEq + psiNeq


def stressPiola(u, Cv):
    """[summary]
        Given `u` and `Cv` this function calculates
        the first Piola-Kirchhoff stress
    Args:
        u ([dolfin.Function]): [FE displacement field]
        Cv ([dolfin.Function]): [FE internal variable]

    Returns:
        [SEq + SNEq]: [First PK stress tensor
        from equilibrium and non-equilibrium states
        of type `ufl.algebra.Sum`]
    """
    F = Identity(len(u)) + grad(u)
    C = F.T * F
    J = det(F)
    Finv = inv(F)
    I1 = tr(C)
    J = det(F)
    Ce = C * inv(Cv)
    Ie1 = tr(Ce)
    SEq = (mu1 * (I1 / 3.)**(alph1 - 1) + mu2 * (I1 / 3.)**(alph2 - 1)) * \
        F - (mu1 + mu2) * Finv.T + mu_pr * J * (J - 1.) * Finv.T
    SNeq = (m1 * (Ie1 / 3.)**(a1 - 1.) + m2 * (Ie1 / 3.)**(a2 - 1)) * F * \
        inv(Cv) - (m1 + m2) * Finv.T

    return SEq + SNeq


def evolEqG(u, Cv):
    """[summary]
        Given `u` and `Cv` this function calculates the
        RHS of the evolution equation G(C, Cv) as described
        in the paper referenced above
    Args:
        u ([dolfin.Function]): [FE displacement field]
        Cv ([dolfin.Function]): [FE internal variable]

    Returns:
        [G(C(u), Cv)]: [RHS of the evolution equation]
    """
    F = Identity(len(u)) + grad(u)
    C = F.T * F
    Iv1 = tr(Cv)
    Ce = C * inv(Cv)
    CvinvC = inv(Cv) * C
    Ie1 = tr(Ce)

    # define etaK
    A2 = m1 * (Ie1 / 3)**(a1 - 1) + m2 * (Ie1 / 3)**(a2 - 1)

    G = A2 / (etaInf + (eta0 - etaInf + K1 * (Iv1**bta1 - 3**bta1)) / (1 + (K2 * \
              ((-Ie1**2 / 6. + 1. / 2 * inner(CvinvC, Ce)) * A2**2))**bta2)) * (C - Ie1 / 3. * Cv)
    G = local_project(G, VQe)
    return G


def k_terms(dt, u, un, Cvn):
    """[summary]
        Given u(tn), Cv(tn), u(t_k, r) and dt this function
        calculates the terms k_i (i=1 to 6) as described in
        the paper above
    Args:
        dt ([float]): [time incremement]
        u ([dolfin.Function]): [FE displacement field]
        un ([dolfin.Function]): [FE displacement field at t=tn]
        Cvn ([dolfin.Function]): [FE internal variable at t=tn]

    Returns:
        [k1 + ... + k6]: [ufl.algebra.Sum]
    """
    un_quart = un + 0.25 * (u - un)
    un_half = un + 0.5 * (u - un)
    un_thr_quart = un + 0.75 * (u - un)
    k1 = evolEqG(un, Cvn)
    k2 = evolEqG(un_half, Cvn + k1 * dt / 2)
    k3 = evolEqG(un_quart, Cvn + 1. / 16 * dt * (3 * k1 + k2))
    k4 = evolEqG(un_half, Cvn + dt / 2. * k3)
    k5 = evolEqG(un_thr_quart, Cvn + 3. / 16 * dt * (-k2 + 2. * k3 + 3. * k4))
    k6 = evolEqG(u, Cvn + (k1 + 4. * k2 + 6. *
                 k3 - 12. * k4 + 8. * k5 ) * dt / 7.)

    kfinal = dt / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)
    kfinal = local_project(kfinal, VQe)
    return kfinal


def local_project(v, V):
    """[summary]
        Helper function to do a local projection element-wise
        Useful for DG-spaces since the projection can be done
        for all the elements in parallel
    Args:
        v ([dolfin.Funcion]): [function to be projected]
        V ([dolfin.Function]): [target `dolfin.FunctionSpace` to be projected on]

    Returns:
        [dolfin.Function]: [target function after projection]
    """
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_) * dx(metadata=metadata)
    b_proj = inner(v, v_) * dx(metadata=metadata)
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u


def evaluate_function(u, x):
    """[summary]
        Helps evaluated a function at a point `x` in parallel
    Args:
        u ([dolfin.Function]): [function to be evaluated]
        x ([Union(tuple, list, numpy.ndarray)]): [point at which to evaluate function `u`]

    Returns:
        [numpy.ndarray]: [function evaluated at point `x`]
    """
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = msh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None

    # Gather the results on process 0
    comm = msh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array(
            [y for y in computed_u if y is not None], dtype=np.double)
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # Broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u

# Choosing an appropriate function space for the displacements
# should be able to capture the incompressibility well.


Ve = VectorElement("CR", msh.ufl_cell(), 1)
Qe = TensorElement("DG", msh.ufl_cell(), 0)
VQe = FunctionSpace(msh, Qe)
W = FunctionSpace(msh, Ve)
un = Function(W, name="displacement")
ures = Function(W, name="result")
u = Function(W)
utrial = TrialFunction(W)
delu = TestFunction(W)
Cvtrial = Function(VQe)
Cvn = Function(VQe, name='Cvn')
Cv = Function(VQe, name='Cv')


# initialize the displacements and the internal variable
un.vector()[:] = 0.
Cv.assign(project(Identity(3), VQe))
Cvn.assign(project(Identity(3), VQe))
strtch = Constant(0.)
bcl = DirichletBC(W.sub(0), Constant(0), left)
bcb = DirichletBC(W.sub(1), Constant(0.), bottom)
bcback = DirichletBC(W.sub(2), Constant(0.), back)
bacFront = DirichletBC(W.sub(2), strtch, frontFace)
bcs = [
    bcl, bcb, bcback, bacFront
]

kap_by_mu = 10.**3

# VHB4910
mu1 = Constant(13.54 * 10**3)
mu2 = Constant(1.08 * 10**3)
mu_pr = Constant(kap_by_mu * (mu1 + mu2))  # make this value very high
alph1 = Constant(1.)
alph2 = Constant(-2.474)
m1 = Constant(5.42 * 10**3)
m2 = Constant(20.78 * 10**3)
a1 = Constant(-10.)
a2 = Constant(1.948)
K1 = Constant(3507 * 10**3)
K2 = Constant(10**(-6))
bta1 = Constant(1.852)
bta2 = Constant(0.26)
eta0 = Constant(7014 * 10**3)
etaInf = Constant(0.1 * 10**3)  # 0.1

# Nitrile
# mu1 = Constant(1.08*10**6)
# mu2 = Constant(0.017 * 10**6)
# mu_pr = Constant(kap_by_mu*13.54*10**3)   #make this value very high
# alph1 = Constant(0.26)
# alph2 = Constant(7.68)
# m1 = Constant(1.57 * 10**6)
# m2 = Constant(0.59*10**6)
# m_pr = Constant(10**4*20.78*10**3)
# a1 = Constant(-10.)
# a2 = Constant(7.53)
# K1 = Constant(442 * 10**6)
# K2 = Constant(0.)  #1289.49*10**(-12)
# bta1 = Constant(3)
# bta2 = Constant(1.929)
# eta0 = Constant(2.11 * 10**6)
# etaInf = Constant(0.1* 10**6)  #0.1


qvals = (mu1 + mu2 + m1 + m2)
ldot = 0.05
target_stretch = 3.0
Tfinal = (target_stretch-1.)/ldot * 2.
timeVals = np.linspace(0, Tfinal, 1003)
dt = timeVals[1] - timeVals[0]
stretchVals = np.hstack((ldot * timeVals[:len(timeVals) // 2], ldot *
                         (-timeVals[len(timeVals) // 2:] + 2 * timeVals[len(timeVals) // 2])))
h = FacetArea(msh)
h_avg = 1. / 2 * (h('+') + h('-'))  # can also use avg(h)
a_uv = derivative(freeEnergy(u, Cv), u, delu) * dx + qvals / \
    h_avg * inner(jump(u), jump(delu)) * dS

Jac = derivative(a_uv, u, utrial)
F = NonlinearVariationalProblem(a_uv, u, bcs, J=Jac)
solver = NonlinearVariationalSolver(F)
solver.parameters.update(snes_solver_parameters)

Vplot = VectorFunctionSpace(msh, 'CG', 1)
VQePlot = TensorFunctionSpace(msh, 'DG', 0)
uplot = Function(Vplot, name='disp')
Cvplot = Function(VQePlot, name='Cv')
uiter = Function(W, name='uk_1')
Cviter = Function(VQe, name='Cvk1')
stressplot = Function(VQePlot, name='S')


dispVals = np.zeros((timeVals.shape[0], 3))
sPiolaVals = np.zeros((timeVals.shape[0], 3))
wfil = XDMFFile('disp_Visco_0_05_realEta_Fixed.xdmf')
wfil.parameters["flush_output"] = True
wfil.parameters["functions_share_mesh"] = True
wfil.parameters["rewrite_function_mesh"] = False

pt = (0.5, 0.5, 0.5)

for i, tk in enumerate(timeVals):
    strtch.assign(stretchVals[i])
    Cvtrial.assign(Cvn)
    print('u_3: {}'.format(float(strtch)))
    solver.solve()
    Cviter = Cvn + k_terms(dt, u, un, Cvn)
    Cviter = local_project(Cviter, VQe)
    norm_delCv = norm(Cvtrial.vector() - Cviter.vector())
    iterCount = 0
    while norm_delCv > 1.e-5 and iterCount <= 10:
        Cvtrial.assign(Cviter)
        Cv.assign(Cviter)
        solver.solve()
        uiter.assign(u)
        Cviter = Cvn + k_terms(dt, uiter, un, Cvn)
        Cviter = local_project(Cviter, VQe)
        iterCount += 1
        norm_delCv = norm(Cvtrial.vector() - Cviter.vector())
        print(
            "Staggered Iteration: {}, Norm = {}".format(
                iterCount,
                norm_delCv))

    un.assign(u)
    Cvn.assign(Cviter)
    stressplot.assign(local_project(stressPiola(un, Cvn), VQePlot))

    uplot.assign(project(un, Vplot))
    Cvplot.assign(local_project(Cvn, VQePlot))
    u_at_111 = evaluate_function(uplot, pt)
    s_at_111 = evaluate_function(stressplot, pt)
    wfil.write(uplot, tk)
    wfil.write(Cvplot, tk)
    dispVals[i] = np.array([u_at_111[0], u_at_111[1], u_at_111[2]], float)
    sPiolaVals[i] = np.array([s_at_111[0], s_at_111[4], s_at_111[8]], float)

# save results for plotting
np.savetxt(f'SvL_{ldot:.2f}.txt', np.vstack(
    (1. + stretchVals, sPiolaVals[:, 2])))
np.savetxt(f'Lvt_{ldot:.2f}.txt', np.vstack((timeVals, 1. + stretchVals)))
from matplotlib import pyplot as plt
plt.plot(1+stretchVals, sPiolaVals[:, 2])
plt.savefig("testVHB.png")
print(f"dt: {timeVals[1] - timeVals[0]:.4f}")
