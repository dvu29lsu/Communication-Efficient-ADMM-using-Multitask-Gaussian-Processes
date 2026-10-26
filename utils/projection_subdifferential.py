import numpy as np
import cvxpy as cp

# Date: Jan 28, 26 - Long Vu

def project_to_ellipsoid(g0, A, B, c, solver="SCS", max_iters = 2000):

    # solving the min ||g - g0|| 
    # trying to find the g inside the intersection of ellipsoids

    p = g0.shape[0]

    m = B.shape[1]

    g = cp.Variable((p))

    cons = []

    # define the constrains for probelm ||g - g0||_2^2
    for j in range(m):
        exp = cp.quad_form(g, A) + 2 * B[:, j].T @ g + c[j]
        cons.append(exp <= 0)

    prob = cp.Problem(cp.Minimize(cp.norm(g - g0)), cons)

    prob.solve(solver=solver, max_iters = max_iters, verbose=False)

    # havign solution or not
    if g.value is not None:

        return np.array(g.value).reshape(-1), True

    else:
        return g0, False
    