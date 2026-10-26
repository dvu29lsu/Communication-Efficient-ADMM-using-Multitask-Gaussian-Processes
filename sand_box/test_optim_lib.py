import cvxpy as cp
import numpy as np
# define the problem

p = 5

gamma = 1.0

A = gamma * np.eye(p)

b = np.random.randn(p)

c = 0.1

g = cp.Variable(p)


# cosntrain
exp = cp.quad_form(g, A) + 2 * b @ g + c

prob = cp.Problem(cp.Minimize(cp.norm(g - np.random.randn(p))), [exp <= 0])

prob.solve(solver="SCS", max_iters = 2000, verbose=False)

print("Optimal parameter g:", g.value)

print("is dcp?", prob.is_dcp())
print("is dpcp?", prob.is_dqcp())
