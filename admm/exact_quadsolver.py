import numpy as np
from scipy.io import loadmat

def solving_moreau_exact_update(z, Gamma, theta, rho, L = None):
    # (gradient_moreau) =  rho * (z_i -  proximal_point)
    # proximal named y here
    # gradient_moreau_env = beta
    # beta_i(z) = rho*(z_i - y)
    # z_i = x_i + z_bar - x_bar - u_bar
    # local update is solving the linear equation: (2G + rho*I)x_i = 2G*theta + rho*(x_i + z_bar - x_bar - u_bar)
    # divide by the rho: ((2/rho)*G + I)x_i = (2/rho)*G*theta + (z_i)
    p = Gamma.shape[0]
    A = np.eye(p) + (2.0/rho) * Gamma
    if L is None:
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(A + 1e-10*np.eye(p))
    b = z + (2.0/rho) * (Gamma @ theta)
    y = np.linalg.solve(L.T, np.linalg.solve(L, b))
    return rho * (z - y)