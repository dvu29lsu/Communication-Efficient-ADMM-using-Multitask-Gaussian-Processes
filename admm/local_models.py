import numpy as np
from admm.proximal_ops import soft_threshold

class QuadraticLocalModel:
    def __init__(self, Gamma, theta):
        self.Gamma = np.asarray(Gamma, dtype=float)
        self.theta = np.asarray(theta, dtype=float).reshape(-1)

    def value(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        return float(u.T @ self.Gamma @ u)

    def grad(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        return 2.0 * (self.Gamma @ u)

    def hess(self, x):
        return 2.0 * self.Gamma


class Poly4LocalModel:
    def __init__(self, Gamma, theta, lambda4):
        self.Gamma = np.asarray(Gamma, dtype=float)
        self.theta = np.asarray(theta, dtype=float).reshape(-1)
        self.lambda4 = float(lambda4)

    def value(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        return float(u.T @ self.Gamma @ u + self.lambda4 * np.sum(u**4))

    def grad(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        return 2.0 * (self.Gamma @ u) + 4.0 * self.lambda4 * (u**3)

    def hess(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        return 2.0 * self.Gamma + np.diag(12.0 * self.lambda4 * (u**2))
    

# directional quartic

class DirectionalPoly4LocalModel:
    def __init__(self, Gamma, theta, lambda4, A):
        self.Gamma = np.asarray(Gamma, dtype=float)
        self.theta = np.asarray(theta, dtype=float).reshape(-1)
        self.lambda4 = float(lambda4)

        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)

        if A.shape[1] != self.theta.shape[0]:
            raise ValueError("A must have shape (r, p) with p = len(theta)")

        self.A = A

    def value(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        Au = self.A @ u
        quad = u.T @ self.Gamma @ u
        quartic = self.lambda4 * np.sum(Au**4)
        return float(quad + quartic)

    def grad(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        Au = self.A @ u
        quad_grad = 2.0 * (self.Gamma @ u)
        quartic_grad = 4.0 * self.lambda4 * (self.A.T @ (Au**3))
        return quad_grad + quartic_grad

    def hess(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = x - self.theta
        Au = self.A @ u

        H = 2.0 * self.Gamma.copy()
        for m in range(self.A.shape[0]):
            a = self.A[m].reshape(-1, 1)
            H += 12.0 * self.lambda4 * (Au[m]**2) * (a @ a.T)
        return H