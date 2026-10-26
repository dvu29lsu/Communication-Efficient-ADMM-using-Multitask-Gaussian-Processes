import numpy as np
from scipy.optimize import minimize

from admm.exact_quadsolver import solving_moreau_exact_update
from admm.local_models import QuadraticLocalModel, Poly4LocalModel, DirectionalPoly4LocalModel

class MoreauOracle:
    def __init__(
        self,
        model,
        rho,
        solver_method="trust-exact",
        solver_tol=1e-6,
        solver_maxiter=300,
    ):
        self.model = model
        self.rho = float(rho)
        self.solver_method = solver_method
        self.solver_tol = solver_tol
        self.solver_maxiter = int(solver_maxiter)

    def prox(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)

        # if model is quadratic function in-form (x-theta)^T Gamma (x-theta), use the exact solver
        if isinstance(self.model, QuadraticLocalModel):
            print("Using exact solver for quadratic local model")
            g = solving_moreau_exact_update(
                z, self.model.Gamma, self.model.theta, self.rho
            )
            y = z - (1.0 / self.rho) * g
            return y

        # otherwise, use scipy minimize to solve the prox problem
        def obj(x):
            return self.model.value(x) + 0.5 * self.rho * np.dot(x - z, x - z)

        def jac(x):
            return self.model.grad(x) + self.rho * (x - z)

        def hess(x):
            return self.model.hess(x) + self.rho * np.eye(len(x))

        res = minimize(
            obj,
            x0=z.copy(),
            jac=jac,
            hess=hess,
            method=self.solver_method,
            tol=self.solver_tol,
            options={"maxiter": self.solver_maxiter},
        )

        print("Prox solve success with scipy")

        if not res.success:
            raise RuntimeError(f"prox solve failed: {res.message}")

        return np.asarray(res.x, dtype=float)

    def moreau_value_and_grad(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        y = self.prox(z)
        g = self.rho * (z - y)
        f_gamma = self.model.value(y) + 0.5 * self.rho * np.dot(y - z, y - z)
        return float(f_gamma), g.astype(float)


def build_local_models(
    gamma_list,
    theta_list,
    local_model="quadratic",
    lambda4=0.5,
    A=None,
):
    models = []

    for G, th in zip(gamma_list, theta_list):
        if local_model == "quadratic":
            print("Building QuadraticLocalModel")
            models.append(QuadraticLocalModel(G, th))

        elif local_model == "poly4":
            print("Building Poly4LocalModel with lambda4 =", lambda4)
            models.append(Poly4LocalModel(G, th, lambda4))

        elif local_model == "dir_poly4":
            if A is None:
                raise ValueError("A must be provided for local_model='dir_poly4'")
            print("Building DirectionalPoly4LocalModel with lambda4 =", lambda4)
            models.append(DirectionalPoly4LocalModel(G, th, lambda4, A))

        else:
            raise ValueError(f"Unknown local_model: {local_model}")

    return models

def build_moreau_oracles(
    gamma_list,
    theta_list,
    rho,
    local_model="quadratic",
    lambda4=0.5,
    A=None, ## direction matrix
    solver_method="trust-exact",
    solver_tol=1e-4,
    solver_maxiter=300,
):
    print("constructing Moreau oracles with local_model =", local_model)

    models = build_local_models(
        gamma_list=gamma_list,
        theta_list=theta_list,
        local_model=local_model,
        lambda4=lambda4,
        A=A,
    )

    return [
        MoreauOracle(
            model=m,
            rho=rho,
            solver_method=solver_method,
            solver_tol=solver_tol,
            solver_maxiter=solver_maxiter,
        )
        for m in models
    ]