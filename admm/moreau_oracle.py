import numpy as np
from scipy.optimize import root

from admm.exact_quadsolver import solving_moreau_exact_update
from admm.local_models import QuadraticLocalModel, Poly4LocalModel, DirectionalPoly4LocalModel

# fixed settings of the quartic prox solver, shared by every method (same as the ICC paper runs)
QUARTIC_ROOT_XTOL = 1e-10        # step tolerance of scipy root (hybr)
QUARTIC_ROOT_MAXFEV = 300        # max function evaluations of scipy root (hybr)
QUARTIC_STATIONARITY_TOL = 1e-8  # accept a prox point only if ||grad f(x) + rho (x - z)||_inf <= this

class MoreauOracle:
    def __init__(self, model, rho):
        self.model = model
        self.rho = float(rho)

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

        # quartic models have no closed form -> solve numerically
        if isinstance(self.model, (Poly4LocalModel, DirectionalPoly4LocalModel)):
            return self._prox_quartic(z)

        raise TypeError(f"Unsupported local model: {type(self.model).__name__}")

    def _prox_quartic(self, z):
        # prox(z) = argmin_x f(x) + rho/2 ||x - z||^2 is strongly convex and smooth,
        # so its unique minimizer is the root of the stationarity equation
        #   F(x) = grad f(x) + rho (x - z) = 0,   with Jacobian  hess f(x) + rho I  (SPD)
        def stationarity(x):
            return self.model.grad(x) + self.rho * (x - z)

        def stationarity_jac(x):
            return self.model.hess(x) + self.rho * np.eye(len(x))

        res = root(
            stationarity,
            x0=z.copy(), # z is close to the prox point for large rho
            jac=stationarity_jac,
            method="hybr",
            options={"xtol": QUARTIC_ROOT_XTOL, "maxfev": QUARTIC_ROOT_MAXFEV},
        )

        # accept the point only if it really satisfies the stationarity equation,
        # so the "exact" oracle is exact for every method (not just res.success)
        x = np.asarray(res.x, dtype=float)
        residual = float(np.linalg.norm(stationarity(x), ord=np.inf))

        if not np.all(np.isfinite(x)) or not np.isfinite(residual) or residual > QUARTIC_STATIONARITY_TOL:
            raise RuntimeError(
                f"quartic prox solve failed: residual={residual:.3e} "
                f"(limit {QUARTIC_STATIONARITY_TOL:.0e}), message={res.message}"
            )

        return x

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
):
    print("constructing Moreau oracles with local_model =", local_model)

    models = build_local_models(
        gamma_list=gamma_list,
        theta_list=theta_list,
        local_model=local_model,
        lambda4=lambda4,
        A=A,
    )

    return [MoreauOracle(model=m, rho=rho) for m in models]