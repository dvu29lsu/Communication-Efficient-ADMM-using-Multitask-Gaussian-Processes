
from dataclasses import dataclass

@dataclass
class ADMMConfig:
    zeta: float = 1.0
    rho: float = 10
    eps_abs: float = 1e-4
    eps_rel: float = 1e-3
    max_iters: int = 10
    burn_in: int = 5

# @dataclass
# class GPConfig:
#     learning_rate: float = 0.01
#     max_epochs: int = 500
#     noise_init: float = 1e-4
#     lengthscale_init: float = 1.0
#     outputscale_init: float = 1.0