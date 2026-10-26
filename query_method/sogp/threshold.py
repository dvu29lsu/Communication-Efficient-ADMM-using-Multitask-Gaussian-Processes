import numpy as np

class GeometricDecayThreshold:
      # psi_i^{k0} = iota * V_i^{k0}
      # psi_i^k = psi_i^{k0} * alpha^(k-k0), k>=k0
      # V_i^{k0} is provided at first GP-use as paper

    def __init__(self, n_agents: int, iota: float = 0.5, alpha: float = 0.98):

        self.n_agents = int(n_agents)
        self.iota = float(iota) # proportion we decide to take from the maximum variance
        self.alpha = float(alpha)

        self.k0 = np.full(self.n_agents, -1, dtype=int)

        # initial psi_k for each agents
        self.psi0 = np.zeros(self.n_agents, dtype=float)

    # initialize based on the first prediction
    def ensure_initialized(self, i: int, k: int, V_k0: float):
        if self.k0[i] < 0:
            self.k0[i] = int(k) # save the first iteration use GP so can apply the decay
            self.psi0[i] = float(self.iota * float(V_k0))

    # adaptive threshold reduction
    def value(self, i: int, k: int) -> float:
        if self.k0[i] < 0:
            return float("inf") # this part need to be adapt as the first value of the psi_0 not inf
        return float(self.psi0[i] * (self.alpha ** (k - self.k0[i])))
