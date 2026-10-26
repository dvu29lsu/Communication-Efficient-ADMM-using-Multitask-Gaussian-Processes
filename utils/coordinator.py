## admm coordinator
import numpy as np
from admm.sharing_admm import SharingADMM

class Coordinator:
    def __init__(self, gamma_list, theta_list, admm_cfg, burn_in=5, local_models=None):
        ## initinalize the solver using admm algorithm

        self.solver = SharingADMM(
            gamma_list, theta_list,
            zeta=admm_cfg.zeta, rho=admm_cfg.rho,
            eps_abs=admm_cfg.eps_abs, eps_rel=admm_cfg.eps_rel,
            n_iters=admm_cfg.max_iters,
            local_models=local_models
        )

        self.burn_in = burn_in ## number of the data point collect before adapt GP

        # comm stats
        self.comm_total = 0 # total agent-communications
        self.comm_per_agent = None  # filled after run
        self.comm_per_iter = None
        self.comm_rounds_iter = 0  ##iters where any agent communicated

    def run(self, provider, verbose=True, log_every=1):
        out = self.solver.run_moreau_admm(
            moreau_grad_provider=provider,
            burn_in=self.burn_in,
            verbose=verbose,
            log_every=log_every
        )

        # comm summaries from solver logs
        masks = out["comm_mask_per_iter"]  # list of bool arrays
        N = self.solver.N
        per_agent = np.zeros(N, dtype=int)
        for m in masks:
            per_agent += m.astype(int)

        per_iter = np.array(out["comm_per_iter"], dtype=int)


        # total communication round for all agents, per agent comm and per iteration 
        self.comm_total = int(per_iter.sum())
        self.comm_per_agent = per_agent
        self.comm_per_iter = per_iter
        self.comm_rounds_iter = int(np.sum(per_iter > 0))

        
        out["comm_total"] = self.comm_total
        out["comm_rounds_iter"] = self.comm_rounds_iter
        out["comm_per_agent"] = self.comm_per_agent
        out["comm_per_iter"] = self.comm_per_iter

        # also include cumulative
        out["comm_cum"] = np.cumsum(per_iter).tolist()

        return out