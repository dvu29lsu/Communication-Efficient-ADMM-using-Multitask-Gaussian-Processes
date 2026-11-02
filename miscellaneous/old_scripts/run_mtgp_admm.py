import numpy as np
import torch
import gpytorch
import os
import time
import json
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_json
from utils.json_processing import process_output_to_jsonl
from utils.set_seed import set_seed

# import the load method
from query_method.mogp.mtgp_provider import MTGPMoreauGradProvider

set_seed(123)
# torch.set_default_dtype(torch.float32)

# try the float64: same GP precision as the cluster-wise MTGP and SOGP runners
torch.set_default_dtype(torch.float64)

# config for the ADMM
cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-6, eps_rel=1e-3, burn_in=5)

# run mtgp admm

def run_mtgp(json_path: str, burn_in: int, iota: float = 1.3, alpha: float = 0.96, train_iter: int = 300, lr: float = 0.03, Q: int = 5, rank: int =1):
    set_name = os.path.basename(json_path) # dataset name for the printed summary

    # global MTGP shares one model across all agents, so the cluster labels are not used
    gamma_list, theta_list, _ = load_data_from_json(json_path)

    N = len(gamma_list) # num agents

    #define the coordinator that take into account for handling all the update of the admm

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=burn_in)

    # define the moreau envelope provider for the admm
    
    ## run MTGP with 1.3 iota and 0.98 ==== less burn in  with 3 points too see => Not converge
    mtgp = MTGPMoreauGradProvider(gamma_list, theta_list, rho=cfg.rho, zeta=cfg.zeta, 
                                min_points=burn_in, iota=iota, alpha=alpha, train_iter=train_iter, lr=lr, 
                                num_tasks=N, num_latents=Q, rank=rank, device="cpu",
                                # dtype=torch.float32)
                                dtype=torch.float64)

    start = time.time()

    # using mtgp as the moreau provider for the ADMM

    def moreau_provider(i, z_i, query_flag, k):
        return mtgp(i, z_i, query_flag, k) # call mtgp provider
    
    # hook the sharing admm with the mtgp provider
    out = coord.run(moreau_provider, verbose=True, log_every=1)

    elapsed = time.time() - start
    out["elapsed"] = elapsed
    print(" - - - - - - - - MTGP - ADMM  - - - - - - - - -")
    print("Set:", set_name)
    print("Running time:", elapsed)

    # getting all log from SharingADMM algorithm
    r_hist = out.get("r_hist", [])
    s_hist = out.get("s_hist", [])
    obj_hist = out.get("obj_hist", [])

    comm_per_iter = out.get("comm_per_iter", [])

    K = min(len(r_hist), len(s_hist), len(obj_hist), len(comm_per_iter))

    q_arr = np.array(comm_per_iter[:K], dtype=float) if K > 0 else np.array([0.0])
    # print(f"Comm per iter: mean={q_arr.mean():.3f}, min={q_arr.min():.0f}, max={q_arr.max():.0f}")
    
    print(f"Total comm (agent queries): {int(q_arr.sum())}")
    
    print(f"Final Objective", obj_hist[-1])

    return out

def main():

    # example dataset from data_generation/
    data_path = "data_generation/homogeneous.json"
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # test outputs go to test/ so they never overwrite saved results
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_output_path = os.path.join(output_dir, f"test_mtgp_{dataset_name}.jsonl")

    # mtgp setting:

    mtgp_iota = 1.0
    mtgp_alpha = 0.96
    mtgp_train_iter = 300
    mtgp_lr = 0.03
    mtgp_Q = 5
    mtgp_rank = 1
    burn_in = 5

    out = run_mtgp(data_path, burn_in=burn_in, iota=mtgp_iota, alpha=mtgp_alpha, train_iter=mtgp_train_iter, lr=mtgp_lr, Q=mtgp_Q, rank=mtgp_rank)

    # proceed data to jsonl
    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(data_path))
    
if __name__ == "__main__":
    main()