import numpy as np
import torch
import gpytorch
import os
import time
import json
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_mat
from utils.json_processing import process_output_to_jsonl
from utils.set_seed import set_seed

# import the load method
from query_method.mogp.mtgp_provider import MTGPMoreauGradProvider

set_seed(42)
torch.set_default_dtype(torch.float32)
# define the selected set
SETS = ["Set1.mat"]

# data path
data_path = os.path.join("data", "10 Agents", "p5")

# output path
jsonl_output_path = os.path.join("results", "mtgp_100_noise_e4_iter_grad_err_admm_results_summary.jsonl")

# config for the ADMM
cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-4, eps_rel=1e-3, burn_in=5)

# run mtgp admm

def run_mtgp(mat_path: str, burn_in: int, iota: float = 1.5, alpha: float = 0.96, train_iter: int = 300, lr: float = 0.03, Q: int = 5, rank: int =1):
    set_name = os.path.splitext(os.path.basename(mat_path))[0]
    
    gamma_list, theta_list = load_data_from_mat(mat_path)
    # finishig loading the data for the function

    N = len(gamma_list) # num agents

    #define the coordinator that take into account for handling all the update of the admm

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=burn_in)

    # define the moreau envelope provider for the admm
    
    ## run MTGP with 1.5 iota and 0.96 ==== less burn in  with 3 points too see => Not converge


    mtgp = MTGPMoreauGradProvider(gamma_list, theta_list, rho=cfg.rho, zeta=cfg.zeta, 
                                min_points=burn_in, iota=iota, alpha=alpha, train_iter=train_iter, lr=lr, 
                                num_tasks=N, num_latents=Q, rank=rank, device="cpu")

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
    # mat_files = [os.path.join(data_path, s) for s in SET]
    # for mat in mat_files:
    #     run_mtgp(mat, burn_in=5)

    # mtgp setting:
    mtgp_iota = 1.3
    mtgp_alpha = 0.96
    mtgp_train_iter = 300
    mtgp_lr = 0.03
    mtgp_Q = 5
    mtgp_rank = 1
    burn_in = 5

    mat_file = os.path.join(data_path, SETS[0])

    out = run_mtgp(mat_file, burn_in=burn_in, iota=mtgp_iota, alpha=mtgp_alpha, train_iter=mtgp_train_iter, lr=mtgp_lr, Q=mtgp_Q, rank=mtgp_rank)

    # print(out)

    # proceed data to jsonl
    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(mat_file))
    
if __name__ == "__main__":
    main()