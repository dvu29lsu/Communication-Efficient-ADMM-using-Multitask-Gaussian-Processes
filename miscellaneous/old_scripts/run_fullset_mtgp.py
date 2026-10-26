import numpy as np
import torch
import gpytorch
import os
import time
import json
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.csv_writer import write_summary_row, extract_metrics
from agents_loader.agent_load import load_data_from_mat
from utils.json_processing import process_output_to_jsonl
from query_method.mogp.mtgp_provider import MTGPMoreauGradProvider

# data path
data_path = os.path.join("data", "10 Agents", "p5")

# config for the ADMM
cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-4, eps_rel=1e-3, burn_in=5)

# run mtgp admm
def run_mtgp(mat_path: str, burn_in: int, iota: float = 1.3, alpha: float = 0.96, train_iter: int = 300, lr: float = 0.03, Q: int = 5, rank: int = 1):
    set_name = os.path.splitext(os.path.basename(mat_path))[0]
    gamma_list, theta_list = load_data_from_mat(mat_path)
    # finishig loading the data for the function

    N = len(gamma_list) # num agents

    #define the coordinator that take into account for handling all the update of the admm

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=burn_in)

    # define the moreau envelope provider for the admm
    
    ## run MTGP with 1.3 iota and 0.98 ==== less burn in  with 3 points too see => Not converge

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
    SETS = os.listdir(data_path)

    for set_name in SETS:
        mat_file = os.path.join(data_path, set_name)
        
        ## setting mtgp
        mtgp_iota = 1.5
        mtgp_alpha = 0.96
        mtgp_train_iter = 300
        mtgp_Q = 5
        mtgp_rank = 1
        mtgp_lr = 0.03
        burn_in = 5

        # run mtgp-admm with the config
        out = run_mtgp(mat_file, burn_in=burn_in, iota=mtgp_iota, alpha=mtgp_alpha, train_iter=mtgp_train_iter, lr=mtgp_lr, Q=mtgp_Q, rank=mtgp_rank)

        # jsonl output data path
        setting_gp = f"iota_{mtgp_iota}_alpha_{mtgp_alpha}_train_{mtgp_train_iter}_lr_{mtgp_lr}_Q_{mtgp_Q}_rank_{mtgp_rank}"
        
        # base dir 
        base_dir = os.path.join("results", "run_mtgp_fullset_noise_free", setting_gp)
        os.makedirs(base_dir, exist_ok=True)
        
        # result path for jsonl and csv
        jsonl_output_path = os.path.join(base_dir, f"mtgp_admm_results_summary_{set_name[:-4]}.jsonl")
        csv_path = os.path.join(base_dir, "summary.csv")

        # print("Output path:", jsonl_output_path)

        # proceed data to jsonl
        process_output_to_jsonl(out, jsonl_output_path)

        # write summary to csv
        n_iter, n_comm, final_obj, converged = extract_metrics(out)

        summary_row = {
            "set": set_name,
            "n_iter": n_iter,
            "n_comm": n_comm,
            "objective": final_obj,
            "time": out.get("elapsed", 0),
            "converged": converged
        }

        write_summary_row(csv_path, summary_row)

        print("Finish:", os.path.basename(mat_file))
    
if __name__ == "__main__":
    main()