import numpy as np
import os
import time
import csv

# from run_mtgp_admm import SET
from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.json_processing import process_output_to_jsonl
from agents_loader.agent_load import load_data_from_mat
from query_method.sogp.sogp_provider import SOGPMoreauProvider

# set the path for the data and the output
SETS = ["Set1.mat"]
# data_path = r"C:\Users\longv\research\admm_mogp\data\10 Agents\p5"
data_path = os.path.join("data", "10 Agents", "p5")

# jsonl output data path
jsonl_output_path = os.path.join("results", "gp_grad_err_sogp_admm_results_summary.jsonl")

# try another config for the ADMM
# cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=500, eps_abs=1e-4, eps_rel=1e-3)

cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-4, eps_rel=1e-3)

# modleing the GP with SOGP
def run_sogp(mat_path: str, burn_in: int, iota: float = 0.5, alpha: float = 0.8, train_iter: int = 100, lr: float = 0.01):
    set_name = os.path.splitext(os.path.basename(mat_path))[0]
    gamma_list, theta_list = load_data_from_mat(mat_path)
    N = len(gamma_list)

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=burn_in)

    ## we can try with differetn config of the GP models -> need tunning
    sogp = SOGPMoreauProvider(
        gamma_list=gamma_list,
        theta_list=theta_list,
        rho=cfg.rho,
        zeta=cfg.zeta,
        min_points=burn_in,
        iota=iota,
        alpha=alpha,   # tune later
        train_iters=train_iter,
        lr=lr,
        device="cpu",
        eval_exact_error=False,
    )

    start = time.time()

    # SharingADMM expects provider returns
    def moreau_provider(i, z_i, query_flag, k):
        return sogp(i, z_i, query_flag, k) # calling the SOGP query

    # hook the moreau envelope provider with the return of the GP
    out = coord.run(moreau_provider, verbose=True, log_every=3)

    elapsed = time.time() - start
    out["elapsed"] = elapsed

    print("\n - - - - GP-ADMM - - - - - - ")
    print("Set:", set_name)
    print("Running time:", elapsed)

    # getting all log from SharingADMM algorithm
    r_hist = out.get("r_hist", [])
    s_hist = out.get("s_hist", [])
    obj_hist = out.get("obj_hist", [])

    comm_per_iter = out.get("comm_per_iter", [])

    # comm_cum = out.get("comm_cum", [])
    # comm_mask_per_iter = out.get("comm_mask_per_iter", [])
    # stats_per_iter = out.get("stats_per_iter", [])

    K = min(len(r_hist), len(s_hist), len(obj_hist), len(comm_per_iter))

    # Communication summary
    q_arr = np.array(comm_per_iter[:K], dtype=float) if K > 0 else np.array([0.0])

    # print(f"Comm per iter: mean={q_arr.mean():.3f}, min={q_arr.min():.0f}, max={q_arr.max():.0f}")
    
    print(f"Total comm (agent queries): {int(q_arr.sum())}")
    
    print(f"Final Objective", obj_hist[-1])

    return out

def main():
    mat_file = os.path.join(data_path, SETS[0])

    # sogp setting
    gp_iota = 0.6
    gp_alpha = 0.8
    gp_train_iter = 100
    gp_lr = 0.01

    #training points
    burn_in = 5

    out = run_sogp(mat_file, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr)

    # proceed data to jsonl
    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(mat_file))

if __name__ == "__main__":
    main()