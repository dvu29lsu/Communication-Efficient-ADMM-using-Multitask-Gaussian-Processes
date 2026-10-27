import numpy as np
import os
import time
import csv
import torch

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.json_processing import process_output_to_jsonl
from agents_loader.agent_load import load_data_from_json, load_direction_matrix
from query_method.sogp.sogp_provider import SOGPMoreauProvider
from utils.set_seed import set_seed

from admm.moreau_oracle import build_local_models

set_seed(123)

# torch.set_default_dtype(torch.float) ##float32

# try the float64: float32 rounds the small predictive variances used by the query rule
torch.set_default_dtype(torch.float64)

# ADMM config
cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-6, eps_rel=1e-3)

# modleing the GP with SOGP
def run_sogp(json_path: str, burn_in: int, iota: float = 0.5, alpha: float = 0.8, train_iter: int = 100, lr: float = 0.01, local_model: str = "quadratic", lambda4: float = 0.5, direct_matrix=None):
    set_name = os.path.basename(json_path) # dataset name for the printed summary

    # SOGP keeps one GP per agent, so the cluster labels are not used
    gamma_list, theta_list, _ = load_data_from_json(json_path)
    N = len(gamma_list)

    # build the local models
    local_models = build_local_models(gamma_list, theta_list, local_model, lambda4, A=direct_matrix)

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=burn_in, local_models=local_models)

    ## we can try with differetn config of the GP models -> need tunning
    sogp = SOGPMoreauProvider(
        gamma_list=gamma_list,
        theta_list=theta_list,
        rho=cfg.rho,
        zeta=cfg.zeta,
        min_points=burn_in,
        iota=iota,
        alpha=alpha,
        train_iters=train_iter,
        lr=lr,
        device="cpu",
        # dtype=torch.float32,
        dtype=torch.float64,
        eval_exact_error=False,
        local_model=local_model,
        lambda4=lambda4,
        direct_matrix=direct_matrix
    )

    start = time.time()

    # SharingADMM expects provider returns
    def moreau_provider(i, z_i, query_flag, k):
        return sogp(i, z_i, query_flag, k) # calling the SOGP for query

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
    
    print(f"Total comm (agent queries): {int(q_arr.sum())}")
    
    print(f"Final Objective", obj_hist[-1])

    return out

def main():
    # example dataset from data_generation/
    # data_path = "data_generation/overlapping_3_clusters.json"
    data_path = "data_generation/separated_2_clusters.json"
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # test outputs go to test/ so they never overwrite saved results
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_output_path = os.path.join(output_dir, f"test_sogp_{dataset_name}.jsonl")

    # sogp setting
    gp_iota = 0.8
    gp_alpha = 0.96
    gp_train_iter = 100
    gp_lr = 0.01

    #training points
    burn_in = 5

    # different function

    #=========== Quadratic or coordinate Poly 4 setting ============

    local_model = "poly4"
    local_model = "quadratic"

    lambda4 = 0.5

    out = run_sogp(data_path, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr, local_model=local_model, lambda4=lambda4)

    #=========== Directional Poly 4 setting ============
    # local_model = "dir_poly4"
    # lambda4 = 0.5

    # # # fixed orthogonal 5x5 directions, shared by all agents and methods (generate_direction_matrix.py)
    # A = load_direction_matrix("data_generation/directional_quartic_dim5.json")

    # out = run_sogp(data_path, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr, local_model=local_model, lambda4=lambda4, direct_matrix=A)

    # proceed data to jsonl
    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(data_path))

if __name__ == "__main__":
    main()