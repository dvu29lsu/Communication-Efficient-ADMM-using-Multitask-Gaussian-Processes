import numpy as np
import os
import time
import csv
import torch

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.json_processing import process_output_to_jsonl
from agents_loader.agent_load import load_data_from_mat, load_data_from_jsonl
from query_method.sogp.sogp_provider import SOGPMoreauProvider
from utils.set_seed import set_seed

from admm.moreau_oracle import build_local_models

set_seed(123)

torch.set_default_dtype(torch.float) ##float32

# create output dir
# jsonl_output_dir = os.path.join("results", "test_run")
# os.makedirs(jsonl_output_dir, exist_ok=True)

# OUTPUT PATH
jsonl_output_path = os.path.join("results", "test_run_sogp_cluster.jsonl")

# ADMM config
cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-4, eps_rel=1e-3)

# modleing the GP with SOGP
def run_sogp(jsonl_path: str, burn_in: int, iota: float = 0.5, alpha: float = 0.8, train_iter: int = 100, lr: float = 0.01, local_model: str = "quadratic", lambda4: float = 0.5, solver_method: str = "trust-constr", solver_tol: float = 1e-4, solver_maxiter: int = 300, direct_matrix=None):
    set_name = "C3 - Cluster"
    gamma_list, theta_list, _ = load_data_from_jsonl(jsonl_path)
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
        eval_exact_error=False,
        local_model=local_model,
        lambda4=lambda4,
        solver_method=solver_method,
        solver_tol=solver_tol,
        solver_maxiter=solver_maxiter,
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
    data_path = "sand_box/set1_one_cluster.jsonl"

    # sogp setting
    gp_iota = 0.5
    gp_alpha = 0.8
    gp_train_iter = 100
    gp_lr = 0.01

    #training points
    burn_in = 5
    
    # proximal solver setting - using scipy
    solver_method = "trust-constr"
    solver_tol = 1e-4
    solver_maxiter = 300

    # different function

    #=========== Poly 4 setting ============

    # local_model = "poly4"
    local_model = "quadratic"

    lambda4 = 0.5

    out = run_sogp(data_path, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr, local_model=local_model, lambda4=lambda4, solver_method=solver_method, solver_tol=solver_tol, solver_maxiter=solver_maxiter)

    #=========== Directional Poly 4 setting ============
    # local_model = "dir_poly4"
    # lambda4 = 0.5
    
    # A = np.array([
    #     [1/np.sqrt(2), 1/np.sqrt(2), 1.5, 0.5, 0.3],
    #     [0.5, 1.0, 1/np.sqrt(3), 1/np.sqrt(3), 0.5],
    #     [1/np.sqrt(3), 0, 1/np.sqrt(2), 0.5, 1.0],
    # ], dtype=float)

    # out = run_sogp(data_path, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr, local_model=local_model, lambda4=lambda4, solver_method=solver_method, solver_tol=solver_tol, solver_maxiter=solver_maxiter, direct_matrix=A)

    # proceed data to jsonl
    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(data_path))

if __name__ == "__main__":
    main()