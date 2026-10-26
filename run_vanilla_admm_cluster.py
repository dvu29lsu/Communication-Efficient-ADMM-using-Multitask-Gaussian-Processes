import numpy as np
import os
import time
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_jsonl
from admm.exact_quadsolver import solving_moreau_exact_update
from utils.json_processing import process_output_to_jsonl

from admm.moreau_oracle import build_moreau_oracles
from admm.local_models import QuadraticLocalModel, Poly4LocalModel
from admm.moreau_oracle import build_local_models

cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=500, eps_abs=1e-4, eps_rel=1e-3)

jsonl_output_dir = os.path.join("results", "directional_poly4_cluster")
os.makedirs(jsonl_output_dir, exist_ok=True)

# select the jsonl file for the cluster function
jsonl_output_path = os.path.join("results", "directional_poly4_cluster", "directional_poly4_c3_vanilla_admm_results_summary.jsonl")

def run_vanilla(jsonl_path, local_model="quadratic", lambda4=0.05, direct_matrix=None, solver_method="trust-constr", solver_tol=1e-6, solver_maxiter=500):
    set_name = "C3 - Cluster"

    gamma_list, theta_list, _ = load_data_from_jsonl(jsonl_path)
    N = len(gamma_list)

    # define the local model // fomr of the function
    local_models = build_local_models(gamma_list, theta_list, local_model, lambda4, A=direct_matrix)

    # define the coordinator that take into account for handling all the update of the admm
    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=0, local_models=local_models)

    # non-local
    # coord = Coordinator(gamma_list, theta_list, cfg, burn_in=0)

    start = time.time()

    # define the local function and solver
    oracles = build_moreau_oracles(
    gamma_list=gamma_list,
    theta_list=theta_list,
    rho=cfg.rho,
    local_model=local_model,
    lambda4=lambda4,
    A=direct_matrix,
    solver_method=solver_method,
    solver_tol=solver_tol,
    solver_maxiter=solver_maxiter,
    )

    # Always communicate
# ======================= April, 8th: verify algo correct, solving the Moreau envelope with the exact solver for the local function =======================

    # non-local
    # coord = Coordinator(gamma_list, theta_list, cfg, burn_in=0)

    # start = time.time()

    # def moreau_provider(i, z_i, query_flag, k):
    #     moreau_grad = solving_moreau_exact_update(z_i, gamma_list[i], theta_list[i], cfg.rho)
    #     # vanilla always queries
    #     return moreau_grad, True, {}

# ================================================================================================================================================

    def moreau_provider(i, z_i, query_flag, k):
        _, g = oracles[i].moreau_value_and_grad(z_i)
        return g, True, {}, 0.0

    out = coord.run(moreau_provider, verbose=True, log_every=3)

    elapsed = time.time() - start
    print("\n=== VANILLA-ADMM ===")
    print("Set:", set_name)
    print("Running time:", elapsed)

    r_hist = out.get("r_hist", [])
    s_hist = out.get("s_hist", [])
    obj_hist = out.get("obj_hist", [])
    comm_per_iter = out.get("comm_per_iter", [])
    # comm_cum = out.get("comm_cum", [])

    K = min(len(r_hist), len(s_hist), len(obj_hist), len(comm_per_iter))

    q_arr = np.array(comm_per_iter[:K], dtype=float) if K > 0 else np.array([0.0])
    print(f"Comm per iter: mean={q_arr.mean():.3f}, min={q_arr.min():.0f}, max={q_arr.max():.0f}")
    print(f"Total comm (agent queries): {int(q_arr.sum())}")
    print(f"Final objective:", obj_hist[-1])

    return out


def main():
    # set the path using the cluster function
    # data_path = "sand_box/set1_three_cluster.jsonl"
    data_path = "sand_box/set1_one_cluster.jsonl"

    solver_method = "trust-constr"
    solver_tol = 1e-4
    solver_maxiter = 300

## Setting 1: Poly 4
    local_model = "quadratic" # "quadratic" or "poly4"
    lambda4 = 0.5 # affect of the 4th order term in local function with poly-4th order
    out = run_vanilla(data_path, local_model=local_model, lambda4=lambda4, solver_method=solver_method, solver_tol=solver_tol, solver_maxiter=solver_maxiter)


## Setting 2: Directional Poly 4
    # local_model = "dir_poly4"
    # lambda4 = 0.5

    # A = np.array([
    #     [1/np.sqrt(2), 1/np.sqrt(2), 1.5, 0.5, 0.3],
    #     [0.5, 1.0, 1/np.sqrt(3), 1/np.sqrt(3), 0.5],
    #     [1/np.sqrt(3), 0, 1/np.sqrt(2), 0.5, 1.0],
    # ], dtype=float)

    # out = run_vanilla(data_path, local_model=local_model, lambda4=lambda4, direct_matrix=A, solver_method=solver_method, solver_tol=solver_tol, solver_maxiter=solver_maxiter)

    process_output_to_jsonl(out, jsonl_output_path)
    print("Finish:", os.path.basename(data_path))


if __name__ == "__main__":
    main()
