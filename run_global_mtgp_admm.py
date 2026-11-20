import os

# CPU budget for this run: torch's default (all 24 cores) is ~20x slower here
# for our small GP matrices; keep (runs in parallel) x N_THREADS <= 24.
# The env vars must be set before numpy/torch are imported (BLAS reads them once at load).
N_THREADS = 24
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)

import numpy as np
import torch
import time

torch.set_num_threads(N_THREADS)

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_json, load_direction_matrix
from utils.json_processing import process_output_to_jsonl
from utils.set_seed import set_seed

from query_method.mogp.mtgp_provider import MTGPMoreauGradProvider

from admm.moreau_oracle import build_local_models

set_seed(123)
# torch.set_default_dtype(torch.float32)

torch.set_default_dtype(torch.float64)

def run_mtgp_global(
    json_path: str,
    admm_cfg: ADMMConfig,
    burn_in: int,
    iota: float = 0.5,
    alpha: float = 0.6,
    train_iter: int = 300,
    refit_iter: int = 100,
    lr: float = 0.03,
    Q: int = 5,
    rank: int = 1,
    specific_task_noise: bool = False,
    local_model: str = "quadratic",
    lambda4: float = 0.5,
    direct_matrix=None
):
    # global MTGP shares one model across all agents => cluster labels are not used
    gamma_list, theta_list, _ = load_data_from_json(json_path)

    set_name = os.path.basename(json_path)
    N = len(gamma_list)

    # local function form, so the objective in the Sharing ADMM loop uses the same f_i as the oracles
    admm_local_models = build_local_models(gamma_list, theta_list, local_model=local_model, lambda4=lambda4, A=direct_matrix)

    coord = Coordinator(gamma_list, theta_list, admm_cfg, burn_in=burn_in, local_models=admm_local_models)

    # one MTGP over all N agents
    mtgp = MTGPMoreauGradProvider(
        gamma_list=gamma_list,
        theta_list=theta_list,
        rho=admm_cfg.rho,
        zeta=admm_cfg.zeta,
        min_points=burn_in,
        iota=iota,
        alpha=alpha,
        train_iter=train_iter,
        refit_iter=refit_iter,
        lr=lr,
        num_tasks=N,
        num_latents=Q,
        rank=rank,
        device="cpu",
        # dtype=torch.float32,
        dtype=torch.float64,
        specific_task_noise=specific_task_noise,
        local_model=local_model,
        lambda4=lambda4,
        direct_matrix=direct_matrix
    )

    start = time.time()

    def moreau_provider(i, z_i, query_flag, k):
        return mtgp(i, z_i, query_flag, k)

    out = coord.run(moreau_provider, verbose=True, log_every=1)

    elapsed = time.time() - start
    out["elapsed"] = elapsed
    out["set_name"] = set_name

    print("- - - - - - - - Global MTGP - ADMM - - - - - - - -")
    print("Set:", set_name)
    print("Local model:", local_model)
    print("Running time:", elapsed)
    print("Total comm (agent queries):", int(np.sum(out["comm_per_iter"])))

    if len(out.get("obj_hist", [])) > 0:
        print("Final Objective:", out["obj_hist"][-1])

    return out

def main():
    # example dataset from data_generation/
    # data_path = "data_generation/homogeneous.json"
    # data_path = "data_generation/heterogeneous.json"
    # data_path = "data_generation/separated_2_clusters.json"
    # data_path = "data_generation/separated_3_clusters.json"
    # data_path = "data_generation/overlapping_2_clusters.json"
    # data_path = "data_generation/overlapping_3_clusters.json"

    # test run for 20 agents cases
    data_path = "data_generation/20 Agents/overlapping_2_clusters.json"

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # test outputs go to test/ so they never overwrite saved results
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_output_path = os.path.join(output_dir, f"global_mtgp_test_20_agents_run_{dataset_name}.jsonl")

    mtgp_iota = 1.3
    mtgp_alpha = 0.98
    mtgp_train_iter = 300 # Adam steps for the first fit
    mtgp_refit_iter = 100 # Adam steps for each later refit
    mtgp_lr = 0.03

    # try with 5 equations first
    mtgp_Q = 10

    #try with Q = 10
    # mtgp_Q = 10

    mtgp_rank = 1
    burn_in = 5
    specific_task_noise = False

    admm_cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-6, eps_rel=1e-3, burn_in=5)

## Setting 1: Quadratic or coordinate Poly 4
    local_model = "quadratic" # "quadratic" or "poly4"
    lambda4 = 0.5

    out = run_mtgp_global(
        json_path = data_path,
        admm_cfg = admm_cfg,
        burn_in = burn_in,
        iota = mtgp_iota,
        alpha = mtgp_alpha,
        train_iter = mtgp_train_iter,
        refit_iter = mtgp_refit_iter,
        lr = mtgp_lr,
        Q = mtgp_Q,
        rank = mtgp_rank,
        specific_task_noise = specific_task_noise,
        local_model = local_model,
        lambda4 = lambda4
    )

## Setting 2: Directional Poly 4
    # local_model = "dir_poly4"
    # lambda4 = 0.5

    # # fixed orthogonal 5x5 directions, shared by all agents and methods (generate_direction_matrix.py)
    # A = load_direction_matrix("data_generation/directional_quartic_dim5.json")

    # out = run_mtgp_global(
    #     json_path = data_path,
    #     admm_cfg = admm_cfg,
    #     burn_in = burn_in,
    #     iota = mtgp_iota,
    #     alpha = mtgp_alpha,
    #     train_iter = mtgp_train_iter,
    #     refit_iter = mtgp_refit_iter,
    #     lr = mtgp_lr,
    #     Q = mtgp_Q,
    #     rank = mtgp_rank,
    #     specific_task_noise = specific_task_noise,
    #     local_model = local_model,
    #     lambda4 = lambda4,
    #     direct_matrix = A
    # )

    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(data_path))

if __name__ == "__main__":
    main()
