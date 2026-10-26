import numpy as np
import torch
import gpytorch
import os
import time
import json
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_jsonl
from utils.json_processing import process_output_to_jsonl
from utils.set_seed import set_seed

from query_method.mogp.cluster_mtgp_provider import ClusterMTGPProvider
from admm.local_models import QuadraticLocalModel, Poly4LocalModel

from admm.moreau_oracle import build_local_models

set_seed(100)
torch.set_default_dtype(torch.float32)

# jsonl output data path
# jsonl_output_path = os.path.join("results", "cluster_specific_mtgp", "increase_stop_latent_3_c1_mtgp_admm_results_summary.jsonl")

# # config for the ADMM
# cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-4, eps_rel=1e-3, burn_in=5)

# run mtgp admm cluster
def compute_comm_per_cluster(comm_mask_per_iter, labels):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    cluster_ids = sorted(np.unique(labels).tolist())

    out = {}
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        total = 0
        per_iter = []
        for mask in comm_mask_per_iter:
            mask = np.asarray(mask, dtype=bool)
            q = int(mask[idx].sum())
            total += q
            per_iter.append(q)

        out[int(c)] = {
            "total": int(total),
            "per_iter": per_iter,
            "agents": idx.tolist(),
        }
    return out

def run_mtgp_clustered(
    jsonl_path: str,
    admm_cfg: ADMMConfig,
    burn_in: int,
    iota: float = 1.0,
    alpha: float = 0.96,
    train_iter: int = 300,
    lr: float = 0.03,
    Q: int = 5,
    rank: int = 1,
    adapt_latents_to_cluster: bool = False,
    specific_task_noise: bool = False,
    local_model: str = "quadratic",
    lambda4: float = 0.05,
    direct_matrix=None,
    solver_method: str = "trust-constr",
    solver_tol: float = 1e-4,
    solver_maxiter: int = 300
):
    gamma_list, theta_list, labels = load_data_from_jsonl(jsonl_path)
    labels = np.asarray(labels[0], dtype=int).reshape(-1)

    set_name = os.path.basename(jsonl_path)
    N = len(gamma_list)

    # build the function form that need for compute the obj in the main Sharing ADMM loop
    # local model as the form of the functios
    admm_local_models = build_local_models(gamma_list, theta_list, local_model=local_model, lambda4=lambda4, A=direct_matrix)

    # coordinator for the ADMM algorithm, feed the local_models as local function form -> feed to admm for compute the objective given the value x 
    coord = Coordinator(gamma_list, theta_list, admm_cfg, burn_in=burn_in, local_models=admm_local_models)

    mtgp = ClusterMTGPProvider(
        gamma_list=gamma_list,
        theta_list=theta_list,
        labels=labels,
        rho=admm_cfg.rho,
        zeta=admm_cfg.zeta,
        min_points=burn_in,
        iota=iota,
        alpha=alpha,
        train_iter=train_iter,
        lr=lr,
        num_latents=Q,
        rank=rank,
        device="cpu",
        dtype=torch.float32,
        adapt_latents_to_cluster=adapt_latents_to_cluster,
        noise_specific_task=specific_task_noise,
        local_model=local_model,
        lambda4=lambda4,
        direct_matrix=direct_matrix,
        solver_method=solver_method,
        solver_tol=solver_tol,
        solver_maxiter=solver_maxiter
    )

    start = time.time()

    def moreau_provider(i, z_i, query_flag, k):
        return mtgp(i, z_i, query_flag, k)

    out = coord.run(moreau_provider, verbose=True, log_every=1)

    elapsed = time.time() - start
    out["elapsed"] = elapsed
    out["set_name"] = set_name
    out["labels"] = labels.tolist()
    out["cluster_summary"] = mtgp.get_cluster_summary()
    out["comm_per_cluster"] = compute_comm_per_cluster(out["comm_mask_per_iter"], labels)

    print("Cluster label:", labels)

    print("- - - - - - - - Clustered MTGP - ADMM - - - - - - - -")
    print("Set:", set_name)
    print("Running time:", elapsed)
    print("Total comm (agent queries):", int(np.sum(out["comm_per_iter"])))

    cluster_ids = sorted(np.unique(labels).tolist())
    for c in cluster_ids:
        print(f"Total comm in cluster {c + 1}:", out["comm_per_cluster"][c]["total"])
        print(f"Agents in cluster {c + 1}:", out["comm_per_cluster"][c]["agents"])

    if len(out.get("obj_hist", [])) > 0:
        print("Final Objective:", out["obj_hist"][-1])
    
    return out

def main():
    # data_path = "./sand_box/set1_three_cluster.jsonl"
    # data_path = "./sand_box/set1_two_cluster.jsonl"
    data_path = "./sand_box/set1_one_cluster.jsonl"

    # jsonl_output_dir = os.path.join("results", "compare_poly4_cluster")

    jsonl_output_dir = os.path.join("results", "compare_directional_poly4_cluster")

    os.makedirs(jsonl_output_dir, exist_ok=True)

    mtgp_iota = 1.0 # 1 clust: 1.0, 2 clust: 1.2, 3 clust: 1.0
    mtgp_alpha = 0.96
    mtgp_train_iter = 300
    mtgp_lr = 0.03
    mtgp_Q = 3
    mtgp_rank = 1
    burn_in = 5
    specific_task_noise = False
    adapt_latents_to_cluster = False ## when one cluster: False, when multiple cluster: True -> taking ceil: num agent / 2 

    solver_method = "trust-constr"
    solver_tol = 1e-4
    solver_maxiter = 300

    admm_cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=61, eps_abs=1e-6, eps_rel=1e-3, burn_in=5)

## Setting 1: Poly 4
    # # local function form and solver options for the prox computation in MTGP provider
    # local_model = "poly4" # "quadratic" or "poly4"
    # lambda4 = 0.5

    # out = run_mtgp_clustered(
    #     jsonl_path = data_path,
    #     admm_cfg = admm_cfg,
    #     burn_in = burn_in,
    #     iota = mtgp_iota,
    #     alpha = mtgp_alpha,
    #     train_iter = mtgp_train_iter,
    #     lr = mtgp_lr,
    #     Q = mtgp_Q,
    #     rank = mtgp_rank,
    #     adapt_latents_to_cluster = adapt_latents_to_cluster,
    #     specific_task_noise=specific_task_noise,
    #     local_model=local_model,
    #     lambda4=lambda4,
    #     solver_method=solver_method,
    #     solver_tol=solver_tol,
    #     solver_maxiter=solver_maxiter   
    # )

## Setting 2: Directional Poly 4
    local_model = "dir_poly4"
    lambda4 = 0.5
    A = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2), 1.5, 0.5, 0.3],
        [0.5, 1.0, 1/np.sqrt(3), 1/np.sqrt(3), 0.5],
        [1/np.sqrt(3), 0, 1/np.sqrt(2), 0.5, 1.0],
    ], dtype=float)

    out = run_mtgp_clustered(
        jsonl_path = data_path,
        admm_cfg = admm_cfg,
        burn_in = burn_in,
        iota = mtgp_iota,
        alpha = mtgp_alpha,
        train_iter = mtgp_train_iter,
        lr = mtgp_lr,
        Q = mtgp_Q,
        rank = mtgp_rank,
        adapt_latents_to_cluster = adapt_latents_to_cluster,
        specific_task_noise=specific_task_noise,
        local_model=local_model,
        lambda4=lambda4,
        direct_matrix=A,
        solver_method=solver_method,
        solver_tol=solver_tol,
        solver_maxiter=solver_maxiter   
    )

    output_name = f"mtgp_clustered_specific_{mtgp_iota}_{mtgp_alpha}_{os.path.splitext(os.path.basename(data_path))[0]}.jsonl"

    # jsonl_output_path = os.path.join("results", "compare_poly4_cluster", output_name)

    jsonl_output_path = os.path.join("results", "compare_directional_poly4_cluster", output_name)

    
    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(data_path))

if __name__ == "__main__":
    main()