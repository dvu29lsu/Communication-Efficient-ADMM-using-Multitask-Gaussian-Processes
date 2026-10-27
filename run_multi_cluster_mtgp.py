import numpy as np
import torch
import gpytorch
import os
import time
import json
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_json, load_direction_matrix
from utils.json_processing import process_output_to_jsonl
from utils.set_seed import set_seed

from query_method.mogp.cluster_mtgp_provider import ClusterMTGPProvider
from admm.local_models import QuadraticLocalModel, Poly4LocalModel

from admm.moreau_oracle import build_local_models

set_seed(123)
# torch.set_default_dtype(torch.float32)

# try the float64
torch.set_default_dtype(torch.float64)


# count exact queries per cluster from the per-iteration communication masks
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
    json_path: str,
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
    direct_matrix=None
):
    # labels (N,) give the known cluster of each agent -> one MTGP per cluster
    gamma_list, theta_list, labels = load_data_from_json(json_path)

    set_name = os.path.basename(json_path)
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
        # dtype=torch.float32,
        dtype=torch.float64,
        adapt_latents_to_cluster=adapt_latents_to_cluster,
        noise_specific_task=specific_task_noise,
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
    # example dataset from data_generation/ (cluster-wise MTGP needs more than one cluster)
    # data_path = "data_generation/overlapping_3_clusters.json"
    data_path = "data_generation/separated_2_clusters.json"

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # test outputs go to test/ so they never overwrite saved results
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_output_path = os.path.join(output_dir, f"test_cluster_mtgp_{dataset_name}.jsonl")

    mtgp_iota = 1.0 # 1 clust: 1.0, 2 clust: 1.2, 3 clust: 1.0
    mtgp_alpha = 0.96
    mtgp_train_iter = 300
    mtgp_lr = 0.03
    mtgp_Q = 3
    mtgp_rank = 1
    burn_in = 5
    specific_task_noise = False
    adapt_latents_to_cluster = True ## when one cluster: False, when multiple cluster: True -> taking ceil: num agent / 2

    admm_cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=150, eps_abs=1e-6, eps_rel=1e-3, burn_in=5)

## Setting 1: Quadratic or coordinate Poly 4
    # # local function form for the prox computation in MTGP provider
    local_model = "quadratic" # "quadratic" or "poly4"
    lambda4 = 0.5

    out = run_mtgp_clustered(
        json_path = data_path,
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
        lambda4=lambda4
    )

## Setting 2: Directional Poly 4
    # local_model = "dir_poly4"
    # lambda4 = 0.5

    # # fixed orthogonal 5x5 directions, shared by all agents and methods (generate_direction_matrix.py)
    # A = load_direction_matrix("data_generation/directional_quartic_dim5.json")

    # out = run_mtgp_clustered(
    #     json_path = data_path,
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
    #     direct_matrix=A
    # )

    process_output_to_jsonl(out, jsonl_output_path)

    print("Finish:", os.path.basename(data_path))

if __name__ == "__main__":
    main()