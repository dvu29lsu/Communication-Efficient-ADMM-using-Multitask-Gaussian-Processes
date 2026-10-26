import os
import time
import csv
import json
import numpy as np

from ADMM_MTGP.miscellaneous.old_scripts.run_mtgp_admm import SETS
from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_mat, load_data_from_jsonl
from utils.json_processing import process_output_to_jsonl
from query_method.mogp.mtgp_provider import MTGPMoreauGradProvider


cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-4, eps_rel=1e-3, burn_in=5)

DEVICE = "cpu"

# MTGP fixed hyperparams
NUM_LATENTS = 5
RANK = 1
TRAIN_ITER = 300
LR = 0.03


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def setting_str(iota: float, alpha: float, q: int, rank: int, train_iter: int, lr: float) -> str:
    return f"iota_{iota:.3g}_alpha_{alpha:.4f}_Q_{q}_rank_{rank}_train_{train_iter}_lr_{lr:.3g}"

def write_summary_row(csv_path: str, row: dict):
    # required columns per your request
    fieldnames = ["setting", "set", "n_iter", "n_comm", "objective", "time", "converged"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

def extract_metrics(out: dict):
    r_hist = out.get("r_hist", [])
    s_hist = out.get("s_hist", [])
    obj_hist = out.get("obj_hist", [])
    comm_per_iter = out.get("comm_per_iter", [])
    converged = out.get("converged", False)
    
    K = min(len(r_hist), len(s_hist), len(obj_hist), len(comm_per_iter))

    q_arr = np.array(comm_per_iter[:K], dtype=float) if K > 0 else np.array([0.0])
    n_comm = int(q_arr.sum())
    final_obj = float(obj_hist[-1])
    n_iter = int(K)
    return n_iter, n_comm, final_obj, converged


# run_mtgp_admm 

def run_one_mtgp(data_path: str, set_name: str, burn_in: int, iota: float, alpha: float,
                train_iter: int = TRAIN_ITER, lr: float = LR, num_latents: int = NUM_LATENTS, 
                rank: int = RANK, device: str = DEVICE, verbose: bool = True):

    gamma_list,theta_list,_ = load_data_from_jsonl(data_path)

    N = len(gamma_list)

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=burn_in)

    mtgp = MTGPMoreauGradProvider(
        gamma_list, theta_list,
        rho=cfg.rho, zeta=cfg.zeta,
        min_points=burn_in,
        iota=iota,
        alpha=alpha,
        train_iter=train_iter,
        lr=lr,
        num_tasks=N,
        num_latents=num_latents,
        rank=rank,
        device=device,
    )

    start = time.time()

    def moreau_provider(i, z_i, query_flag, k):
        return mtgp(i, z_i, query_flag, k)

    out = coord.run(moreau_provider, verbose=verbose, log_every=1)

    elapsed = time.time() - start
    n_iter, n_comm, final_obj, converged = extract_metrics(out)

    if verbose:
        print("-" * 60)
        print(f"Set: {set_name}")
        print(f"iota={iota}, alpha={alpha}, Q={num_latents}, rank={rank}, train_iter={train_iter}, lr={lr}")
        print(f"n_iter={n_iter}, total_comm={n_comm}, final_obj={final_obj}, time={elapsed:.2f}s")

    return out, {
        "set": set_name,
        "n_iter": n_iter,
        "n_comm": n_comm,
        "objective": final_obj,
        "time": elapsed,
        "converged": converged,
    }


## sweeping different iota and alpha for MTGP-ADMM

def sweep_iota(data_path: str, set_name: str, burn_in: int, iota_list, alpha_fixed: float,
               out_root: str = os.path.join("results", "cluster", "mtgp_sweep_iota"),
               save_verbose: bool = True):

    ensure_dir(out_root)
    detail_dir = os.path.join(out_root, set_name, "jsonl")
    ensure_dir(detail_dir)

    summary_csv = os.path.join(out_root, set_name, "summary.csv")

    for iota in iota_list:
        setting = setting_str(iota, alpha_fixed, NUM_LATENTS, RANK, TRAIN_ITER, LR)

        detail_jsonl = os.path.join(detail_dir, f"{setting}.jsonl")

        out, metrics = run_one_mtgp(
            data_path=data_path,
            set_name=set_name,
            burn_in=burn_in,
            iota=iota,
            alpha=alpha_fixed,
            train_iter=TRAIN_ITER,
            lr=LR,
            num_latents=NUM_LATENTS,
            rank=RANK,
            device=DEVICE,
            verbose=save_verbose,
        )

        # Save full iteration logs for THIS setting
        process_output_to_jsonl(out, detail_jsonl)

        # Write one row summary
        row = {
            "setting": setting,
            "set": metrics["set"],
            "n_iter": metrics["n_iter"],
            "n_comm": metrics["n_comm"],
            "objective": metrics["objective"],
            "time": metrics["time"],
            "converged": metrics["converged"],
        }
        write_summary_row(summary_csv, row)

        print(f"Saved detail as {detail_jsonl}")
        print(f"Saved summary to {summary_csv}")


def sweep_alpha(data_path: str, set_name: str, burn_in: int, alpha_list, iota_fixed: float,
                out_root: str = os.path.join("results", "cluster", "mtgp_sweep_alpha"),
                save_verbose: bool = True):

    ensure_dir(out_root)
    detail_dir = os.path.join(out_root, set_name, "jsonl")
    ensure_dir(detail_dir)

    summary_csv = os.path.join(out_root, set_name, "summary.csv")

    for alpha in alpha_list:
        setting = setting_str(iota_fixed, alpha, NUM_LATENTS, RANK, TRAIN_ITER, LR)

        detail_jsonl = os.path.join(detail_dir, f"{setting}.jsonl")

        out, metrics = run_one_mtgp(
            data_path=data_path,
            set_name=set_name,
            burn_in=burn_in,
            iota=iota_fixed,
            alpha=alpha,
            train_iter=TRAIN_ITER,
            lr=LR,
            num_latents=NUM_LATENTS,
            rank=RANK,
            device=DEVICE,
            verbose=save_verbose,
        )

        process_output_to_jsonl(out, detail_jsonl)

        row = {
            "setting": setting,
            "set": metrics["set"],
            "n_iter": metrics["n_iter"],
            "n_comm": metrics["n_comm"],
            "objective": metrics["objective"],
            "time": metrics["time"],
            "converged": metrics["converged"],
        }
        write_summary_row(summary_csv, row)

        print(f"Saved detail as {detail_jsonl}")
        print(f"Saved summary to {summary_csv}")

# main

def main():
    data_path = ".\\sand_box\\set1_three_cluster.jsonl"
    set_name = "set1_three_cluster"

    burn_in = cfg.burn_in

    # iota sweep
    # iota_list = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    # alpha_fixed = 0.98
    # sweep_iota(data_path, set_name, burn_in, iota_list, alpha_fixed)

    # pick a iota in a range, then set the iota -> decay rate alpha

    # alpha sweep
    iota_fixed = 1.0
    alpha_list = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    sweep_alpha(data_path, set_name, burn_in, alpha_list, iota_fixed)

if __name__ == "__main__":
    main()