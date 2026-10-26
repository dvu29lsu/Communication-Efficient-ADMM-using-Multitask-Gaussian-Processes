import numpy as np
import os
import time
import csv

# from run_mtgp_admm import SET
from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.json_processing import process_output_to_jsonl
from agents_loader.agent_load import load_data_from_mat, load_data_from_jsonl
from query_method.sogp.sogp_provider import SOGPMoreauProvider


cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-4, eps_rel=1e-3, burn_in=5)

DEVICE = "cpu"

# fix hyperparams for the SOGP provider in the sweep
TRAIN_ITER = 100
LR = 0.01

# utils make dir
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# setting str for the sogp config
def setting_str(iota: float, alpha: float, train_iter: int, lr: float) -> str:
    return f"iota_{iota:.3g}_alpha_{alpha:.4f}_train_{train_iter}_lr_{lr:.3g}"

def write_summary_row(csv_path: str, row: dict):
    # required columns per your request
    fieldnames = ["setting", "set", "n_iter", "n_comm", "objective", "time"]
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

    # K = min(len(r_hist), len(s_hist), len(obj_hist), len(comm_per_iter))

    # if K <= 0:
    #     return 0, 0, float("nan")

    # q_arr = np.array(comm_per_iter[:K], dtype=float)
    
    K = min(len(r_hist), len(s_hist), len(obj_hist), len(comm_per_iter))

    q_arr = np.array(comm_per_iter[:K], dtype=float) if K > 0 else np.array([0.0])
    n_comm = int(q_arr.sum())
    final_obj = float(obj_hist[-1])
    n_iter = int(K)
    return n_iter, n_comm, final_obj, converged

def run_one_sogp(data_path: str, set_name: str, burn_in: int, iota: float, alpha: float,
                train_iter: int = TRAIN_ITER, lr: float = LR, device: str = DEVICE):
    
    gamma_list, theta_list,_ = load_data_from_jsonl(data_path)
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
        device=device,
        retrain_every=1,
    )

    start = time.time()

    # SharingADMM expects provider returns
    def moreau_provider(i, z_i, query_flag, k):
        return sogp(i, z_i, query_flag, k) # calling the SOGP query

    # hook the moreau envelope provider with the return of the GP
    out = coord.run(moreau_provider, verbose=True, log_every=1)

    elapsed = time.time() - start
    n_iter, n_comm, final_obj, converged = extract_metrics(out)


    print("-" * 30)
    print("======================= SOGP-ADMM =======================")
    print(f"Set: {set_name}")
    print(f"iota={iota}, alpha={alpha}, train_iter={train_iter}, lr={lr}")
    print(f"n_iter={n_iter}, total_comm={n_comm}, final_obj={final_obj}, time={elapsed:.2f}s")

    return out, {
        "set": set_name,
        "n_iter": n_iter,
        "n_comm": n_comm,
        "objective": final_obj,
        "time": elapsed,
        "converged": converged,
    }

def sweep_iota(data_path: str, burn_in: int, iota_list, alpha_fixed: float, set_name: str,
               out_root: str = os.path.join("results", "cluster", "sogp_sweep_iota")):

    ensure_dir(out_root)

    detail_dir = os.path.join(out_root, set_name, "jsonl")
    ensure_dir(detail_dir)

    summary_csv = os.path.join(out_root, set_name, "summary.csv")

    for iota in iota_list:
        setting = setting_str(iota, alpha_fixed, TRAIN_ITER, LR)

        detail_jsonl = os.path.join(detail_dir, f"{setting}.jsonl")

        out, metrics = run_one_sogp(
            data_path=data_path,
            set_name=set_name,
            burn_in=burn_in,
            iota=iota,
            alpha=alpha_fixed,
            train_iter=TRAIN_ITER,
            lr=LR,
            device=DEVICE
        )

        # Save full iteration logs for current setting
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


def sweep_alpha(data_path: str, burn_in: int, alpha_list, iota_fixed: float, set_name: str,
                out_root: str = os.path.join("results", "cluster", "sogp_sweep_alpha")):

    ensure_dir(out_root)
    detail_dir = os.path.join(out_root, set_name, "jsonl")
    ensure_dir(detail_dir)

    summary_csv = os.path.join(out_root, set_name, "summary.csv")

    for alpha in alpha_list:
        setting = setting_str(iota_fixed, alpha, TRAIN_ITER, LR)

        detail_jsonl = os.path.join(detail_dir, f"{setting}.jsonl")

        out, metrics = run_one_sogp(
            data_path=data_path,
            set_name=set_name,
            burn_in=burn_in,
            iota=iota_fixed,
            alpha=alpha,
            train_iter=TRAIN_ITER,
            lr=LR,
            device=DEVICE
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


def main():
    # mat_file = os.path.join(data_path, SETS[0])
    data_path = ".\\sand_box\\set1_two_cluster.jsonl"
    set_name = "set1_two_cluster"

    burn_in = cfg.burn_in

    # iota sweep
    # iota_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    # alpha_fixed = 0.98
    # sweep_iota(data_path, burn_in, iota_list, alpha_fixed, set_name=set_name)

    # alpha sweep
    alpha_list = [0.8, 0.85, 0.87, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    iota_fixed = 1.1
    sweep_alpha(data_path, burn_in, alpha_list, iota_fixed, set_name=set_name)

if __name__ == "__main__":
    main()