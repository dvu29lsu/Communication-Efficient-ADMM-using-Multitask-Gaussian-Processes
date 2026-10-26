import numpy as np
import os
import time
import csv

# from run_mtgp_admm import SET
from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.csv_writer import write_summary_row, extract_metrics
from utils.json_processing import process_output_to_jsonl
from agents_loader.agent_load import load_data_from_mat
from query_method.sogp.sogp_provider import SOGPMoreauProvider


data_path = os.path.join("data", "10 Agents", "p5")

cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-4, eps_rel=1e-3)

# modleing the GP with SOGP
def run_sogp(mat_path: str, burn_in: int = 5, iota: float = 0.5, alpha: float = 0.8, train_iter: int = 100, lr: float = 0.01):
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
        alpha=alpha,
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
    SETS = os.listdir(data_path)
    for set_name in SETS:
        mat_file = os.path.join(data_path, set_name)
        
        ## setting gp
        gp_iota = 0.6
        gp_alpha = 0.8
        gp_train_iter = 100
        gp_lr = 0.01
        burn_in = 5

        # run gp-admm with the config
        out = run_sogp(mat_file, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr)

        # jsonl output data path
        setting_gp = f"iota_{gp_iota}_alpha_{gp_alpha}_train_{gp_train_iter}_lr_{gp_lr}"
        
        # base dir 
        base_dir = os.path.join("results", "run_sogp_fullset_noise_free", setting_gp)
        os.makedirs(base_dir, exist_ok=True)
        
        # result path for jsonl and csv
        jsonl_output_path = os.path.join(base_dir, f"sogp_admm_results_summary_{set_name[:-4]}.jsonl")
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

        # print("Summary CSV path:", csv_path)
        write_summary_row(csv_path, summary_row)

        print("Finish:", os.path.basename(mat_file))
        # break # just for test

if __name__ == "__main__":
    main()