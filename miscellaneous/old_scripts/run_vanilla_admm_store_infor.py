import numpy as np
import os
import time
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_mat
from admm.exact_quadsolver import solving_moreau_exact_update
from utils.json_processing import process_output_to_jsonl

# set the path
SETS = ["Set1.mat"]

# data_path = r"C:\Users\longv\research\admm_mogp\data\10 Agents\p5"

data_path = os.path.join("data", "10 Agents", "p5")

cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=1000, eps_abs=1e-4, eps_rel=1e-3)

jsonl_output_path = os.path.join("results", "vanilla_admm_results_summary.jsonl")

def run_vanilla(mat_path: str):
    set_name = os.path.splitext(os.path.basename(mat_path))[0]
    gamma_list, theta_list = load_data_from_mat(mat_path)
    N = len(gamma_list)

    coord = Coordinator(gamma_list, theta_list, cfg, burn_in=0)

    start = time.time()

    # Always communicate
    def moreau_provider(i, z_i, query_flag, k):
        moreau_grad = solving_moreau_exact_update(z_i, gamma_list[i], theta_list[i], cfg.rho)
        # vanilla always queries
        return moreau_grad, True, {}

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
    mat_files = [os.path.join(data_path, s) for s in SETS]
    for mat in mat_files:
        out = run_vanilla(mat)
        process_output_to_jsonl(out, jsonl_output_path)
        print("Finish:", os.path.basename(mat))


if __name__ == "__main__":
    main()
