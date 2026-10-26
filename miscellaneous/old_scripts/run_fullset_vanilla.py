import numpy as np
import os, csv,time
import scipy as sp

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from agents_loader.agent_load import load_data_from_mat
from utils.csv_writer import write_summary_row, extract_metrics
from utils.json_processing import process_output_to_jsonl
from admm.exact_quadsolver import solving_moreau_exact_update

# set the data path

data_path = os.path.join("data", "10 Agents", "p5")

# ADMM algorithm config paramters
cfg = ADMMConfig(zeta=1.0, rho=10, max_iters=300, eps_abs=1e-4, eps_rel=1e-3)

# modeling ADMM without GP in full set
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
    SETS = os.listdir(data_path)

    for set_name in SETS:
        mat_file = os.path.join(data_path, set_name)

        out = run_vanilla(mat_file)

        base_dir = os.path.join("results", "run_vanilla_fullset", "summary")

        os.makedirs(base_dir, exist_ok=True)

        csv_path = os.path.join(base_dir, "summary.csv")
        
        n_iter, n_comm, final_obj, converged = extract_metrics(out)

        summary_row = {
            "set": set_name,
            "n_iter": n_iter,
            "n_comm": n_comm,
            "objective": final_obj,
            "time": out.get("elapsed", 0),
            "converged": converged
        }
        write_summary_row(csv_path, set_name, summary_row)
        print("Saved summary:", csv_path)
        print("Finish")

if __name__ == "__main__":
    main()
