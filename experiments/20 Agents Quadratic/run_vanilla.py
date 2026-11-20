
import contextlib
import csv
import io
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.json_processing import process_output_to_jsonl
from agents_loader.agent_load import load_data_from_json
from admm.moreau_oracle import build_local_models, build_moreau_oracles

# ---------------- experiment setup ----------------
FUNCTION = "quadratic"  # local_model passed to build_local_models / build_moreau_oracles
LAMBDA4 = 0.5 # weight of the quartic term (unused for quadratic, kept for the poly4 copy)

DATASETS = [
    "homogeneous",
    "heterogeneous",
    "separated_2_clusters",
    "separated_3_clusters",
    "overlapping_2_clusters",
    "overlapping_3_clusters",
]

DATA_DIR = "data_generation/20 Agents"

# setting ADMM
ADMM_CFG = ADMMConfig(zeta=1.0, rho=10, max_iters=300, eps_abs=1e-6, eps_rel=1e-3)

# OUTPUT_DIR = os.path.join("results", FUNCTION, "vanilla")

OUTPUT_DIR = os.path.join("results", "20 Agents Quadratic", "vanilla")

def run_vanilla(json_path, admm_cfg, local_model, lambda4, direct_matrix=None):
    # vanilla ADMM does not use the cluster labels
    gamma_list, theta_list, _ = load_data_from_json(json_path)

    # local function form, used for the objective in the Sharing ADMM loop
    local_models = build_local_models(gamma_list, theta_list, local_model=local_model, lambda4=lambda4, A=direct_matrix)
    coord = Coordinator(gamma_list, theta_list, admm_cfg, burn_in=0, local_models=local_models)

    # exact Moreau oracles: every agent queries at every iteration
    oracles = build_moreau_oracles(gamma_list, theta_list, rho=admm_cfg.rho, local_model=local_model, lambda4=lambda4, A=direct_matrix)

    def moreau_provider(i, z_i, query_flag, k):
        _, g = oracles[i].moreau_value_and_grad(z_i)
        return g, True, {}, 0.0

    start = time.time()

    # silence the per-solve prints of the oracle; one summary line per dataset is printed in main()
    with contextlib.redirect_stdout(io.StringIO()):
        out = coord.run(moreau_provider, verbose=False)

    out["elapsed"] = time.time() - start
    return out


def summary_row(dataset, out, admm_cfg):
    # one flat row per run; n_iter counts logged iterations (the solver's "iters" is the 0-based k)
    return {
        "dataset": dataset,
        "function": FUNCTION,
        "n_iter": len(out["obj_hist"]),
        "converged": bool(out["converged"]),
        "total_queries": int(out["comm_total"]),
        "final_obj": float(out["obj_hist"][-1]),
        "final_r": float(out["r"]),
        "final_s": float(out["s"]),
        "elapsed_s": round(float(out["elapsed"]), 3),
        "rho": admm_cfg.rho,
        "zeta": admm_cfg.zeta,
        "eps_abs": admm_cfg.eps_abs,
        "eps_rel": admm_cfg.eps_rel,
        "max_iters": admm_cfg.max_iters,
    }


def main():
    traj_dir = os.path.join(OUTPUT_DIR, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        json_path = os.path.join(DATA_DIR, f"{dataset}.json")
        out = run_vanilla(json_path, ADMM_CFG, local_model=FUNCTION, lambda4=LAMBDA4)

    # per-iteration trajectory
        process_output_to_jsonl(out, os.path.join(traj_dir, f"{dataset}.jsonl"))

        row = summary_row(dataset, out, ADMM_CFG)
        rows.append(row)
        print(f"{dataset:24s} iters={row['n_iter']:4d}  queries={row['total_queries']:5d}  "
              f"obj={row['final_obj']:.8f}  converged={row['converged']}  time={row['elapsed_s']:.2f}s")

    # summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
