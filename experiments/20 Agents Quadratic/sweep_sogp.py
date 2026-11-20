
import os
import sys

# repo root on the import path, so `from utils...` works however the script is launched
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# 2 threads per run: N_WORKERS runs share the CPU instead of fighting over it.
# Must be set before numpy/torch are imported (BLAS reads them once at load).
N_THREADS = 2
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)

import contextlib
import io
import json
import time

import torch

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.json_processing import save_run_output
from utils.set_seed import set_seed
from agents_loader.agent_load import load_data_from_json
from admm.moreau_oracle import build_local_models
from query_method.sogp.sogp_provider import SOGPMoreauProvider
from experiments.sweep_utils import (
    config_name, make_run_id, outcome_metrics, empty_outcome,
    save_error, short_error, run_sweep,
)

# ---------------- sweep setup ----------------
METHOD = "sogp"
FUNCTION = "quadratic"  # local_model for build_local_models and the provider's oracles
LAMBDA4 = 0.5  #weight of the quartic term (unused for quadratic, kept for the poly4 copy)

DATASETS = [
    "homogeneous",
    "heterogeneous",
    "separated_2_clusters",
    "separated_3_clusters",
    "overlapping_2_clusters",
    "overlapping_3_clusters",
]


# just for smoke test
# DATASETS = [
#     "homogeneous",
# ]
# IOTAS = [0.5]
# ALPHAS = [0.9]


# grid search: iota = [0.5, 0.6, ..., 1.5] and alpha = [0.80, 0.81, ..., 0.99]
IOTAS = [round(0.5 + 0.1 * i, 2) for i in range(11)]
ALPHAS = [round(0.80 + 0.01 * i, 2) for i in range(20)]


# fixed SOGP settings (same as run_sogp_admm_cluster.py)
BURN_IN = 5
TRAIN_ITER = 100
LR = 0.01
SEED = 123
DTYPE = torch.float64

ADMM_CFG = ADMMConfig(zeta=1.0, rho=10, max_iters=200, eps_abs=1e-6, eps_rel=1e-3)

N_WORKERS = 8  # parallel runs

# DATA_DIR = os.path.join(ROOT, "data_generation")
# OUTPUT_DIR = os.path.join(ROOT, "results", FUNCTION, METHOD)

## ------- Sweeping throgh 20 Agents Quadratic datasets  -------

DATA_DIR = os.path.join(ROOT, "data_generation", "20 Agents")
OUTPUT_DIR = os.path.join(ROOT, "results", "20 Agents Quadratic", "sogp")


def run_sogp_once(dataset, iota, alpha):
    # one SOGP-ADMM run; same pipeline as run_sogp_admm_cluster.py
    set_seed(SEED)  # workers are reused, so re-seed before every run
    torch.set_default_dtype(DTYPE)

    # SOGP keeps one GP per agent, so the cluster labels are not used
    gamma_list, theta_list, _ = load_data_from_json(os.path.join(DATA_DIR, f"{dataset}.json"))

    local_models = build_local_models(gamma_list, theta_list, local_model=FUNCTION, lambda4=LAMBDA4)
    coord = Coordinator(gamma_list, theta_list, ADMM_CFG, burn_in=BURN_IN, local_models=local_models)

    sogp = SOGPMoreauProvider(
        gamma_list=gamma_list,
        theta_list=theta_list,
        rho=ADMM_CFG.rho,
        zeta=ADMM_CFG.zeta,
        min_points=BURN_IN,
        iota=iota,
        alpha=alpha,
        train_iters=TRAIN_ITER,
        lr=LR,
        device="cpu",
        dtype=DTYPE,
        eval_exact_error=False,
        local_model=FUNCTION,
        lambda4=LAMBDA4,
    )

    def moreau_provider(i, z_i, query_flag, k):
        return sogp(i, z_i, query_flag, k)

    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # drop the per-agent, per-iteration prints
        out = coord.run(moreau_provider, verbose=False)
    elapsed = time.time() - start

    # out is returned exactly as Coordinator.run gives it; the wall time goes separately
    return out, elapsed


def base_row(task):
    # identity + fixed settings of one run (the same for ok and failed runs)
    dataset, iota, alpha = task["dataset"], task["iota"], task["alpha"]
    return {
        "run_id": task["run_id"],
        "method": METHOD,
        "function": FUNCTION,
        "dataset": dataset,
        "iota": iota,
        "alpha": alpha,
        "traj_path": os.path.join(dataset, "trajectories", f"{config_name(iota, alpha)}.json"),
        "burn_in": BURN_IN,
        "train_iter": TRAIN_ITER,
        "lr": LR,
        "rho": ADMM_CFG.rho,
        "zeta": ADMM_CFG.zeta,
        "eps_abs": ADMM_CFG.eps_abs,
        "eps_rel": ADMM_CFG.eps_rel,
        "max_iters": ADMM_CFG.max_iters,
        "seed": SEED,
        "dtype": str(DTYPE).replace("torch.", ""),
    }


def sweep_task(task):
    # runs in a worker: one configuration -> one summary row; never raises,
    # so a crashing configuration (e.g. NotPSDError) does not stop the sweep
    row = base_row(task)
    start = time.time()
    try:
        out, elapsed = run_sogp_once(task["dataset"], task["iota"], task["alpha"])

        # trajectory = the full out of this run (SharingADMM outputs + Coordinator comm totals)
        traj_file = os.path.join(OUTPUT_DIR, row["traj_path"])
        os.makedirs(os.path.dirname(traj_file), exist_ok=True)
        save_run_output(out, traj_file)

        row.update({"status": "ok", **outcome_metrics(out, elapsed), "error": ""})

    except Exception as exc:
        error_file = os.path.join(OUTPUT_DIR, task["dataset"], "errors",
                                  f"{config_name(task['iota'], task['alpha'])}.json")
        info = {key: row[key] for key in ("run_id", "dataset", "iota", "alpha")}
        info["elapsed_s"] = round(time.time() - start, 3)
        save_error(error_file, info, exc)

        row["traj_path"] = ""  # no trajectory for a failed run
        row.update({"status": "failed", **empty_outcome(), "error": short_error(exc)})

    return row


def write_sweep_config():
    # the grid and every fixed setting of this sweep, for reproducibility
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = {
        "method": METHOD,
        "function": FUNCTION,
        "lambda4": LAMBDA4,
        "datasets": DATASETS,
        "iotas": IOTAS,
        "alphas": ALPHAS,
        "burn_in": BURN_IN,
        "train_iter": TRAIN_ITER,
        "lr": LR,
        "seed": SEED,
        "dtype": str(DTYPE).replace("torch.", ""),
        "admm": vars(ADMM_CFG),
        "n_workers": N_WORKERS,
        "n_threads_per_run": N_THREADS,
    }
    with open(os.path.join(OUTPUT_DIR, "sweep_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    write_sweep_config()

    tasks = [
        {"run_id": make_run_id(dataset, iota, alpha), "dataset": dataset, "iota": iota, "alpha": alpha}
        for dataset in DATASETS
        for iota in IOTAS
        for alpha in ALPHAS
    ]

    run_sweep(tasks, sweep_task, n_workers=N_WORKERS, output_dir=OUTPUT_DIR, n_threads=N_THREADS)
    print("Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
