import os
import csv
import numpy as np


def write_summary_row(csv_path: str, row: dict):
    # required columns per your request
    fieldnames = ["set", "n_iter", "n_comm", "objective", "time", "converged"]
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