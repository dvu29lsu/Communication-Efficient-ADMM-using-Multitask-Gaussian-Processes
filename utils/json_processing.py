import json
import numpy as np


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f"Not JSON serializable: {type(x)}")

def process_output_to_jsonl(output_data, jsonl_output_path):
    r_hist = output_data.get("r_hist", [])
    s_hist = output_data.get("s_hist", [])
    obj_hist = output_data.get("obj_hist", [])
    comm_per_iter = output_data.get("comm_per_iter", [])
    stats_per_iter = output_data.get("stats_per_iter", [])
    grads_error_per_iter = output_data.get("grads_error_per_iter", [])

    n = len(r_hist)

    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            record = {
                "iteration": i,
                "primal_residual": r_hist[i],
                "dual_residual": s_hist[i],
                "obj_val": obj_hist[i],
                "comm_per_iter": comm_per_iter[i],
                "grads_error_per_iter": grads_error_per_iter[i], # gradient error at this iteration (each is norm 2)
                "other_stats": stats_per_iter[i],  # more detail stats information in iteration i
            }
            f.write(json.dumps(record, default=to_jsonable) + "\n")