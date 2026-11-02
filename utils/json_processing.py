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

def save_run_output(out, json_path):
    # full record of one ADMM run: every output of SharingADMM.run_moreau_admm plus the
    # communication totals added by Coordinator.run. Each field is read with out["key"],
    # so a missing output raises instead of saving an incomplete record.
    # If SharingADMM gains a new output, add one line here.
    # Load with json.load(f): same keys as below.
    record = {
        # ---- final state of the run ----
        "iters": out["iters"],          # last iteration index k (0-based); n_iter = len(obj_hist)
        "converged": out["converged"],  # both residuals below their tolerances
        "r": out["r"],                  # final primal residual
        "s": out["s"],                  # final dual residual
        "x": out["x"],                  # final x_i of every agent, (N, p)
        "y_blocks": out["y_blocks"],    # final y_i of every agent, (N, p)
        "y_bar": out["y_bar"],          # final y_bar, (p,)
        "u_bar": out["u_bar"],          # final scaled dual u_bar, (p,)

        # ---- per-iteration histories, (n_iter,) ----
        "r_hist": out["r_hist"],                # primal residual
        "s_hist": out["s_hist"],                # dual residual
        "eps_pri_hist": out["eps_pri_hist"],    # primal tolerance
        "eps_dual_hist": out["eps_dual_hist"],  # dual tolerance
        "obj_hist": out["obj_hist"],            # objective sum_i f_i(x_i) + zeta * ||sum_i y_i||_1
        "comm_per_iter": out["comm_per_iter"],  # number of agents that queried exactly

        # ---- per iteration and per agent, (n_iter, N, ...) ----
        "comm_mask_per_iter": out["comm_mask_per_iter"],      # True = agent queried exactly
        "query_points_hist": out["query_points_hist"],        # query point q_i of each agent, (n_iter, N, p)
        "grads_error_per_iter": out["grads_error_per_iter"],  # ||g_true - g_GP|| before projection; 0.0 when queried
        "stats_per_iter": out["stats_per_iter"],              # GP decision stats (g_var, var_max, psi_k, do_query); {} in burn-in

        # ---- communication totals (added by Coordinator.run) ----
        "comm_total": out["comm_total"],              # total exact queries over the run
        "comm_rounds_iter": out["comm_rounds_iter"],  # iterations in which at least one agent queried
        "comm_per_agent": out["comm_per_agent"],      # total exact queries of each agent, (N,)
        "comm_cum": out["comm_cum"],                  # cumulative queries per iteration, (n_iter,)
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, default=to_jsonable)