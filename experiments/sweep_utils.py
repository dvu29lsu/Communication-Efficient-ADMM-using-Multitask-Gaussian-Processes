# Shared bookkeeping for the (iota, alpha) sweeps of every method and function type.
# A sweep script (e.g. experiments/quadratic/sweep_sogp.py) only defines its settings and
# how to run its method once; this module handles run ids, resume, result rows, error files
# and the worker pool.
#
# Layout written under output_dir (e.g. results/quadratic/sogp/):
#   summary.csv                                   every run (ok and failed), one row each
#   <dataset>/summary.csv                         the runs of one dataset (same rows)
#   <dataset>/trajectories/iota0.50__alpha0.80.jsonl
#   <dataset>/errors/iota0.50__alpha0.81.json     only for runs that raised

import csv
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import torch


# outcome columns of a summary row, in a fixed order so ok and failed rows share one header
OUTCOME_FIELDS = ["converged", "n_iter", "total_queries", "final_obj", "final_r", "final_s", "elapsed_s"]


def config_name(iota, alpha):
    # file name of one configuration inside its dataset folder
    return f"iota{iota:.2f}__alpha{alpha:.2f}"


def make_run_id(dataset, iota, alpha):
    # unique across the whole sweep; links a summary row to its trajectory / error file
    return f"{dataset}__{config_name(iota, alpha)}"


def outcome_metrics(out, elapsed):
    # raw facts of one finished run, from the dict returned by Coordinator.run;
    # the wall time is passed separately so `out` stays exactly as Coordinator.run returned it
    # (n_iter counts logged iterations; the solver's own "iters" field is the 0-based k)
    return {
        "converged": bool(out["converged"]),
        "n_iter": len(out["obj_hist"]),
        "total_queries": int(out["comm_total"]),
        "final_obj": float(out["obj_hist"][-1]),
        "final_r": float(out["r"]),
        "final_s": float(out["s"]),
        "elapsed_s": round(float(elapsed), 3),
    }


def empty_outcome():
    # outcome columns of a failed run: left empty, so the row keeps the same header
    return {key: "" for key in OUTCOME_FIELDS}


def save_error(path, info, exc):
    # error file of a run that raised: its identity plus the full traceback
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(info)
    record["error_type"] = type(exc).__name__
    record["message"] = str(exc)
    record["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def short_error(exc):
    # one-line error for the summary row; the full traceback is in the error file
    lines = str(exc).strip().splitlines()
    first_line = lines[0] if lines else ""
    return f"{type(exc).__name__}: {first_line}"


def completed_run_ids(summary_csv):
    # run ids already in the summary (ok or failed); runs are deterministic, so they are not retried
    if not os.path.exists(summary_csv):
        return set()
    with open(summary_csv, newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def append_row(csv_path, row):
    # append one row; write the header first if the file is new
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def init_worker(n_threads):
    # each worker runs one ADMM run at a time on a fixed number of threads
    torch.set_num_threads(n_threads)


def run_sweep(tasks, task_fn, n_workers, output_dir, n_threads=1):
    # tasks: list of dicts with at least "run_id" and "dataset"
    # task_fn(task) -> summary row (dict); it must catch its own errors and never raise
    summary_csv = os.path.join(output_dir, "summary.csv")

    # resume: skip every run that already has a row
    done = completed_run_ids(summary_csv)
    todo = [t for t in tasks if t["run_id"] not in done]
    print(f"{len(tasks)} runs in the sweep, {len(done)} already done, {len(todo)} runs left")
    if not todo:
        return

    # spawn: fresh worker processes that re-import the sweep script, so its thread
    # env vars are set before torch loads (forking after torch is imported is fragile)
    ctx = multiprocessing.get_context("spawn")

    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=init_worker, initargs=(n_threads,)) as pool:
            futures = [pool.submit(task_fn, t) for t in todo]

            # only the main process writes the summaries, so there are no write races
            for k, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                append_row(summary_csv, row)
                append_row(os.path.join(output_dir, row["dataset"], "summary.csv"), row)
                print(f"[{k}/{len(todo)}] {row['run_id']:42s} {row['status']:6s} "
                      f"iters={row['n_iter']} queries={row['total_queries']} time={row['elapsed_s']}s")

    except BrokenProcessPool:
        # a worker was killed (segfault, out of memory): finished rows are already saved
        print("Worker pool broke (a run was killed). Re-run the sweep to resume from where it stopped.")
