import os
import csv
import math

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_iter_metrics_csv(save_path: str, rows: list[dict]):
    """
    rows: list of dicts with identical keys
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def write_agent_metrics_csv(save_path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
