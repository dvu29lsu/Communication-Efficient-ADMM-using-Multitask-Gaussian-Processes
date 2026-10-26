import numpy as np
import torch
import gpytorch
import os
import time
import json
import csv

from utils.config import ADMMConfig
from utils.coordinator import Coordinator
from utils.csv_writer import write_summary_row, extract_metrics
from agents_loader.agent_load import load_data_from_mat
from utils.json_processing import process_output_to_jsonl
from run_mtgp_admm import run_mtgp
from ADMM_MTGP.miscellaneous.old_scripts.run_sogp_admm import run_sogp

from query_method.mogp.mtgp_provider import MTGPMoreauGradProvider
from query_method.sogp.sogp_provider import SOGPMoreauProvider

# data path
data_path = os.path.join("data", "10 Agents", "p5")

# trying to run 10 times MTGP adn SOGP with set 1 and getting result
# each method will save each time run in a separate csv summary result, not need to save jsonl

def main():
    # define the selected set for run n_times
    SETS = ["Set1.mat", "Set2.mat", "Set4.mat", "Set6.mat", "Set7.mat"]
    # set_name = "Set1.mat"

    # mat_file = os.path.join(data_path, set_name)

    # num training points
    burn_in = 5

    # config for the MTGP provider
    mtgp_iota = 1.3
    mtgp_alpha = 0.96
    mtgp_train_iter = 300
    mtgp_lr = 0.03
    mtgp_Q = 5
    mtgp_rank = 1
    
    # config for the SOGP provider
    gp_iota = 0.6
    gp_alpha = 0.8
    gp_train_iter = 100
    gp_lr = 0.01

    n_runs = 10 # for each method

    # method = ["mtgp", "sogp"]
    method = ["sogp"] # just for comparing 

    # running mtgp-admm and sogp-admm for set1 for 10 times each with the same config and save the result in a separate csv file for each method
    for set_name in SETS:
        mat_file = os.path.join(data_path, set_name)
        for run in range(n_runs):
            print(f"Running iteration {run+1}/{n_runs} for set {set_name}")
            # run mtgp-admm
            if "mtgp" in method:
                # create the path for saving data for the mtgp result for this run
                mtgp_base_dir = os.path.join("results", "mtgp", set_name[:-4], f"iota_{mtgp_iota}_alpha_{mtgp_alpha}_train_{mtgp_train_iter}_lr_{mtgp_lr}_Q_{mtgp_Q}_rank_{mtgp_rank}")
                os.makedirs(mtgp_base_dir, exist_ok=True)
                
                # mtgp csv path
                mtgp_csv_path = os.path.join(mtgp_base_dir, f"mtgp_summary_run_{n_runs}_{set_name[:-4]}.csv")

                out = run_mtgp(mat_file, burn_in=burn_in, iota=mtgp_iota, alpha=mtgp_alpha, train_iter=mtgp_train_iter, lr=mtgp_lr, Q=mtgp_Q, rank=mtgp_rank)
                
                n_iter, n_comm, final_obj, converged = extract_metrics(out)

                # write csv summary for the mtgp
                summary_row = {
                    "set": set_name,
                    "n_iter": n_iter,
                    "n_comm": n_comm,
                    "objective": final_obj,
                    "time": out.get("elapsed", 0),
                    "converged": converged
                }
                write_summary_row(mtgp_csv_path, summary_row)

            # run sogp-admm
            if "sogp" in method:
                gp_base_dir = os.path.join("results", "sogp", set_name[:-4], f"iota_{gp_iota}_alpha_{gp_alpha}_train_{gp_train_iter}_lr_{gp_lr}")
                os.makedirs(gp_base_dir, exist_ok=True)
                gp_csv_path = os.path.join(gp_base_dir, f"sogp_summary_run_{n_runs}_{set_name[:-4]}.csv")
                out = run_sogp(mat_file, burn_in=burn_in, iota=gp_iota, alpha=gp_alpha, train_iter=gp_train_iter, lr=gp_lr)
                n_iter, n_comm, final_obj, converged = extract_metrics(out)
                
                # write the output to the summary csv for sogp
                summary_row = {
                    "set": set_name,
                    "n_iter": n_iter,
                    "n_comm": n_comm,
                    "objective": final_obj,
                    "time": out.get("elapsed", 0),
                    "converged": converged
                }
                write_summary_row(gp_csv_path, summary_row)

    
if __name__ == "__main__":
    main()