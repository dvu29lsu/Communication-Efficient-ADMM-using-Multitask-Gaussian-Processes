import scipy as sp
import numpy as np
import pandas as pd
import json
from io import StringIO

#checking the hessian matrix is PSD or  not

def checking_psd(matricies):
    for i in range(len(matricies)):
        eigen_vals = np.linalg.eigvals(matricies[i])
        if np.all(eigen_vals >= 0):
            print(f"The hessian of function {i} satisfying")
        else:
            print(f"local function {i} does not satisfying")


def load_data_from_mat(mat_path):

    data = sp.io.loadmat(mat_path)
    #extract data

    gamma_list = data["Phi"][0] ## hessian for each agent local function ## they are symetric
    
    theta_list = [np.asarray(t).reshape(-1) for t in data["theta"][0]] ## theta of the f_i(x_i) = (x_i-theta_i) @ phi @ (x_i-theta_i)

    # theta_list = data['theta'][0] 

    return gamma_list, theta_list


def load_data_from_jsonl(jsonl_path):
    # data = pd.read_json(StringIO(json_path), lines=True)

    data = pd.read_json(jsonl_path, lines=True, encoding='utf-8')

    # print(data.head())

    gamma_list = data["gamma_list"][0]
    theta_list = data["theta_list"][0]
    labels = data["labels"]

    return np.array(gamma_list), np.array(theta_list), np.array(labels)


def load_data_from_json(json_path):
    # read one dataset written by data_generation/data_generation.ipynb
    # the file is a single (pretty-printed) JSON object, not JSON Lines
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gamma_list = np.asarray(data["gamma_list"], dtype=float) # (N, p, p) hessian-like SPD matrices
    theta_list = np.asarray(data["theta_list"], dtype=float) # (N, p) minimizers
    labels = np.asarray(data["labels"], dtype=int).reshape(-1) # (N,) cluster label of each agent

    return gamma_list, theta_list, labels


def load_direction_matrix(json_path):
    # read the fixed direction matrix A of the directional quartic objective
    # written by data_generation/generate_direction_matrix.py
    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    A = np.asarray(record["matrix"], dtype=float) # (p, p), rows are the quartic directions

    # fail early on a wrong file: A must be square and orthogonal (A A^T = I)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Direction matrix must be square, got shape {A.shape}")

    orth_error = np.linalg.norm(A @ A.T - np.eye(A.shape[0]), ord="fro")
    if orth_error > 1e-10:
        raise ValueError(f"Direction matrix is not orthogonal, ||A A^T - I||_F = {orth_error:.3e}")

    return A
    
