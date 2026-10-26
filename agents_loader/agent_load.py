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
    
