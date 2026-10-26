import numpy as np


# max std -> may sensitive for the mtgp
def max_std_from_var(var_vec: np.ndarray) -> float:
    return float(np.sqrt(np.max(var_vec)))

def should_query_max_std(std_max: float, psi_k: float) -> bool:
    return bool(std_max > psi_k)


# max var
def max_var(var_vec: np.array) -> float:
    return float(np.max(var_vec))

def should_query_max_var(var_max: float, psi_k: float) -> bool:
    return bool(var_max > psi_k)