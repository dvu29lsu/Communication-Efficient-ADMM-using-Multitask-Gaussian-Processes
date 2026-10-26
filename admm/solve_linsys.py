import numpy as np

def cho_solve(L, b):
    #solving the (L L^T = b)

    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


#safe cho_solve 
def safe_cholesky(A, jitter=1e-10, max_tries=5):
    I = np.eye(A.shape[0])
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(A + (jitter*(10**k))*I)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed even with jitter.")