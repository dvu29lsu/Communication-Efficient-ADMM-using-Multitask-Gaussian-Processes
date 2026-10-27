## generate the fixed orthogonal direction matrix A for the directional quartic objective
## f_i(x) = u^T Gamma_i u + lambda4 * sum((A u)^4),  u = x - theta_i
## the same A is shared by every agent, dataset, and method, so it is generated once and saved

## recipe and seed follow the ICC paper experiments (ADMM_MTGP), so the matrix is identical
## run from the repo root:  python data_generation/generate_direction_matrix.py

import os
import json
import numpy as np

SEED = 20260813
DIMENSION = 5 # input dimension p of the generated datasets

# save next to this script, independent of the working directory
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "directional_quartic_dim5.json")


def generate_orthogonal_matrix(seed, dimension):
    # QR of a standard normal matrix gives an orthogonal Q
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(raw)

    # QR is unique only up to column signs -> force diag(R) > 0 so the result is reproducible
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)

    # rows of A are the quartic directions
    return (q * signs).T


def save_direction_matrix(path, matrix, seed):
    record = {
        "seed": seed,
        "dimension": matrix.shape[0],
        "matrix": matrix.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def main():
    A = generate_orthogonal_matrix(SEED, DIMENSION)

    # sanity check: A A^T = I
    orth_error = np.linalg.norm(A @ A.T - np.eye(DIMENSION), ord="fro")
    assert orth_error < 1e-10, f"A is not orthogonal, error = {orth_error:.3e}"

    save_direction_matrix(OUTPUT_PATH, A, SEED)

    # print("Orthogonality error ||A A^T - I||_F:", orth_error)
    # print("Saved direction matrix to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
