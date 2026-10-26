import numpy as np
import torch
import gpytorch

# Date: Jan 28, 26 - Long Vu
# from prediction of the mtgp model, we can define a set of ellipsoids
# as follow the paper: ommunication reduction in distributed  
# optimization via estimation of the proximal  operator

# compute the ellipsoids constraints following the eq (14) in the paper
# approx of the ellipsoids will satisfying:
# gamma = 1/rho

# G(v;z_j) := {g | gamma|| g - grad_j||_2^2 - <g - grad_j, v - z_j> <=0}
# expanding it we can get: 
# g^T.(gamma.I).g + ((-2.gamma.grad_j) - (v - z_j))^T@g + (gamma.||grad_j||^2 + <grad_j, v - z_j>) <= 0

# define the constrains interm of g^T @ A g + b^T g + c <=0
# maching we can compute the A, b and c

# v: current query point
# z: past point want to form the ellipsoids
# d: gradient at point z

def compute_constrain_sets(v, Z_hist, Y_hist, gamma, max_num_ellipsoids=5):
    # compute the ellipsoids constrains sets for all the past points
    # Z_hist: list of past query points
    # Y_hist: list of past gradient points for each query points in Z_hists

    p = v.shape[0]

    m = min(len(Z_hist), max_num_ellipsoids) # num ellipsoid will use

    A = gamma * np.eye(p) # same at most
    
    B = np.zeros((p, m))

    c = np.zeros((m,))

    for i in range(m):
        z = Z_hist[-(i+1)] # get the point from last m-th index

        y = Y_hist[-(i+1)] # observation data at point z

        d = y[1:] # gradient part y = [m_val, grad_1, grad_2,..., grad_p]

        delta = v - z

        # vector 
        b = -gamma * d - 0.5 * delta #  2b^T@g = (-2.gamma.grad_j - (v - z_j))^T@g

        # scalar
        cj = gamma * np.dot(d, d) + np.dot(d, delta) # gamma.||grad_j||^2 + <grad_j, v - z_j>

        B[:, i] = b

        c[i] = cj

    return A, B, c



