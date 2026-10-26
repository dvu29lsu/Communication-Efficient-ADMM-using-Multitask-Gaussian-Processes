import numpy as np
from utils.projection_subdifferential import project_to_ellipsoid

# Date: Jan 28, 26 - Long Vu

# checking the gradient part inside the gradient ellipsoid or not

# passing the qualify: g^T @ A g +2b^T g + c >= tols o 

def in_ellipsoid_intersec(g, A, B, c, tols = 1e-12):
    
    gAg = g.T @ (A @ g)

    for i in range(B.shape[1]):
        val = gAg + 2 * B[:, i].T @ g + c[i]

        if val >= tols: # not inside
            return False
        
    return True

def projection2ellipsoid(g0, A, B, c):

    # if B.shape[1] == 0:
    #     return g0, False
    
    if in_ellipsoid_intersec(g0, A, B, c, tols=1e-12):
        return g0, False
    
    g_proj, status = project_to_ellipsoid(g0, A, B, c, solver="SCS", max_iters = 3000)

    return g_proj, status
