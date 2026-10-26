import numpy as np
from .proximal_ops import soft_threshold
from .solve_linsys import cho_solve, safe_cholesky

class SharingADMM:
    def __init__(self, gamma_list, theta_list, zeta = 1.0, rho=1.0, eps_abs=1e-4, eps_rel = 1e-3, n_iters=300, local_models=None):
        self.N = len(gamma_list)
        # assert self.N == len(theta_list), "gamma theta mismatch"

        self.p = theta_list[0].shape[0] ## input dimension

        # getting hession matrix
        self.G = [np.array(G, float) for G in gamma_list]

        # getting the theta params
        self.theta = [np.array(t, float).reshape(-1) for t in theta_list]

        # setting the zeta and also the rho for the proximal and the regularization norm 1
        self.zeta = float(zeta)
        self.rho = float(rho)
        
        # set the residual
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)

        self.max_iters = int(n_iters)

        #starting variable
        # self.x = [np.zeros(self.p) for _ in range(self.N)] # set the initfial value
        # self.z_blocks = [np.zeros(self.p) for _ in range(self.N)]
        # self.z_bar = np.zeros(self.p)
        # self.u_bar = np.zeros(self.p)

        self.x = [np.zeros(self.p) for _ in range(self.N)]
        self.y_blocks = [np.zeros(self.p) for _ in range(self.N)]
        self.y_bar = np.zeros(self.p)
        self.u_bar = np.zeros(self.p)

        #cholesky cache
        self._chol_cache = [None]*self.N

        #residual history collects
        self.r_hist = []
        self.s_hist = []

        #query history collects
        self.query_points_hist = [] #store the query history for each agent

        #eps_pri_hist
        self.eps_pri_hist = [] #store the primal dishory
        self.eps_dual_hist = [] #store the dual residual history
        self.obj_hist = [] #store the objective history

        #comm hist
        self.comm_per_iter = []
        self.comm_mask_per_iter = []
        self.stats_per_iter = [] # list of per-agent dict
        self.grads_error_per_iter = [] # list of per-agent grad error
        
        # convergence flag
        self.converged = False

        # define the local models/functions that form the obj for each agent
        self.local_models = local_models

# ============= April, 7th Adapt new ver for the test ting with many different function =============================================

    #### OLD
    # compute the objective as the sum of each prox + norm 1 
    # def _objective(self):
    #     # sum (x_i - theta)^T G_i (x_i - theta) + zeta * ||z_bar||

    #     val = 0.0
    #     for i in range(self.N):
    #         d = self.x[i] - self.theta[i]
    #         val += float(d.T @ self.G[i] @ d)
        
    #     val += float(self.zeta * np.linalg.norm(np.sum(self.z_blocks, axis=0), ord=1)) # adding the norm 1 part

    #     return val

    #### NEW: adaptive different local models / functions
    def _objective(self):
        val = 0.0

        for i in range(self.N):
            if self.local_models is None:
                # quadratic objective
                d = self.x[i] - self.theta[i]
                val += float(d.T @ self.G[i] @ d) # just the quadratic function if not define the local_model
            else:
                # just the local function value without the proximal part
                val += float(self.local_models[i].value(self.x[i])) 

        # val += float(self.zeta * np.linalg.norm(np.sum(self.z_blocks, axis=0), ord=1))
        val += float(self.zeta * np.linalg.norm(np.sum(self.y_blocks, axis=0), ord=1))
        return val
    
# =====================================================================================================
    def _residuals(self, y_prev_blocks):
        N, rho = self.N, self.rho

        # primal residual: ||x - y||
        r = np.sqrt(sum(
            np.dot(
                self.x[i] - self.y_blocks[i],
                self.x[i] - self.y_blocks[i]
            )
            for i in range(N)
        ))

        # dual residual: rho * ||y - y_prev||
        s = rho * np.sqrt(sum(
            np.dot(
                self.y_blocks[i] - y_prev_blocks[i],
                self.y_blocks[i] - y_prev_blocks[i]
            )
            for i in range(N)
        ))

        xnorm = np.sqrt(sum(np.dot(xi, xi) for xi in self.x))
        ynorm = np.sqrt(sum(np.dot(yi, yi) for yi in self.y_blocks))
        unorm = rho * np.sqrt(N) * np.linalg.norm(self.u_bar)

        return r, s, xnorm, ynorm, unorm
    
    #compute the primal residual
    def _eps_pri(self, xnorm, ynorm):
        return ((self.N * self.p)**0.5 * self.eps_abs + self.eps_rel * max(xnorm, ynorm)) ## following the formular of boyd book
    
    #compute the dual residual
    def _eps_dual(self, unorm):
        return (self.N * self.p)**0.5 * self.eps_abs + self.eps_rel * unorm ## following the boyd book (modify 03/01/26)

    # def update_step(self, moreau_update_list):
    #     N = self.N
    #     rho = self.rho
    #     x_bar = sum(self.x)/(self.N)

    #     # curretn local z_i^k defined by consensus variable 
    #     z_i = [self.x[i] + self.z_bar - x_bar - self.u_bar for i in range(N)]

    #     # update x via gradient of moreau envelope
    #     for i in range(self.N):
    #         self.x[i] = z_i[i] - (1/rho)*moreau_update_list[i] ## prox = y - muy * gradient_moreau & muy = 1/rho

    #     # z_bar and u_bar update.
    #     # z_bar = soft(a_bar, zeta/rho)
    #     # u_bar = a_bar - z_bar where a_bar = u_bar + x_bar
    #     # we have a = x_i + u_bar. We have the 
    #     a = [self.x[i] + self.u_bar for i in range(N)]
    #     a_bar = sum(a)/N

    #     tau = self.zeta / rho
        
    #     #update z_bar
    #     self.z_bar = soft_threshold(a_bar, tau)

    #     #update z_i
    #     for i in range(N):
    #         self.z_blocks[i] = a[i] + (self.z_bar - a_bar)

    #     #update u_bar u_bar = u_bar + x_bar - z_bar
    #     self.u_bar = a_bar - self.z_bar

    def update_step(self, moreau_update_list):
        N = self.N
        rho = self.rho
        x_bar = sum(self.x) / N

        # query points computed from the previous ADMM state
        q_list = [
            self.x[i] + self.y_bar - x_bar - self.u_bar
            for i in range(N)
        ]

        # update the local variables x
        for i in range(N):
            self.x[i] = q_list[i] - (1 / rho) * moreau_update_list[i]

        a = [self.x[i] + self.u_bar for i in range(N)]
        a_bar = sum(a) / N

        # update the auxiliary variables y
        tau = self.zeta / rho
        self.y_bar = soft_threshold(a_bar, tau)

        for i in range(N):
            self.y_blocks[i] = a[i] + (self.y_bar - a_bar)

        # update the scaled dual variable u
        self.u_bar = a_bar - self.y_bar

    def run_moreau_admm(self, moreau_grad_provider, burn_in=5, verbose=True, log_every=1):
        # moreau_grad_provider here adapt the moreau envelope mechanism but not using gps

        #reset log
        self.r_hist = []
        self.s_hist = []
        self.eps_pri_hist = []
        self.eps_dual_hist = []
        self.obj_hist = []
        self.comm_per_iter = []
        self.comm_mask_per_iter = []
        self.stats_per_iter = []
        self.grads_error_per_iter = []
        self.query_points_hist = []

        N = self.N

        # start
        # z_prev = [zb.copy() for zb in self.z_blocks]
        # x_bar = sum(self.x)/N
        # z_list = [self.x[i] + self.z_bar - x_bar - self.u_bar for i in range(N)]

        ## ============= In case changing the starting point to non zero ============
        # # first iteration
        # moreau_grad_0 = [moreau_grad_provider(i, z_list[i], True, 0) for i in range(N)]
        # 
        # #update using the moreau evelop gradient information
        # self.update_step(moreau_grad_0)

        # #residual at the k = 0

        # r, s, xnorm, znorm, unorm = self._residuals(z_prev)
        # self.r_hist.append(float(r))
        # self.s_hist.append(float(s))

        # e_pri = self._eps_pri(xnorm, znorm)
        # e_dual = self._eps_dual(unorm)

        # if verbose:
        #     print(f"iter 0: r={r:3e} -- s={s:3e}")
        # ===============================================================
        
        # for k in range(self.max_iters):
        #     z_prev = [zb.copy() for zb in self.z_blocks]
        #     x_bar = sum(self.x)/N
        #     z_list = [self.x[i] + self.z_bar - x_bar - self.u_bar for i in range(N)]

        #     moreau_grad_k = []

        #     # communicate to get the moreau gradient
        #     for i in range(N):
        #         query_flag = (k < burn_in)
        #         moreau_grad_k.append(moreau_grad_provider(i, z_list[i], query_flag, k))

        #     # updaet using moreau env
        #     self.update_step(moreau_grad_k)

        #     # residuals at k
        #     r, s, xnorm, znorm, unorm = self._residuals(z_prev)

        #     self.r_hist.append(float(r))
        #     self.s_hist.append(float(s))

        #     e_pri = self._eps_pri(xnorm, znorm)
        #     e_dual = self._eps_dual(unorm)

        #     if verbose and (k % log_every == 0 or (r <= e_pri and s <= e_dual)):
        #         print(f"iter {k}: r={r:.3e} (eps_pri={e_pri:.2e}) | s={s:.3e} (eps_dual={e_dual:.2e})")

        #     if r <= e_pri and s <= e_dual:
        #         break

        # return dict(
        #     iters=k, r=r, s=s, eps_pri=e_pri, eps_dual=e_dual,
        #     x=[xi.copy() for xi in self.x],
        #     z_blocks=[zi.copy() for zi in self.z_blocks],
        #     z_bar=self.z_bar.copy(), u_bar=self.u_bar.copy(),
        #     r_hist=self.r_hist, s_hist=self.s_hist,
        # )

        for k in range(self.max_iters):
            # z_prev = [zb.copy() for zb in self.z_blocks]
            # x_bar = sum(self.x) / N
            # z_list = [self.x[i] + self.z_bar - x_bar - self.u_bar for i in range(N)] # query infor compute 
            y_prev = [yi.copy() for yi in self.y_blocks]

            x_bar = sum(self.x) / N

            q_list = [
                self.x[i] + self.y_bar - x_bar - self.u_bar
                for i in range(N)
            ]

            # store the query points for each agent at iteration k
            self.query_points_hist.append([qi.copy() for qi in q_list])

            grads = []
            comm_mask = np.zeros(N, dtype=bool)
            stats_list = [None] * N
            grads_error_list = [None] * N

            for i in range(N):
                # query flag to getting the training data and active GP
                query_flag = (k < burn_in) # query flag to getting training data

                # getting grad query of the agent i
                # ret = moreau_grad_provider(i, z_list[i], query_flag, k)
                ret = moreau_grad_provider(i, q_list[i], query_flag, k)


                if len(ret) < 4:
                    g_i, did_query, stats = ret
                    grad_err = None
                # in case using the vanilla with exact comm
                else:
                    g_i, did_query, stats, grad_err = ret

                # print(f"Agent {i+1} at iter {k}: grad = {g_i}, stats = {stats}, did_query = {did_query}") # debug
                
                ## getting gradient to update iteration
                grads.append(g_i)

                # getting grad error for agent i
                grads_error_list[i] = grad_err

                # record the agent i communicate at this iteration
                comm_mask[i] = bool(did_query)
                
                # stores extra for plotting
                stats_list[i] = stats

            # update ADMM 
            self.update_step(grads)

            # compute the residual, dual and objective
            # r, s, xnorm, znorm, unorm = self._residuals(z_prev)

            # e_pri = self._eps_pri(xnorm, znorm)
            
            # e_dual = self._eps_dual(unorm)
            r, s, xnorm, ynorm, unorm = self._residuals(y_prev)
            e_pri = self._eps_pri(xnorm, ynorm)
            e_dual = self._eps_dual(unorm)
            
            obj = self._objective()

            self.r_hist.append(float(r))
            self.s_hist.append(float(s))
            self.eps_pri_hist.append(float(e_pri))
            self.eps_dual_hist.append(float(e_dual))
            self.obj_hist.append(float(obj))

            self.comm_mask_per_iter.append(comm_mask.copy())
            self.comm_per_iter.append(int(comm_mask.sum()))
            self.stats_per_iter.append(stats_list)
            self.grads_error_per_iter.append(grads_error_list)
            
            if verbose:
                print(f"Iter {k + 1}: r={r:.3e} (<= {e_pri:.2e}) | s={s:.3e} (<= {e_dual:.2e}) | obj={obj:.6e} | q={int(comm_mask.sum())}")

            if r <= e_pri and s <= e_dual:
                self.converged = True
                break
            elif k == self.max_iters - 1:
                print(f"Reached max iters {self.max_iters} without convergence")
                self.converged = False
                
        # return whole trajectory of the ADMM run
        return dict(
            iters=k,
            r=float(self.r_hist[-1]),
            s=float(self.s_hist[-1]),
            x=[xi.copy() for xi in self.x],
            # z_blocks=[zi.copy() for zi in self.z_blocks],
            # z_bar=self.z_bar.copy(),
            y_blocks=[yi.copy() for yi in self.y_blocks],
            query_points_hist=self.query_points_hist,
            y_bar=self.y_bar.copy(),    
            u_bar=self.u_bar.copy(),
            r_hist=self.r_hist,
            s_hist=self.s_hist,
            eps_pri_hist=self.eps_pri_hist,
            eps_dual_hist=self.eps_dual_hist,
            obj_hist=self.obj_hist,
            comm_per_iter=self.comm_per_iter,
            comm_mask_per_iter=self.comm_mask_per_iter,
            stats_per_iter=self.stats_per_iter,
            grads_error_per_iter=self.grads_error_per_iter,
            converged=self.converged,
        )
















        


    




