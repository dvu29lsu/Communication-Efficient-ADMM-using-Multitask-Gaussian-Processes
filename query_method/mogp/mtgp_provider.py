import torch
import gpytorch
import numpy as np

from scipy.optimize import minimize

from mogp.mtgp_derivative_model import ExactLCMGrad
from mogp.train_mtgp import train_mtgp_exact
from admm.exact_quadsolver import solving_moreau_exact_update

# threshold adaptive and decision rule
from .mtgp_decision import max_std_from_var, should_query_max_std
from .mtgp_decision import max_var, should_query_max_var
from .mtgp_threshold import MTGPGeometricDecayThreshold

# projection utils
from utils.compute_ellipsoids import compute_constrain_sets
from utils.projection_to_ell import projection2ellipsoid

# custom likelihood
from .task_noise_specific_likelihood import SpecificTaskNoiseDerivativeMTGPLikelihood

# model
from admm.moreau_oracle import build_moreau_oracles

class MTGPMoreauGradProvider:
    def __init__(
        self, 
        gamma_list, 
        theta_list, 
        rho: float, 
        zeta: float, 
        min_points: int = 5,
        
        # threshold schedule
        iota: float = 0.7,
        alpha: float = 0.98, # decay
        
        # mtgp training
        train_iter: int = 200,
        lr: float = 0.01,
        num_tasks: int = 10, # num_agent
        num_latents: int = 10, # num_latent process using
        rank: int = 1, #  rank of the mixing matrix for index kernel

        # torch
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,

        # noise options
        specific_task_noise: bool = False,
        
        # local function options (the exact prox solver is fixed inside MoreauOracle)
        local_model: str = "quadratic", # "quadratic", "poly4" or "dir_poly4"
        lambda4: float = 0.05,
        direct_matrix=None
    ):
        self.gamma_list = gamma_list
        self.theta_list = theta_list
        self.rho = float(rho)
        self.zeta = float(zeta) # just incase do not enter the float

        self.N = len(gamma_list)
        self.p = int(theta_list[0].shape[0]) # input dim

        # for mtgp model
        self.min_points = int(min_points)
        self.train_iters = int(train_iter)
        self.lr = float(lr)

        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.rank = rank

        self.models = None
        self.likelihoods = None

        # device
        self.device = device
        self.dtype = dtype

        # noise options
        self.specific_task_noise = specific_task_noise

        #local form
        self.local_model = local_model # function form
        self.lambda4 = lambda4 # regularization for poly4
        self.direct_matrix = direct_matrix
        
        # local solver
        self.oracles = build_moreau_oracles(
                        gamma_list=self.gamma_list,
                        theta_list=self.theta_list,
                        rho=self.rho,
                        local_model=self.local_model,
                        lambda4=self.lambda4,
                        A=self.direct_matrix
                    )

        # threshold module
        self.thresh = MTGPGeometricDecayThreshold(n_agents=self.N, iota=iota, alpha=alpha)

        # per agent buffer
        self.Z = [[] for _ in range(self.N)] # list of z (np.arrays)
        self.Y = [[] for _ in range(self.N)] # list of observation [f, grad, ...]

        #phase state
        self.mtgp_active = False # collect data -> all function have same amoint -> active
        self._trained_once = False # train_all once done

        # count new points _ since last regit when in gp-active mode (just incase not upate each time having new points)

        self._new_points_since_fit = [0] * self.N # store the index of new points that use for quantify when to refit
        self.counting_projection = 0 # counting the number of projection happened

    # ========== April, 7th: exact computation -> getting data for modeling ==========================
    # @staticmethod
    # def quad_value(x: np.ndarray, G: np.ndarray, theta: np.ndarray) -> float:
    #     # compute (x - theta)^T G (x - theta)
    #     d = x  - theta
    #     return float(d.T @ G @ d)

    # def exact_moreau_value_and_grad(self, i: int, z: np.ndarray):
    #     # calculate the exact Moreau grad and Moreau val

    #     # exact grad
    #     g = solving_moreau_exact_update(z, self.gamma_list[i], self.theta_list[i], self.rho)
        
    #     prox = z - (1.0  / self.rho) * g # reconstruct the prox point

    #     # compute the objective at the point
    #     f_val = self.quad_value(prox, self.gamma_list[i], self.theta_list[i])

    #     f_gamma = f_val + 0.5 * self.rho * float(np.dot(prox - z, prox - z)) # just compute the sub-objective

    #     y = np.concatenate([[f_gamma], g.astype(float)]) # observation for each point z_i

    #     return y, g

    # ==============================================================================================
    # April, 7th, 2026: update the exact solver to cope with more complex function

    def exact_moreau_value_and_grad(self, i: int, z: np.ndarray):
        f_gamma, g = self.oracles[i].moreau_value_and_grad(z)
        y = np.concatenate([[f_gamma], g.astype(float)])
        return y, g
        
    #need build proper data to feed the model
    @staticmethod
    def build_stack_dataset(Z_buffer, Y_buffer):

        # transform the collected in np arrays => torch tensors for gpytorch training
        # z_buffer = list of list [z_1, z_2, ... z_n] for each agent
        # y_buffer = list of list [y_1, y_2, ... y_n] for each agent

        Z_list, Y_list = [], []

        for i, (Zi, Yi) in enumerate(zip(Z_buffer, Y_buffer)):
            if len(Zi) == 0:
                continue
            
            # Z_tensor = torch.as_tensor(np.stack(Zi, axis=0), dtype=torch.float32) # (ni,p)
            # Y_tensor = torch.as_tensor(np.stack(Yi, axis=0), dtype=torch.float32) # (ni,p+1)

            # float64 test: float32 here rounds nearby points even when the GP runs in float64
            Z_tensor = torch.as_tensor(np.stack(Zi, axis=0), dtype=torch.float64) # (ni,p)
            Y_tensor = torch.as_tensor(np.stack(Yi, axis=0), dtype=torch.float64) # (ni,p+1)

            Z_list.append(Z_tensor)
            Y_list.append(Y_tensor)

        Z_all = torch.cat(Z_list, dim = 0) # concate all the input data (Z) vertical

        #set the task id for each data point
        task_id = torch.cat([
            torch.full((Z.shape[0],), t) for t, Z in enumerate(Z_list)
        ], dim = 0)

        # combine the task_id and input as each data input will be like (z_i, i)

        Z_aug = torch.cat([
            Z_all, task_id.float().unsqueeze(1)
        ], dim = 1) # shape: (Ntottal, p + 1) as input

        # structure the output data Y
        Y_all = torch.cat(Y_list, dim = 0) # shape (Ntotal, p + 1) as output

        # stretch to (Ntotal * (p + 1), ) for gpytorch training
        Y_aug = Y_all.reshape(-1)

        return Z_aug, Y_aug

    def _fit_or_update(self):

        # structure the data kind of data does not look like that SOGP with derivatives
        # y -> [y1, g1, g2, ... gp] for each point and stack them vertically for each agent
        # z -> z_i for each point and stack them vertically for each agent ((Ntotal, p+1))

        # Build stacked dataset
        Z_aug, Y_aug = self.build_stack_dataset(self.Z, self.Y)

        Z_aug = Z_aug.to(self.device, self.dtype)
        Y_aug = Y_aug.to(self.device, self.dtype)

        ## ===== Fixing bug (March, 6th, 2026): refit -> define model again even already have it -> unstable =====

        if self.models is None:
            # Create likelihood once and make sure it small enough since we already add noise in the model
        
        # ============== March, 31, 2026: Custom llh function =========================
            # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(self.device)
            
            # diff task noise for each task
            if self.specific_task_noise:
                likelihood = SpecificTaskNoiseDerivativeMTGPLikelihood(
                    num_tasks=self.N,
                    num_outputs_per_input=self.p+1,
                    task_feature_index=self.p, # the last feature in the input is the task id
                ).to(self.device, self.dtype)
                
            else:
                # same task noise for all task in mtgp model
                # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
                #                                                      noise_prior=gpytorch.priors.SmoothedBoxPrior(0.15, 1.5, sigma=0.001)).to(self.device)

                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3)).to(self.device, self.dtype)

            # ===== Experiment Feb 4, 26 =====
            
            # need freeze this noise since already add noise in the model // mtgp_derivative_model.py
            # likelihood.noise = 1e-4
            # likelihood.raw_noise.requires_grad_(False)

            # ============================ March, 6th, 2026 ============================
            # Create model once
            model = ExactLCMGrad(
                Z_aug, Y_aug, likelihood,
                num_tasks=self.N,
                input_dim=self.p,
                num_latents=self.num_latents,
                rank=self.rank
            ).to(self.device, self.dtype) # model parameters in the same dtype as the data

            self.models = model
            self.likelihoods = likelihood
        else:
            model = self.models
            likelihood = self.likelihoods

            # set new train iteration for update since using the previous model as initialization 
            # -> just do few steps of training for update

            self.train_iters = 100

            # update training data in-place with last comm
            model.set_train_data(inputs=Z_aug, targets=Y_aug, strict=False)


        # train or update model with new data
        train_mtgp_exact(model, likelihood, Z_aug, Y_aug, lr=self.lr, iters=self.train_iters, verbose=False, specific_task_noise=self.specific_task_noise)
        # =================================================================================================
 

    def _fit_all(self):
        self._fit_or_update()
        self._new_points_since_fit = [0] * self.N # reset count
        self._trained_once = True
        self.mtgp_active = True # after fit -> active MTGP mode

    # ======== Predict gradient via MTGP =============
    @torch.no_grad()
    def _predict_grad(self, i: int, z:np.ndarray):
        # predict the gradient for agent i at point z via the mtgp model
        d = self.p

        model = self.models
        likelihood = self.likelihoods

        model.eval()
        likelihood.eval()

        # build the augmented input
        z_data = torch.tensor(z, device=self.device, dtype=self.dtype).unsqueeze(0) # (1, p)
        
        # task id for input of the kernel function since we use custom kernel to incorporate the task information -> need to specify which task we want to predict
        task_id = torch.tensor([i], device=self.device, dtype=self.dtype).unsqueeze(1) # (1, 1)

        z_aug = torch.cat([z_data, task_id], dim = 1) # (1, p + 1)

        # get the predictive distribution
        with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.max_root_decomposition_size(10000), gpytorch.settings.fast_pred_var(False):
            # pred_dist = likelihood(model(z_aug)) # nosise predictive distribution (y*)
            
            # noise free model using the latent prediction (f*|data) // custom :D
            # pred_dist = model(z_aug, noise_free=True) 

            # latent prediction
            pred_dist = model(z_aug)

            # ========== March, 24, 26 ============================
                # pred_dist = model.latent_forward(z_aug) # get the latent function predictive distribution
            # ================================================

            mean_vec = pred_dist.mean.reshape(-1) # (p + 1, )

            covar_mtrx = pred_dist.covariance_matrix # (p + 1, p + 1)

            # extract the gradient mean and variance
            g_mean = mean_vec[1 : ] # (p, )
            g_var = torch.diag(covar_mtrx)[1 : ] # (p, )

        return g_mean.cpu().numpy(), g_var.cpu().numpy()

    # =================== Phase switch checking ===============================
    # check whether we have enough data to active the mtgp

    def _ready_to_activate_gp(self, query_flag: bool) -> bool:
        if query_flag:
            return False  # still burn-in for data collection
        # all agents have enough samples
        return all(len(self.Z[j]) >= self.min_points for j in range(self.N))
    
    # =================== Main Provider ===============================
    ### ======= Feb, 11, 2026 Correct logic =============

    def __call__(self, i: int, z_i: np.ndarray, query_flag: bool, k: int , *args, **kargs):
        
        # if not self.mtgp_active:
        #     y, g = self.exact_moreau_value_and_grad(i, z_i)
        #     self.Z[i].append(z_i.copy()) # build data
        #     self.Y[i].append(y.copy())

        #     # after burn in -> check if enough data -> train first time

        #     if (i==0) and (not self._trained_once) and self._ready_to_activate_gp(query_flag):
        #         self._fit_all() # first model fit
        #         print("GP Activated")

        #     return g, True, {}

        if not self.mtgp_active and (not self._ready_to_activate_gp(query_flag)):
            print(f"Collecting data for agent {i+1} at iter {k+1}, current samples: {len(self.Z[i])}")
            y, g = self.exact_moreau_value_and_grad(i, z_i)
            self.Z[i].append(z_i.copy()) # build data
            self.Y[i].append(y.copy())
            
            # gradient error
            grad_err = 0.0 # zero since we do exact query here
            return g, True, {}, grad_err

            # after burn in -> check if enough data -> train first time

        if (not self._trained_once) and self._ready_to_activate_gp(query_flag):
            self._fit_all() # first model fit
            print("GP Activated at iter", k+1)
        
        # check the previous iteration end -> refit if have new data
        if (i==0) and self.mtgp_active and (sum(self._new_points_since_fit) > 0):
            # end of iteration -> check do we have new data -> yes -> refit with new data
            self._fit_or_update()
            print("Refit MTGP with new data")

        # ========== Fixing this bug March, 5, 26 ============================
            self._new_points_since_fit = [0] * self.N # reset count
        # ====================================================================

        # using mtgp for predict the gradient
        # prediction
        print(f"Predicting with MTGP for agent {i+1} at iter {k+1}")

        g_mean, g_var = self._predict_grad(i, z_i)


        # hidden exact gradient, used ONLY to log the prediction error (diagnostic)
        # it must not affect the query decision, the GP data, or the ADMM update
        _, g_true = self.exact_moreau_value_and_grad(i, z_i)

        # grad error BEFORE projection: raw GP prediction quality, so SOGP and MTGP
        # are compared on the model alone (same definition as the SOGP provider)
        grad_err = float(np.linalg.norm(g_true - g_mean))
        print(f"MTGP - Grad error before projection for agent {i+1} at iter {k+1}: {grad_err:.4f}")

        # ========== Projection to the ellipsoid constraint (Jan, 28 - Long Vu) ===================
        A, B, c = compute_constrain_sets(z_i, self.Z[i], self.Y[i], gamma=1.0/(self.rho), max_num_ellipsoids=5)
        g_mean, proj_stats = projection2ellipsoid(g_mean, A, B, c)

        if proj_stats:
            self.counting_projection += 1
            print(f"MTGP - Projection at iter {k+1} for agent {i+1}")
            print(f"MTGP - Current total Projection times: {self.counting_projection}")

        # ========== end projection ======================================

        # Query decision
        # ======== May change it to varinace too more align with prev paper (Jan, 14 ====== 

        # std_max = max_std_from_var(g_var) 

        # max variance
        var_max = max_var(g_var)
        
        # ========================================================================

        # init threshold at first MTGP decision for the agent
        self.thresh.ensure_initialized(i=i, k=k, V_k0=var_max) # set initial threshold

        # update the psi after each iteration
        psi_k = self.thresh.value(i=i, k=k)

        # decision for each agent
        # do_query = bool(query_flag) or should_query_max_std(std_max, psi_k)

        do_query = should_query_max_var(var_max, psi_k)

        # stats data for analysis
        stats = {"g_var": g_var, "var_max": var_max, "psi_k": psi_k, "do_query": do_query}

        # print(f"Stats of agent{i+1} at iter {k+1}\n", stats)

        if do_query:
            y, g = self.exact_moreau_value_and_grad(i, z_i)
            self.Z[i].append(z_i.copy())
            self.Y[i].append(y.copy())
            print(f"MTGP - Query for exact solution {k+1} for agent {i+1}")

            # refit if at the iteration k we have new points (the mtgp model)
            # check entire iteration to see whether they have new points -> yes -> refit & reset count
            # check when we consider the last agent query in the iteration
            self._new_points_since_fit[i] += 1

            # query so grad error also zero 
            grad_err = 0.0
            return g, True, stats, grad_err

        # prediction is used: no communication, return the projected MTGP gradient
        return g_mean, False, stats, grad_err