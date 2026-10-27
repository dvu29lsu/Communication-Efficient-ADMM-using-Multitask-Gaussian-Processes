import numpy as np
import torch
import gpytorch

from gp.sogp_derivative_model import GPModelWithDerivatives
from gp.train import train_gp_exact
from admm.exact_quadsolver import solving_moreau_exact_update

from .threshold import GeometricDecayThreshold

from .decision import max_std_from_var, should_query_max_std
from .decision import max_var, should_query_max_var

from utils.compute_ellipsoids import compute_constrain_sets
from utils.projection_to_ell import projection2ellipsoid

from admm.moreau_oracle import build_moreau_oracles

class SOGPMoreauProvider:
    
    # Colect data -> behave like normal ADMMs
    #  communicate -> get exact update
    #  collect (z, f_gamma(z), grad f_gamma(z)) for all agent

    # Swtich for training GP once all agents have >= min_points samples
    #  using burn-in is over with querying flag turn False
    #  all agents have >= min_points samples

    # GPmode 
    #  each agent, predict grad mean/var and gate with threshold
    #  if query exact -> check with ellipsoid -> okay -> use pred gradient -> update admm (skip comm)
    #  if pred grad uncertainty too hight -> comm for exact -> update GP

    def __init__(
        self,
        gamma_list,
        theta_list,
        rho: float,
        zeta: float,
        min_points: int = 5,

        # threshold schedule
        iota: float = 0.6,
        alpha: float = 0.97,

        # GP training
        train_iters: int = 50,
        lr: float = 0.01,

        # torch
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,

        # diagnostic pred error
        eval_exact_error: bool = True,

        # refit frequency after exact queries in GP mode
        retrain_every: int = 1,  # 1 = refit immediately each time agent queried

        # local function options (the exact prox solver is fixed inside MoreauOracle)
        local_model: str = "quadratic", # "quadratic", "poly4" or "dir_poly4"
        lambda4: float = 0.5,
        direct_matrix: np.ndarray = None
    ):
        self.gamma_list = gamma_list
        self.theta_list = theta_list
        self.rho = float(rho)
        self.zeta = float(zeta)

        self.N = len(gamma_list)
        self.p = int(theta_list[0].shape[0])

        self.min_points = int(min_points)
        self.train_iters = int(train_iters)
        self.lr = float(lr)

        self.device = device
        self.dtype = dtype

        self.eval_exact_error = bool(eval_exact_error) # whether evaluate the gradient prediction with te exact grad from query

        self.retrain_every = int(retrain_every)
        assert self.retrain_every >= 1

        # threshold module
        self.thresh = GeometricDecayThreshold(self.N, iota=iota, alpha=alpha)

        # per-agent buffers
        self.Z = [[] for _ in range(self.N)]  # list of z (np arrays)
        self.Y = [[] for _ in range(self.N)]  # list of y = (p+1) arrays [f_gamma, grad...]

        # per-agent GP models
        self.models = [None] * self.N
        self.likelihoods = [None] * self.N

        # phase state
        self.gp_active = False     # Phase 0 until we activate GP
        self._trained_once = False # Train-all-once done?

        # count new points since last refit (per agent) in GP-active mode
        self._new_points_since_fit = [0] * self.N
        self.counting_projection = 0

        # local form
        self.local_model = local_model # function form
        self.lambda4 = lambda4 # regularization for poly4
        self.direct_matrix = direct_matrix

        # local function for each agents -> provide the exact solution
        self.oracles = build_moreau_oracles(
                gamma_list=self.gamma_list,
                theta_list=self.theta_list,
                rho=self.rho,
                local_model=self.local_model,
                lambda4=self.lambda4,
                A=self.direct_matrix
            )

    # ---------- exact computation (communication) ----------
    # @staticmethod
    # def quad_value(x: np.ndarray, G: np.ndarray, theta: np.ndarray) -> float:
    #     d = x - theta
    #     return float(d.T @ G @ d)

    # def exact_moreau_value_and_grad(self, i: int, z: np.ndarray):

    #     # exact Moreau gradient and envelope value via prox identity

    #     g = solving_moreau_exact_update(z, self.gamma_list[i], self.theta_list[i], self.rho)

    #     prox = z - (1.0 / self.rho) * g # reconstruc the prox point
        
    #     f_val = self.quad_value(prox, self.gamma_list[i], self.theta_list[i])
        
    #     f_gamma = f_val + 0.5 * self.rho * float(np.dot(prox - z, prox - z))
        
    #     y = np.concatenate([[f_gamma], g.astype(float)])
        
    #     return y, g

    # April, 8th, 26: changing to use the orable for exact solutios -> adapt with diff local function form
    def exact_moreau_value_and_grad(self, i: int, z: np.ndarray):
        f_gamma, g = self.oracles[i].moreau_value_and_grad(z)
        y = np.concatenate([[f_gamma], g.astype(float)])
        return y, g

    # ---------- GP fit/predict ----------
    def _fit_or_update(self, i: int):
        train_x = torch.tensor(np.stack(self.Z[i], axis=0), device=self.device, dtype=self.dtype)  # (n,p)
        train_y = torch.tensor(np.stack(self.Y[i], axis=0), device=self.device, dtype=self.dtype)  # (n,p+1)

        if self.models[i] is None:
            # Multitask likelihood already include the noise -> no need adding noise
            # DEFAULT: Task Noise as well as Global Noise)
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.p + 1).to(self.device, self.dtype) ## using the RBFKernelsGrads -> as MTGP they sharing the same input for grad, value

            # ================ changing task noise (each task -> only global noise, not specific noise varriance for the Moreau and grad)
            # (Feb 9, 26) ==========================
            # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=(self.p + 1), has_task_noise=False).to(self.device) ## using the RBFKernelsGrads -> as MTGP they sharing the same input for grad, value
            # ================ end changing task noise ==========================

            # ========= freeze noise (Jan 13 26) ==========
            # freeze noise for now (noise free model)
            # tiny = torch.tensor(1e-4, device=self.device, dtype=self.dtype)

            # if hasattr(likelihood, "raw_noise"):
            #     likelihood.noise = tiny
            #     likelihood.raw_noise.requires_grad_(False)

            # if hasattr(likelihood, "raw_task_noises"):
            #     likelihood.task_noises = tiny.expand(self.p + 1)
            #     likelihood.raw_task_noises.requires_grad_(False)

            # ============================================
            
            model = GPModelWithDerivatives(train_x, train_y, likelihood, p=self.p).to(self.device, self.dtype) # model parameters in the same dtype as the data

            self.models[i] = model
            
            self.likelihoods[i] = likelihood
        else:
            model = self.models[i]
            likelihood = self.likelihoods[i]
            model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        train_gp_exact(model, likelihood, train_x, train_y, lr=self.lr, iters=self.train_iters, verbose=False)

    # each iteration fit one time (if they have new value)
    def _fit_all_agents_once(self):
        for i in range(self.N):
            self._fit_or_update(i)
            self._new_points_since_fit[i] = 0
        self._trained_once = True # train
        self.gp_active = True # after fit GP -> active GP mode

    @torch.no_grad()
    def _predict_grad(self, i: int, z: np.ndarray):
        model = self.models[i]
        likelihood = self.likelihoods[i]
        model.eval(); likelihood.eval() ## switching mode -> eval

        test_x = torch.tensor(z[None, :], device=self.device, dtype=self.dtype)

        ## March, 4th, 2026 => Error: changing the prediction result to use noise free covariance and exact latent mean prediction

        with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.max_root_decomposition_size(10000), gpytorch.settings.fast_pred_var(False):
            # pred = likelihood(model(test_x)) ## prediction as the noisy prediction distribution (y*)
            pred = model(test_x)  # direct latent function prediction -> noise free covariance 

            mean = pred.mean.squeeze(0).detach().cpu().numpy()    # => (p+1,)
            var = pred.variance.squeeze(0).detach().cpu().numpy()   # => (p+1,)
        return mean[1:], var[1:]  # gradient mean/var

    # ---------- phase switch check ----------
    def _ready_to_activate_gp(self, query_flag: bool) -> bool:
        if query_flag:
            return False  # still burn-in
        # all agents have enough samples 
        return all(len(self.Z[j]) >= self.min_points for j in range(self.N))

    # ---------- main callable ----------
    def __call__(self, i: int, z_i: np.ndarray, query_flag: bool, k: int):

        # if still burnin -> return the true value of the gradient/ moreau envelope value
        # collect data burn in
        
        if not self.gp_active and (not self._ready_to_activate_gp(query_flag)): 
            # gp not active here we query to get the data to build the GP model
            # always communicate
            print(f"Collecting data for agent {i+1} at iter {k+1}")
            y, g = self.exact_moreau_value_and_grad(i, z_i)
            self.Z[i].append(z_i.copy())
            self.Y[i].append(y.copy())

            grad_err = 0.0 # since here we do query so the error is 0.0 here
            return g, True, {}, grad_err

            # After burn-in ends and all agents have enough data points, train all.
            # trigger training only once per iteration by checking i==0.
            # if (i == 0) and (not self._trained_once) and self._ready_to_activate_gp(query_flag):
            #     self._fit_all_agents_once()
                        # trigger training only once per iteration by checking i==0.
            
            # ====== Feb, 10, 26 changing the logic of the starting training GP ======
        if (not self._trained_once) and self._ready_to_activate_gp(query_flag):
            self._fit_all_agents_once()
            print("GP Activated at iter", k+1)
            # ========================================
        
        # else:
        #     pass 
    
        # # already in GP mode, go to the GP prediction and decision logic below
        # using gp for predict gradient
        print(f"Predicting with SOGP for agent {i+1} at iter {k+1}")
        g_mean, g_var = self._predict_grad(i, z_i)

        # hidden exact gradient, used ONLY to log the prediction error (diagnostic)
        # it must not affect the query decision, the GP data, or the ADMM update
        _, g_true = self.exact_moreau_value_and_grad(i, z_i)

        # grad error BEFORE projection: raw GP prediction quality, so SOGP and MTGP
        # are compared on the model alone (same definition as the MTGP provider)
        grad_err = float(np.linalg.norm(g_true - g_mean))
        print(f"GP - Grad error before projection for agent {i+1} at iter {k+1}: {grad_err:.4f}")

    # ========== Projection onto the ellipsoids (Jan, 28, 26) =========================
        
        #define ellipsoids constrains from the past points
        A, B, c = compute_constrain_sets(z_i, self.Z[i], self.Y[i], gamma=1.0/(self.rho), max_num_ellipsoids=5)

        #checking and project
        g_mean, proj_stats = projection2ellipsoid(g_mean, A, B, c)

        if proj_stats:
            self.counting_projection += 1
            print(f"GP - Projection count at iter {k+1} for agent {i+1}")
            print(f"GP - Current total Projection times: {self.counting_projection}")

        # ========== end projection ======================================

        # Query decision based on max_var or max_std

        # =============== Start decision here ========
        # var_max = max_std_from_var(g_var)
        var_max = max_var(g_var)

        # init threshold at first GP decision for this agent
        self.thresh.ensure_initialized(i=i, k=k, V_k0=var_max) # set the initial threshold

        # update the psi after each iteration
        psi_k = self.thresh.value(i=i, k=k)

        # print("Agent:", i, "Iter:", k, "Var max:", var_max, "Psi_k:", psi_k)

        # decision
        # do_query = bool(query_flag) or should_query_max_std(std_max, psi_k)

        # ======= Feb, 9, 26: fixing the redundant checking query ======
        
        # do_query = bool(query_flag) or should_query_max_var(var_max, psi_k)

        do_query = should_query_max_var(var_max, psi_k)

        # gettingthe stats for analysis (Feb, 9, 26)
        stats = {"g_var": g_var, "var_max": var_max, "psi_k": psi_k, "do_query": do_query}
        
        # print(stats) ## just for debug

        if do_query:
            y, g = self.exact_moreau_value_and_grad(i, z_i)
            self.Z[i].append(z_i.copy())
            self.Y[i].append(y.copy())
            print(f"GP - Query for exact solution {k+1} for agent {i+1}")

            # refit only this agent, optionally every few new points
            self._new_points_since_fit[i] += 1
            if self._new_points_since_fit[i] == self.retrain_every:
                self._fit_or_update(i) # update if have new point
                print(f"Refit GP {i+1} with new query data")
                self._new_points_since_fit[i] = 0 # back to base value

            # grad error
            grad_err = 0.0 # since here we do query adn we need return 0 here
            return g, True, stats, grad_err

        # prediction is used: no communication, return the projected GP gradient
        return g_mean, False, stats, grad_err