## define the model for the mtgp derivative information 

import gpytorch
import torch
import torch.nn.functional as F
from gpytorch.constraints import Interval, Positive
from linear_operator.operators import DiagLinearOperator
from .mtgp_derivative_kernel import LMCGradKernel

# inherite the exact model

class ExactLCMGrad(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, input_dim, num_latents, rank, use_scale_kernel = True):
        super().__init__(train_x, train_y, likelihood)

        self.d = input_dim

        self.K = num_tasks

        self.mean_module = gpytorch.means.ConstantMeanGrad() #constant mean grad
        # need to set is as 0 for 0 mean kernel & frozen it
        self.mean_module.initialize(constant=0.0)
        # self.mean_module.constant_requires_grad = False
        self.mean_module.constant.requires_grad_(False)

        # ======= March, 24, 2026: fixing noise, likelihood, scale kernel as options ==============

        base_kernel = LMCGradKernel(num_tasks, input_dim, num_latents, rank)

        if use_scale_kernel:
            # self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_constraint=Interval(1e-3, 10.0))
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        else:
            self.covar_module = base_kernel
        
        ### ======= Scale kernel option (March, 24, 2026) ============================
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     LMCGradKernel(num_tasks, input_dim, num_latents, rank)
        # )

        ### ==============================================

        # ================= Not Scale Jan, 13=========================================
        # self.covar_module = LMCGradKernel(num_tasks, input_dim, num_latents, rank)
        # actually do not need scale since we alreayd scale by the index kernel => not overparamterizing

        # =================== Val noise, Deriv noise diff ===================== 
        # each task will have 2 noises: val noise + grad noise (K tasks => 2K noise variances)
        # self.raw_val_noise = torch.nn.Parameter(torch.zeros(num_tasks))
        # self.raw_grad_noise = torch.nn.Parameter(torch.zeros(num_tasks))

        # =================== iid noies ======================================
        
        # define noise for each task they need to be i.i.d 
        # self.raw_task_noise = torch.nn.Parameter(torch.full((num_tasks,), 0.1)) # initial noise as soft_plus(-4) with small 
        self.raw_task_noise = torch.nn.Parameter(torch.zeros(num_tasks))
        self.noise_floor = 1e-4 # noise floor
        # self.noise_ceiling = 1.0 # noise ceiling 

        # init params
        # self.init_hyperparams()
    
    def task_noise(self, task_idx):
        sig = torch.sigmoid(self.raw_task_noise[task_idx])
        
        return self.noise_floor + sig * (self.noise_ceiling - self.noise_floor) 
  
                 
    def forward(self, x_aug, noise_free=False):

        d = self.d

        x_data = x_aug[:, :d]

        t = x_aug[:, d].long() # N

        mean_vec = self.mean_module(x_data).reshape(-1) #flat it (N*(d+1),)

        covar_mtrx = self.covar_module(x_aug, x_aug) #(N(d+1) x N(d+1))

        # ======================= Val noise _ Deriv noise diff (Jan 13) ===========================
        # perpoint variance
        # val_var = F.softplus(self.raw_val_noise[t]) + self.noise_floor #
        # grad_var = F.softplus(self.raw_grad_noise[t]) + self.noise_floor #

        # expand to value and also gradient of each point for each task
        # noise_mat = torch.cat([val_var[:, None], grad_var[:, None].repeat(1,d)],  dim=1)

        # noise_vec = noise_mat.reshape(-1)

        # cov_noise = covar_mtrx + DiagLinearOperator(noise_vec)
        # =================================================================================
        

        ## March, 25, 26 shutdown the noise in the model using the noise from LLH
        # ======================== IID noise ==============================================
        # base_var = F.softplus(self.raw_task_noise[t]) + self.noise_floor 
        # # each point -> repeat by d + 1 (val + d grads)

        # # ### March, 24, 2026: change to sigmoid to have more stable noise level and 
        # # also have upper bound on the noise level to avoid too much noise that can cause numerical issue

        # # base_var = self.task_noise(t)

        # noise_mat = base_var[:, None].repeat(1, d+1)
        # noise_vec = noise_mat.reshape(-1)

        # # cov noise
        # cov_noise = covar_mtrx + DiagLinearOperator(noise_vec)

        # # Note: already adding the noise, need freeze the noise built in the Likelihood function of gpytorch
        # if noise_free:
        #     # print("pred noise free")
        #     return gpytorch.distributions.MultivariateNormal(mean_vec, covar_mtrx)
        
        # return gpytorch.distributions.MultivariateNormal(mean_vec, cov_noise)
        # =======================================================================================
        return gpytorch.distributions.MultivariateNormal(mean_vec, covar_mtrx)