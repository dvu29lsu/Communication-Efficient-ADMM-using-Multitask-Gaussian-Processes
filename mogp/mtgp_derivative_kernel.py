import gpytorch
import torch
from gpytorch.constraints import Interval
## custom kernel for the MTGP with derivative information

# kernel(x,c,i),(x',c',i') => k_(i,i')*k({x,c},{x',c'}). Here the k(x,c,x',c') -> grad kernel
# c: channel (partial derivative)
# i: task id
# x,x' input

# noise model is i.i.d (task noise -> both value and partial derivative but iid)

class LMCGradKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_tasks: int, input_dim: int, num_latents: int, rank: int):
        super().__init__() # just inherite the Kernel base
        self.d = input_dim # input dimension
        self.Q = num_latents # number of the latent processes

        # inter task information sharing kernel
        self.task_kernels = torch.nn.ModuleList([
            gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
            for _ in range(self.Q)
        ])
        
        # each latent is a GP kernel with derivative info
        # self.latent_grad_kernels = torch.nn.ModuleList([
        #     gpytorch.kernels.RBFKernelGrad(ard_num_dims=input_dim,
        #                                    lengthscale_constraint=Interval(1e-4, 10.0)) # lengthscale constraint to avoid too small lengthscale that can cause numerical issue
        #     for _ in range(self.Q)
        # ])
        self.latent_grad_kernels = torch.nn.ModuleList([
            gpytorch.kernels.RBFKernelGrad(ard_num_dims=input_dim) # lengthscale constraint to avoid too small lengthscale that can cause numerical issue
            for _ in range(self.Q)
        ])
    
    # each point point correlation return a d + 1 covariance matrix (include value and partial derivative)
    def num_outputs_per_input(self, x1, x2):
        return self.d + 1

    # forward when calling the kernel
    def forward(self, x1, x2, **params):
        d = self.d # input dim

        x1_data = x1[:, :d]
        x2_data = x2[:, :d]

        t1 = x1[:, d].long() # getting the task id of each sample
        t2 = x2[:, d].long() # just incase the task not the integet

        K = None # big K in clude the inter task and intra task covariance

        for q in range(self.Q):

            # N1*(d+1), N2*(d+1)
            # Kx = self.latent_grad_kernels[q](x1_data, x2_data, **params).to_dense() #get full tensors

            # March, 25, 2026 just test for better memory and stability algorithm
            Kx = self.latent_grad_kernels[q](x1_data, x2_data, **params)

            # task kernel

            Kt = self.task_kernels[q](t1, t2).to_dense() # need go dense since the pytorhc -> return the lazy tensor (here we want to get full tensor)

            # extrapolate => expand to big task kernel (repeat each tensor elemense d+1 times throught the vertical and also the horizontal axis)
            Kt_big = Kt.repeat_interleave(d+1, dim = 0).repeat_interleave(d+1, dim = 1)

            # multiply them together the Kx and Kt

            Kfull = Kx * Kt_big

            K = Kfull if K is None else (K + Kfull) 

        return K