import gpytorch

# define the GPmodle with derivative using exact GP for each agents
class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, p: int):
        super().__init__(train_x, train_y, likelihood)
        self.p = int(p)

        self.mean_module = gpytorch.means.ConstantMeanGrad()

        # freeze params here for using the mean function as 0 for the GP with derivative model
        self.mean_module.initialize(constant=0.0)
        # self.mean_module.constant_requires_grad = False
        self.mean_module.constant.requires_grad_(False)

        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=self.p)
    
        #scale kernel with sigma_p as the formular of the original RBF kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
