import torch
import gpytorch
from torch import nn
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.likelihood import Likelihood

from linear_operator.operators import DiagLinearOperator, LinearOperator
from gpytorch.constraints import Positive, GreaterThan, Interval
import numpy as np


# class SpecificTaskNoiseDerivativeMTGPLikelihood(_GaussianLikelihoodBase):
class SpecificTaskNoiseDerivativeMTGPLikelihood(gpytorch.likelihoods.Likelihood):
    ## Long Vu: March, 21, 2026 (Not work)
    ## specific task noise for the derivative MTGP model (each task has its own noise level
    ## for general cases (1 noise for all) -> Use the general Gpytorch.GaussianLikelihood

    def __init__(self, 
                 num_tasks: int,
                 num_outputs_per_input: int,
                 task_feature_index: int = -1,
                 noise_constraint=None,
                 **kwargs):
        
        self.has_analytic_marginal = True
        
        super().__init__()

        ## need define the noise covar => get the cov matrix of the noise (diagonal matrix)

        ## define the noises as parameter need to tune 

        self.num_tasks = num_tasks
        self.num_outputs_per_input = num_outputs_per_input
        self.task_feature_index = task_feature_index

        # just define the constraint for the noise level
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        # register 
        # define noise params for learning
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(num_tasks)))

        ## register the constraints to the model (module gpytorch) => 
        self.register_constraint("raw_noise", noise_constraint)

    # getter will transform to get the positive value -> satisfy the length, noise, scale
    # optimized params is the raw_noise but when using we use need transform it to get the positive value
    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    # assign method for the model noise
    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    ## set noise -> set the positive -> inv to get neg -> assign to the raw noise (the optimized param) 
    # -> when optimize we optimize the raw noise but when use we use the noise (positive value)

    def _set_noise(self, value):
        if torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=self.raw_noise.dtype, device=self.raw_noise.device) 
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):

        # shape of the input will be taken from he *params shape
        if len(params) == 0:
            raise ValueError("need feed the x_aug to modeling the task specfic noise for the derivative mtgp")
        
        x_aug = params[0]
        if x_aug.dim() != 2:
            raise ValueError("should have the shape (n,p+1) where p is the input dim")

        task_ids = x_aug[:, self.task_feature_index].long() ## (n,)

        n = task_ids.numel() ## num of data points
        block = self.num_outputs_per_input # num output per input 

        # expected shape of the noise covar (n*blokc, n*block)
        expected = n * block

        if base_shape[-1] != expected:
            raise ValueError("noise shape missmatch with the covar shape") 

        # self.noise: (num_tasks,) -> map the task ids to the corresponding noise level for each data point 
        # (n,) -> then repeat interleave
        
        point_noises = self.noise[task_ids] ## (n,) -> then iterleave

        # interleave the noise for each output per input
        interleaved_noise = point_noises.repeat_interleave(block) # (n*(p+1),) # output have func val + partial deriv

        return DiagLinearOperator(interleaved_noise)

    # overide the method to get the noise covar for the likelihood p(y|f)
    def forward(self, function_samples, *params, **kwargs):
        # function sample here is the prior from the model 
        # -> need to add the noise covar to get the likelihood distribution p(y|f) for training and evaluation

        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)

        return base_distributions.Normal(function_samples, noise.sqrt())

    # handling the marginal llh p(y|x)
    def marginal(self, function_distributions: MultivariateNormal, *params, **kwargs):
        mean = function_distributions.mean
        covar = function_distributions.lazy_covariance_matrix

        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)

        # full covar with noisy for training + eval if need to compute predictive distribution p(y*|x*) for test data
        full_covar = covar + noise_covar

        return function_distributions.__class__(mean, full_covar)



