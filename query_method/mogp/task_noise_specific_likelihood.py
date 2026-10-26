import torch
import gpytorch
from torch import nn
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.likelihood import Likelihood
from linear_operator.operators import DiagLinearOperator, LinearOperator
from gpytorch.constraints import Positive, GreaterThan, Interval

from .custom_noise_model import MultitaskDerivativeNoiseModel
import numpy as np


class SpecificTaskNoiseDerivativeMTGPLikelihood(_GaussianLikelihoodBase):
    ## Long Vu: April, 1, 2026
    ## specific task noise for the derivative MTGP model (each task has its own noise level
    ## for general cases (1 noise for all) -> Use the general Gpytorch.GaussianLikelihood

    def __init__(self, 
                 num_tasks: int,
                 num_outputs_per_input: int,
                 task_feature_index: int = -1,
                 noise_constraint=None,
                 **kwargs):
        
        # self.has_analytic_marginal = True

        # definee the noise model for this specific likelihood
        noise_covar = MultitaskDerivativeNoiseModel(
            num_tasks = num_tasks, 
            num_outputs_per_input=num_outputs_per_input, 
            task_feature_index=task_feature_index, 
            noise_constraint=noise_constraint
        )
        
        super().__init__(noise_covar=noise_covar)

    # getter will transform to get the positive value -> satisfy the length, noise, scale
    # optimized params is the raw_noise but when using we use need transform it to get the positive value
    @property
    def noise(self):
        return self.noise_covar.noise

    # assign method for the model noise
    @noise.setter
    def noise(self, value):
        self.noise_covar.initialize(value)

    @property
    def raw_noise(self):
        return self.noise_covar.raw_noise
    
    ## set noise -> set the positive -> inv to get neg -> assign to the raw noise (the optimized param) 
    # -> when optimize we optimize the raw noise but when use we use the noise (positive value)

    # forward to get the P(y|f)
    def forward(self, function_samples, *params, **kwargs):
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        return base_distributions.Normal(function_samples, noise.sqrt())
    ## marginal func already inherit fromt the _GaussianLikelihoodBase -> use the noise_covar to get the cov matrix of the noise and add to the cov of the function samples to get the marginal distribution

    def marginal(self, function_dist: MultivariateNormal, *params, **kwargs):
        mean = function_dist.mean ## mean of p(f|x)
        covar = function_dist.lazy_covariance_matrix ## cof of p(f)

        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)

        full_covar = covar + noise_covar ## cov of p(y|x) = cov of p(f|x) + cov of noise

        return function_dist.__class__(mean, full_covar)
        # pass