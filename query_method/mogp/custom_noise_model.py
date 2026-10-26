import torch
import gpytorch
import numpy as np

from gpytorch.likelihoods.noise_models import Noise
from linear_operator.operators import DiagLinearOperator
from gpytorch.constraints import Positive, GreaterThan, Interval

# Long Vu: April, 1st, 2026
# define the custom noise module for the specific task noise likelihood function

class MultitaskDerivativeNoiseModel(Noise):
    def __init__(self, 
                num_tasks: int,
                num_outputs_per_input: int,
                task_feature_index: int = -1,
                noise_constraint=None,
                **kwargs):
        
        # just init the noise model
        super().__init__()

        ## just define some params for the noise model

        self.num_tasks = num_tasks

        self.num_outputs_per_input = num_outputs_per_input # num out / 1 input

        self.task_feature_index = task_feature_index #index of the task feature in the input

        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        ## register the noise params and the noise_constraints to the model

        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(num_tasks)))

        ## register the noise constraint to the model (as the module of gpytorch)
        self.register_constraint("raw_noise", noise_constraint)

# define the getter -> transform the optimized params -> positve with transform as the softplus function
    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

# define the setter -> assign value for the noise val
    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=self.raw_noise.dtype, device=self.raw_noise.device) ## tensor trans

        # init the raw noise with the inverse transform of the value (get the raw noise that can be optimized)

        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))


    def forward(self, *params, shape=None, **kargs):
        # shape of the input will be taken from the *params[0]

        if len(params) == 0:
            raise ValueError("need the x_aug to modeling the task specific noise for the derivative MTGP model")

        # print("params types", type(params[0]))

        if isinstance(params[0], list) or isinstance(params[0], tuple):
            x_aug = params[0][0]

        else:
            x_aug = params[0]
        
        task_ids = x_aug[:, self.task_feature_index].long() ## task index and stretch (n,)
        
        block = self.num_outputs_per_input # number output per input -> create a block wise

        n = task_ids.numel() # num data points 

        # get the expected shape of the noise covar (n*block, n*block)

        expected_shape = n*block

        if shape is not None and shape[-1] != expected_shape:
            raise ValueError("noise shape mismatch")
        
        point_noises = self.noise[task_ids] # (n,) -> match the task id with the noise level

        # interleave the noise for eeach data -> block wise

        interleaved_noise = point_noises.repeat_interleave(block) # (n*block,)

        return DiagLinearOperator(interleaved_noise)
