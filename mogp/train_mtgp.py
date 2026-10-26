import torch
import gpytorch
import torch.nn.functional as F
from linear_operator.operators import DiagLinearOperator

def train_mtgp_exact(model, likelihood, train_x, train_y, lr, iters, verbose=False, specific_task_noise=False):
    # switch to train mode
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)



    with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.max_root_decomposition_size(10000):  
        for t in range(iters):
            optimizer.zero_grad()
            output = model(train_x) # function distribution

            # March, 31, 2026 try custom llh function    
            # need to feed the train_x to make the custom llh can read the task id

            if specific_task_noise:
                loss = -mll(output, train_y, train_x)
            else:
                loss = -mll(output, train_y) 

            loss.backward()
            optimizer.step()
        # print(f"train iter {t}")