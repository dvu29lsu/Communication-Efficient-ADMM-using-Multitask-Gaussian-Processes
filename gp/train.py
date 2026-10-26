import torch
import gpytorch

def train_gp_exact(model, likelihood, train_x, train_y, lr, iters, verbose=False):
    #switch mode
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # set the max cholesky size to avoid the error of decomposition fail when the size of training data large -> switch btw the CG and cholesky method
    with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.max_root_decomposition_size(10000):
        for t in range(iters):
            optimizer.zero_grad()
            output = model(train_x) 
            loss = -mll(output, train_y)

            loss.backward()
            optimizer.step()

            # if (t == 5):
            #     print(f"GP train iter {t}: neg log likelihood = {loss.item():.4f}")