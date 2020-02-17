import numpy as np
import torch
import gpytorch
import time



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 2))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    

    
    
    
    
def train_GP_model(model, likelihood, train_x, train_y, training_iter = 150):


    model.train()
    likelihood.train()


    optimizer = torch.optim.Adam([
        {'params': model.parameters()}, ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
    return model, likelihood


start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_x = torch.load("/home/yz2547/BoTorch_Tests/GP_fitting/train_x.pt")
train_y = torch.load("/home/yz2547/BoTorch_Tests/GP_fitting/train_y.pt")
train_x = train_x.to(device)
train_y = train_y.to(device)



for i in range(3, 5):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x[:i,:], train_y[:i], likelihood)
    model = model.to(device)
    likelihood = likelihood.to(device)
    model, likelihood = train_GP_model(model, likelihood, train_x[:i,:], train_y[:i], 100)
    print(i - 3, ' th iterations time is ', time.time() - start_time)
    

