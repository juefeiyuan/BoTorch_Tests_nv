import torch
import numpy as np
import math
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.fit import fit_gpytorch_torch
import time

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.get_device_name(device))
train_x = torch.load("/home/yz2547/BoTorch_Tests/GP_fitting/train_x.pt")
train_y = torch.load("/home/yz2547/BoTorch_Tests/GP_fitting/train_y.pt").unsqueeze(-1)
train_x = train_x.to(device)
train_y = train_y.to(device)


for i in range(3, 5):
    
    model = SingleTaskGP(train_X=train_x[:i,:], train_Y=train_y[:i,:])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.to(device)
    mll = mll.to(device)
    fit_gpytorch_model(mll, fit_gpytorch_torch)
    print(i - 3, ' th iterations time is ', time.time() - start_time)    
 