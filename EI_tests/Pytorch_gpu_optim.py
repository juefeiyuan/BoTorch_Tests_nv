import torch
import numpy as np
import math
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_torch
import time 


start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
train_x = torch.tensor([[-2.018476606457733, 10.28593793628148], [0.8510493167919746, 6.888642195088246], [9.881155137925708, 1.8820277741834306]],\
                      dtype=dtype, device=device)
branin = Branin()
train_obj = branin(train_x).unsqueeze(-1)

for i in range(2):
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.to(device)
    mll = mll.to(device)
    fit_gpytorch_model(mll, fit_gpytorch_torch)
    best_value = train_obj.min()
    EI = ExpectedImprovement(model=model, best_f=best_value, maximize = False)
    new_point_analytic, _ = optimize_acqf(
        acq_function=EI,
        bounds=torch.tensor([[-5.0, 0.0] , [10.0, 15.0]], dtype=dtype, device=device),
        q=1,
        num_restarts=1000,
        raw_samples=1000,
        options={},
    )
    train_x =  torch.cat((train_x, new_point_analytic), 0)
    train_obj = torch.cat((train_obj, branin(new_point_analytic).unsqueeze(-1)), 0)
    print(i, ' th iterations time is ', time.time() - start_time)
