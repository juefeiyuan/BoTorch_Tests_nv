import os
import tempfile
import copy
import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.stats    as sps
import scipy.optimize as spo
import multiprocessing
import ast
import numpy as np
import numpy.random as npr
import numpy as np
import random
import warnings
from torch.distributions import Normal
import time


import math
import scipy.linalg as spla
import scipy.optimize as spo
import scipy.stats as sps
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import functools
import sys






def target(X):
    x1 = X[0]
    x2 = X[1]
    
    result = (x2- (5.1/(4*math.pi*math.pi))*x1*x1 + (5/math.pi)*x1 - 6)**2 + 10*(1 - (1/(8*math.pi)))*torch.cos(x1) + 10
    return result






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




def EI(x, train_y, model, likelihood):

    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.resize_((1,2))
    x = x.to(device)
    f_star = torch.min(train_y)

    model.eval()
    likelihood.eval()
    f_preds = model(x)
    pos_mu = f_preds.mean
    pos_var = f_preds.variance
    pos_sigma = pos_var.clamp_min(1e-9).sqrt()


    W = (f_star - pos_mu)/pos_sigma
    normal = Normal(torch.zeros_like(W), torch.ones_like(W))
    ucdf = normal.cdf(W)
    updf = torch.exp(normal.log_prob(W))
    ei = pos_sigma * (updf + W * ucdf)
    result = -ei
    result = result.cpu().detach().contiguous().double().clone().numpy()

    return result




def next_sampling_point(EI, train_y, model, likelihood, bnds, opt_method, maxiter, dtype, device, nRestart =25):


    min_value = 1000
    point_to_sample = None


    objective = functools.partial(EI, train_y = train_y,model = model, likelihood = likelihood)
    initial_points = torch.rand(nRestart, 2, dtype=dtype, device=device)*15 + torch.tensor([-5.0, 0.0], dtype=dtype, device=device)
    initial_points = initial_points.cpu()
    
    for i in range(nRestart):
        warnings.filterwarnings("ignore")
       
        initial_point = initial_points[i]
        function_to_optimize = objective

        warnings.filterwarnings("error")

        res = spo.minimize(function_to_optimize, initial_point, method = optimize_method, \
                               options={'maxiter':maxiter, 'disp': False}, bounds = bnds)  

        if res.fun < min_value:
            min_value = res.fun
            point_to_sample = res.x   
#         except:
#             continue

    point_to_sample = torch.from_numpy(point_to_sample)
    point_to_sample = point_to_sample.type(dtype)
    point_to_sample = point_to_sample.reshape(1,2)
    point_to_sample = point_to_sample.to(device)
    return point_to_sample


start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
nRestart = 1000
optimize_method = 'SLSQP'
maxiter = 10000
d = 2
training_iter = 100
# print(torch.cuda.get_device_name(device))


bnds = [np.array([-5., 10.]), np.array([ 0., 15.])]
opt_method = 'SLSQP'




train_x = torch.tensor([[-2.018476606457733, 10.28593793628148], [0.8510493167919746, 6.888642195088246], [9.881155137925708, 1.8820277741834306]],\
                      dtype=dtype, device=device)

train_y = torch.zeros(train_x.shape[0], dtype=dtype, device=device)
for i in range(train_x.shape[0]):
    train_y[i] = target(train_x[i,:])



iterations = 2

for i in range(iterations):
    warnings.filterwarnings('ignore')


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model = model.to(device)
    likelihood = likelihood.to(device)
    model, likelihood = train_GP_model(model, likelihood, train_x, train_y, training_iter)

    new_x = next_sampling_point(EI, train_y, model, likelihood, bnds, opt_method, maxiter, dtype, device, nRestart)
    new_y = target(new_x[0])
    train_x =  torch.cat((train_x, new_x), 0)
    train_y = torch.cat((train_y, new_y.reshape(1)), 0)
    print(i, ' th iterations time is ', time.time() - start_time)
    