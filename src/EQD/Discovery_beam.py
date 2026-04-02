#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:53:05 2022

@author: user
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname('EQD'), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils_data
import bayes_torch
import scipy.io as sio
import utils

torch.manual_seed(0)
np.random.seed(0)
device = 'cpu' 

# %%
""" Generating system response """

# Steel properties: 
rho = 7800
E = 2e11
b, d = 0.0254, 0.002
A = b*d 
I = b*d**3/12 
L = 1
c1, c2 = 0, 0
mode = 0
sparams = [rho, E, b, d, A, I, L, mode, c1, c2]
print('True parameter: {}'.format( (E*I)/(rho*A)) )

# The time parameters: 
Ne = 200
dx = 1/Ne
T, dt = 1, 1e-3
tparams = [Ne, T, dt ]
xgrid = np.arange(L/Ne, L+L/Ne, L/Ne)

theta_true = np.zeros(15)
theta_true[6] = (E*I)/(rho*A)

record_dt = 1e-2
# Force parameters: 
F = lambda t : 0
# F = lambda t : np.sin(np.pi*t) 

# Load the data:
t_eval, xt = utils_data.cantilever(sparams, tparams, force=F)
# force = F(t_eval)

# Subsample the data for computational efficiency:
xt_record = xt[:, :, ::int(record_dt/dt)]
t_eval = t_eval[::int(record_dt/dt)]
# force = force[::int(record_dt/dt)].repeat(Ne)

# Plot the data for visualization:
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 18
    
fig1, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,6), dpi=100)
plt.subplots_adjust(hspace=0.3)
label = ['$u(t)$', '$\dot{u}(t)$', '$\ddot{u}(t)$']
for i in range(3):
    im = ax[i].imshow(xt_record[i, ...].T, cmap='jet', origin='lower', aspect='auto'); 
    ax[i].set_ylabel(label[i]); 
    plt.colorbar(im, ax=ax[i], pad=0.01)
plt.show()

# %%
# Convert the data to tensor:
xt_record = torch.tensor(xt_record, device=device)
force = torch.zeros(Ne, len(t_eval), device=device)

yy = xt_record[2, ...] 
xx = xt_record[:2, ...] 
yy = yy + 0.02*torch.std(yy)*torch.rand(yy.shape, device=device)

# %%
# Dictionary Creation:
# force = 0.05*np.sin(2*np.pi*t_eval) 
D, nl = bayes_torch.library_pde(xx, dx=dx, force=force, modulus=True, device=device)

# %%
# SINDy
lam = 5
Xi = bayes_torch.sparsifyDynamics(D, yy, lam).cpu().numpy()
print('SINDy-error:{}'.format(np.linalg.norm(theta_true-Xi)/np.linalg.norm(Xi)))

# The gibbs sampler:
model = bayes_torch.Gibbs(ns=2, nl=nl, nt=Ne*len(t_eval), iterations=1000, device=device) 
ppp
theta_mean, z_mean, theta_std = model.forward(D, yy, verbose=True, verbose_interval=10)    

PIP = 0.5
theta_mean[ torch.where(z_mean < PIP) ] = 0
theta_mean = theta_mean.numpy()
z_mean = z_mean.numpy()
theta_std = theta_std.numpy()
print('Bayes-error:{}'.format(np.linalg.norm(theta_mean-Xi)/np.linalg.norm(Xi)))

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

fig1 = plt.figure(figsize = (10, 6))
yr = 0.5*np.ones(nl)
xr = np.arange(nl)
plt.stem(xr, z_mean, linefmt='blue')
plt.axhline(0.5, color='r', linestyle='--')
plt.legend(["Identified states", "P(Y=0.5)"])
plt.xlabel('Library functions'); 
plt.ylabel('Posterior inclusion probability (PIP)');
plt.grid(True, alpha=0.35); 
plt.title('Identification')
plt.show()

# %%
fig2 = plt.figure(figsize=(10, 5))
plt.hist(x=model.theta[6,:].numpy(), bins=25, density=True, color='b')
plt.xlabel('Parameter: 'r'$\theta (\partial_{xxxx} u)$', fontweight='bold'); 
plt.ylabel('$p'r'(\theta_{\partial_{xxxx}u})$')
plt.grid(True, alpha=0.35) 
plt.xlim([theta_mean[6] - 6*np.sqrt(np.abs(theta_std[6,6])),
          theta_mean[6] + 6*np.sqrt(np.abs(theta_std[6,6]))])
plt.show()

# %%
"""
For saving the trained model and prediction data
"""
utils.savemodel(model, "discovery_ebeam_19.obj")
sio.savemat('results/discovery_beam_19.mat', mdict={'sindy_theta':Xi,
                                                 'theta_mean':theta_mean,
                                                 'z_mean':z_mean,  
                                                 'theta_std':theta_std})
