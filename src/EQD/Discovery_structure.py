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
    
import numpy as np
import matplotlib.pyplot as plt
import utils_data
import bayes_numpy
import scipy.io as sio
import utils

np.random.seed(0)

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 20

# %%
""" Generating system response """

# Configure the system parameters:
T = 2.5
dt = 0.001
record_dt = 0.001
states = 77         # ROM-states dimension 
D0 = -np.exp(-0.03*np.arange(states))
D0 = 0.0001 * (D0-np.min(D0))
print('Max. displacement: {:0.4f} m'.format(np.max(D0)))
V0 = np.zeros(states)

# Get the reduced-order model:
M77, K77, C77, id_ = utils_data.skyscraper()
x0 = np.concatenate((D0, V0))

# Generate the response:
t, xt, ft = utils_data.solve_76dof(M77, C77, K77, ic=x0, T=T, dt=dt)

# Cut initial steps:
burn_steps = int(0.05/dt)
t, xt, ft = t[burn_steps:], xt[..., burn_steps:], ft[..., burn_steps:]

# Subsample data-points:
t, xt, ft = t[::int(record_dt/dt)], xt[:,:,::int(record_dt/dt)], ft[:,::int(record_dt/dt)]

# Nonlinear response:
# nxt = ft - M77 @ xt[2] - C77 @ xt[1] - K77 @ xt[0] 

# fl = np.linalg.inv(M77) @ K77 @ xt[0]
# alpha = np.linalg.inv(M77) @ (1e14 * np.ones(states))
# fnl = np.stack(([np.multiply(alpha, xt[0,:,i]**3) for i in range(xt.shape[-1])])).T
# ratio = (fnl/fl)[[35,52,70], :]

# subtract the linear information
force = ft[[35,52,70], ...]

# %%
floor = [5,35,52,70]

figure1, ax = plt.subplots(nrows=len(floor), ncols=3, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35, wspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(t, xt[0, floor[i],:], label='DOF:{}th'.format(floor[i]))
    # ax[i,0].plot(t, nxt[floor[i], :], color='k', label='DOF:{}th'.format(floor[i]))
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('$X({})$'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    
    ax[i,1].plot(t, xt[1, floor[i],:], label='DOF:{}th'.format(floor[i]))
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('$X({})$'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 
    
    ax[i,2].plot(t, xt[2, floor[i],:], label='DOF:{}th'.format(floor[i]))
    if i == len(floor)-1: ax[i,2].set_xlabel('Time (Sec)')
    ax[i,2].set_ylabel('$X({})$'.format(floor[i]))
    ax[i,2].grid(True, alpha=0.25) 

plt.suptitle('Response of 76 DOF Skyscraper', y=0.94)
plt.show()

# %%
# Convert the data to tensor:
xx = np.concatenate((xt[0, ...], xt[1, ...]))
yy = xt[2, [35,52,70], ...]
# yy = nxt[[35,52,70], ...]

# %%
# Dictionary Creation:
D, nl = bayes_numpy.library(xx, polyn=5, force=force, harmonic=False, modulus=False)
nt = D.shape[0]

# %%
# SINDy
Xi = []
lam = [1e2,1e3,1e2] 
for i in range(3):
    Xit = bayes_numpy.sparsifyDynamics(D, yy[i:i+1], lam[i])
    Xi.append(Xit)

Xi = np.array(Xi).squeeze() 
print('SinDy Identification:', Xi[([0,1,2],[267,284,302])])

true = np.array([-3.53697430e+09, -4.50235599e+09, -5.89515819e+09])
print('True: ', true)

# %%
# Add noise
yn = yy + 0.05 * np.std(yy, axis=-1, keepdims=True) * np.random.randn(*yy.shape)
# yn = yy

# %%
# Gibbs MCMC sampling
theta_mean, z_mean, theta_store, theta_std = [], [], [], []
PIP = 0.5
iters, burns = [2500, 2500, 2500], [500, 500, 500]
for i in range(3):
    # The gibbs sampler:
    model = bayes_numpy.Gibbs(ns=77, nl=nl, nt=nt, iterations=iters[i], burn_in=burns[i]) 
    theta_m, z_m, theta_s, theta_st = model.forward(D, yn[i], verbose=True, verbose_interval=50)    
    
    theta_mean.append( theta_m )
    z_mean.append( z_m )
    theta_store.append( theta_s )
    theta_std.append( theta_st )
    
    utils.savemodel(model, "discovery_estr_dof{}.obj".format(floor[i+1]))

# Get the numpy arrays
theta_mean = np.stack(( theta_mean ))
z_mean = np.stack(( z_mean ))
theta_store = np.stack(( theta_store ))
theta_std = np.stack(( theta_std ))
# theta_mean[ np.where(z_mean < PIP) ] = 0

print('Bayesian Identification:', z_mean[([0,1,2],[267,284,302])])
print('Bayesian Identification:', theta_mean[([0,1,2],[267,284,302])])

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

fig1, ax = plt.subplots(ncols=1, nrows=3, figsize=(16,8), dpi=100)
plt.subplots_adjust(hspace=0.5)

yr = 0.5*np.ones(nl)
xr = np.arange(nl)
xmin, xmax = -0.5, 0.5
dofs = (['(a)', '(b)', '(c)'], [35,52,70])

xrp = np.arange(xmin,nl+xmax,1)
yrp = np.ones(len(xrp))

ax[0].stem(xr, z_mean[0], linefmt='blue', label='Identified basis')
ax[1].stem(xr, z_mean[1], linefmt='blue', label='Identified basis')
ax[2].stem(xr, z_mean[2], linefmt='blue', label='Identified basis')

for i in range(3):
    ax[i].fill_between(xrp, 0*yrp, 0.5*yrp, color='grey', alpha=0.25)
    ax[i].fill_between(xrp, 0.5*yrp, yrp, color='b', alpha=0.15)
    
    # ax[i].axhline(0.5, color='r', linestyle='--', label=''r'P($z_k=0.5 | \boldsymbol{y}$)')
    ax[i].axhline(0, color='k')

    ax[i].set_ylabel('mPIP');
    ax[i].grid(True, alpha=0.35); 
    ax[i].set_xlim([-0.5,nl-1+0.5])
    ax[i].set_title('{} Model selection of DOF: {}'.format(dofs[0][i], dofs[1][i]))

ax[0].text(267,0.8, ''r'$\alpha_{35} u_{35}^3$')
ax[1].text(284,0.8, ''r'$\alpha_{52} u_{52}^3$')
ax[2].text(302,0.8, ''r'$\alpha_{70} u_{70}^3$')

ax[0].legend()
ax[2].set_xlabel('Library functions')

plt.show()

# %%
bases = ['$u_{35}^3$', '$u_{52}^3$', '$u_{70}^3$']

fig2, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4), dpi=100)
plt.subplots_adjust(wspace=0.4)

ax[0].hist(x=-1*theta_store[0][267,:], bins=25, density=True, color='b')
ax[1].hist(x=-1*theta_store[1][284,:], bins=25, density=True, color='b')
ax[2].hist(x=-1*theta_store[2][302,:], bins=25, density=True, color='b')

ax[0].set_xlabel('$'r'\theta_{35}^3$') 
ax[0].set_ylabel('$p'r'\left(\theta_{35}^3\right)$') 
ax[1].set_xlabel('$'r'\theta_{52}^3$') 
ax[1].set_ylabel('$p'r'\left(\theta_{52}^3\right)$') 
ax[2].set_xlabel('$'r'\theta_{70}^3$') 
ax[2].set_ylabel('$p'r'\left(\theta_{70}^3\right)$') 

ax[0].axvline(x=np.abs(theta_mean[0,267]), color='r', linewidth=3, linestyle='--', label='Mean')
ax[1].axvline(x=np.abs(theta_mean[1,284]), color='r', linewidth=3, linestyle='--', label='Mean')
ax[2].axvline(x=np.abs(theta_mean[2,302]), color='r', linewidth=3, linestyle='--', label='Mean')

for i in range(3):
    ax[i].grid(True, alpha=0.35) 
    # ax[i].axvline(x=np.abs(true[i]), color='k', linewidth=4, linestyle='--', label='True')
    ax[i].set_title('{} PDF of {}'.format(dofs[0][i],bases[i]))
    ax[i].legend(borderaxespad=0.1)

ax[0].set_xlim([-2e6, 2e6])
ax[1].set_xlim([-1e6, 1e6])
ax[2].set_xlim([-2e6, 2e6])

plt.show()

# %%
"""
For saving the trained model and prediction data
"""
# sio.savemat('results/discovery_str.mat', mdict={'sindy_theta':Xi,
#                                                 'theta_mean':theta_mean,
#                                                 'z_mean':z_mean, 
#                                                 'theta_store':theta_store,
#                                                 'theta_std':theta_std})
