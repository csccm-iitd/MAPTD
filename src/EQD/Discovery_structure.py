#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:53:05 2022

@author: user
"""

# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

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
T = 10
dt = 0.01
record_dt = 0.01
states = 77         # full-model dimension (76 floors + TMD)
alpha = [4.6e7, 3.7e6, 3.7e5]   # cubic stiffness at nl_idx (kN/m^3)
nl_idx = [35, 52, 75]           # nonlinear DOFs
u_ref = 0                       # wind speed at 10 m (m/s); 0 = free vibration
amp = 0.1                       # max. initial displacement (m); sets the nonlinearity level
D0 = -np.exp(-0.03*np.arange(states))
D0 = amp * (D0-np.min(D0)) / np.max(D0-np.min(D0))
print('Max. displacement: {:0.4f} m'.format(np.max(D0)))
V0 = np.zeros(states)

# Get the full-order model:
M77, K77, C77, id_ = utils_data.skyscraper()
x0 = np.concatenate((D0, V0))

# Generate the response:
t, xt, ft = utils_data.solve_76dof(M77, C77, K77, ic=x0, T=T, dt=dt, alpha=alpha, nl_idx=nl_idx, u_ref=u_ref)

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

# Forcing at the nonlinear DOFs (no force bases in free vibration):
force = ft[nl_idx, ...] if u_ref > 0 else None

# Library index of the cubic basis u_j^3: [1, x (77), v (77), x^2 (77), x^3 (77), ...]
cub_idx = [1 + 2*states + states + j for j in nl_idx]

# %%
floor = [5] + nl_idx

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
	ax[i,1].set_ylabel(r'$\dot X({})$'.format(floor[i]))
	ax[i,1].grid(True, alpha=0.25)

	ax[i,2].plot(t, xt[2, floor[i],:], label='DOF:{}th'.format(floor[i]))
	if i == len(floor)-1: ax[i,2].set_xlabel('Time (Sec)')
	ax[i,2].set_ylabel(r'$\ddot X({})$'.format(floor[i]))
	ax[i,2].grid(True, alpha=0.25)

plt.suptitle('Response of 76 DOF Skyscraper', y=0.94)
plt.show()

# %%
# Convert the data to tensor:
xx = np.concatenate((xt[0, ...], xt[1, ...]))
yy = xt[2, nl_idx, ...]
# yy = nxt[[35,52,70], ...]

# %%
# Dictionary Creation:
D, nl = bayes_numpy.library(xx, polyn=4, force=force, harmonic=False, modulus=False)
nt = D.shape[0]

# %%
# SINDy
Xi = []
lam = [1e2,1e2,1e2]
for i in range(3):
	Xit = bayes_numpy.sparsifyDynamics(D, yy[i:i+1], lam[i])
	Xi.append(Xit)

Xi = np.array(Xi).squeeze()
print('SinDy Identification:', Xi[([0,1,2],cub_idx)])

# True coefficient of u_j^3 in the acceleration equation of DOF j: -[M^-1]_jj * alpha_j
true = -np.diag(np.linalg.inv(M77))[nl_idx] * np.asarray(alpha)
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
	model = bayes_numpy.Gibbs(ns=states, nl=nl, nt=nt, iterations=iters[i], burn_in=burns[i])
	theta_m, z_m, theta_s, theta_st = model.forward(D, yn[i], verbose=True, verbose_interval=50)

	theta_mean.append( theta_m )
	z_mean.append( z_m )
	theta_store.append( theta_s )
	theta_std.append( theta_st )

	utils.savemodel(model, "discovery_estr_dof{}.obj".format(nl_idx[i]))

# Get the numpy arrays
theta_mean = np.stack(( theta_mean ))
z_mean = np.stack(( z_mean ))
theta_store = np.stack(( theta_store ))
theta_std = np.stack(( theta_std ))
# theta_mean[ np.where(z_mean < PIP) ] = 0

print('Bayesian Identification:', z_mean[([0,1,2],cub_idx)])
print('Bayesian Identification:', theta_mean[([0,1,2],cub_idx)])

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

fig1, ax = plt.subplots(ncols=1, nrows=3, figsize=(16,8), dpi=100)
plt.subplots_adjust(hspace=0.5)

yr = 0.5*np.ones(nl)
xr = np.arange(nl)
xmin, xmax = -0.5, 0.5
dofs = (['(a)', '(b)', '(c)'], nl_idx)

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

for i in range(3):
	ax[i].text(cub_idx[i], 0.8, r'$\alpha_{%d} u_{%d}^3$' % (nl_idx[i], nl_idx[i]))

ax[0].legend()
ax[2].set_xlabel('Library functions')

plt.show()

# %%
bases = ['$u_{%d}^3$' % j for j in nl_idx]

fig2, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4), dpi=100)
plt.subplots_adjust(wspace=0.4)

for i in range(3):
	ax[i].hist(x=-1*theta_store[i][cub_idx[i],:], bins=25, density=True, color='b')
	ax[i].set_xlabel(r'$\theta_{%d}^3$' % nl_idx[i])
	ax[i].set_ylabel(r'$p\left(\theta_{%d}^3\right)$' % nl_idx[i])
	ax[i].axvline(x=-1*theta_mean[i,cub_idx[i]], color='r', linewidth=3, linestyle='--', label='Mean')
	ax[i].axvline(x=-1*true[i], color='k', linewidth=4, linestyle='--', label='True')
	ax[i].grid(True, alpha=0.35)
	ax[i].set_title('{} PDF of {}'.format(dofs[0][i],bases[i]))
	ax[i].legend(borderaxespad=0.1)

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
