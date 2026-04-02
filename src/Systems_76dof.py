#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for generating response for the 76 DOF structure
"""

import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from functools import partial
from EQD import utils

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 20
np.random.seed(0)

# %%
def normalize(mode):
    nmode = np.zeros(mode.shape)
    for i in range(len(mode[0])):
        nmode[:,i] = mode[:,i]/mode[-1,i]
    return nmode

""" The 76 storey building properties """
data = sio.loadmat('data/B76_inp.mat')
M76 = data['M76']
K76 = data['K76']
C76 = data['C76']

# Get the natural frequencies and modes:
eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(M76),K76))

# Get the first modal mass:
ms = 153000 
idx = np.where(eigval == np.min(eigval))
phi = normalize(eigvec)
modal_m = np.matmul( np.dot(phi[:,idx[0]].T, M76), phi[:,idx[0]] ).squeeze()

# Modal mass ratio:
md = 500 
mu = md/modal_m  # mass ratio

# Optimum damper parameters:
fopt = 1/(1+mu)
zopt = np.sqrt((3*mu)/(8*(1+mu)))

# Find the optimum stiffness and damping
omega = np.sqrt(eigval)/2/np.pi
kd = (fopt**2 * omega[idx]**2 * md).squeeze()
cda = (2 * zopt * fopt * omega[idx] * md).squeeze()

M77 = np.zeros([77,77])
K77 = np.zeros([77,77])
C77 = np.zeros([77,77])
M77[:76,:76] = M76
M77[-1,-1] = md

K77[:76,:76] = K76
K77[-2,-2] = K76[-2,-2] + kd
K77[-2,-1] = -kd
K77[-1,-2] = -kd
K77[-1,-1] = kd

C77[:76,:76] = C76
C77[-2,-2] = C76[-2,-2] + cda
C77[-2,-1] = -cda
C77[-1,-2] = -cda
C77[-1,-1] = cda

def solve_76dof(M,C,K,ic,T,dt,action):
    # Statespace matrix
    sys = np.row_stack(( np.column_stack(( np.zeros(M.shape), np.eye(M.shape[0]) )), \
                         np.column_stack(( -np.matmul(np.linalg.inv(M),K), 
                                           -np.matmul(np.linalg.inv(M),C) )) ))
        
    def dydt(x,t,statespace,sigma,force,H,action,iM):
        
        # Calculate the wind force distribution over the surface
        force = sigma @ iM @ force
        force = np.matmul(sigma,force)
        force = np.concatenate(( np.zeros_like(force), force ))
        
        # Nonlinearity
        alpha = iM @ (-1e5 * np.ones(dof))
        
        nonlin = np.zeros(77) 
        nonlin[35] = alpha[35] * (x[35])**3
        nonlin[52] = alpha[52] * (x[52])**3
        nonlin[75] = alpha[75] * (x[75])**3
    
        nonlin = np.concatenate(( np.zeros_like(nonlin), nonlin ))

        # Define control force        
        control_force = np.matmul(H,action) 
        control_force = np.concatenate(( np.zeros_like(control_force), control_force ))

        # Evaluate the derivatives
        dydt = np.matmul(statespace,x) + nonlin + control_force + force
        return dydt
        
    # Wind load calculation:
    L = 42
    z_ref = 10
    # u_ref = 1
    rho = 1.22
    cd = 1.2
    z = np.concatenate((
                        np.array([10]),
                        10 + np.linspace(4.5,9,2),
                        10 + 9 + np.linspace(3.9,136.5,35),
                        10 + 9 + 136.5 + np.linspace(4.5,9,2),
                        10 + 9 + 136.5 + 9 + np.linspace(3.9,132.6,34),
                        10 + 9 + 136.5 + 9 + 132.6 + np.linspace(4.5,9,2)
                       ))
    
    # Define the forcing function:
    fun = lambda rho, L, cd, u_ref, z_ref, z: 0.5*rho*L*cd*u_ref**2*(z/z_ref)**0.365
    
    # Compressing the forcing function:
    intensity = partial(fun, rho=rho, L=L, cd=cd, z_ref=z_ref, z=z)
    
    # Define the excitation influence matrix:
    dof = M.shape[0]
    sigma = np.eye(dof)
    sigma[-1,-1] = 0
    
    # Inverse of mass matrix
    iM = inv(M)
    
    # Define control influence matrix:        
    c_idx = np.array([5, 10, 15, 25, 35, 45, 55, 60, 65, 72, 75, 76])
    H = np.zeros((77, len(c_idx)))                      # Influence matrix
    H[(c_idx, np.arange(len(c_idx)))] = 1

    # Time integration:
    t_eval = np.arange(0, T+dt, dt)
    xt = np.vstack(ic)
    force = []
    for i in range(len(t_eval)-1):
        ft = np.concatenate(( intensity(u_ref = np.sin(np.pi*i*dt)), np.zeros(1) ))
        sol = odeint(func=dydt, t=[i*dt, i*dt+dt], y0=ic, mxstep=1000, args=(sys,sigma,ft,H,action,iM))
        solx = np.vstack(sol[-1,:])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        ic = np.ravel(solx)
        force.append(ft)
    
    # Extract the displacement and velocities:
    states = len(ic)//2
    D = xt[:states, :-1]
    V = xt[states:, :-1]
    
    ft = [ np.concatenate(( intensity(u_ref = np.sin(np.pi*t)), np.zeros(1) )) for t in t_eval ]
    force = iM @ np.column_stack(( ft ))[:, :-1]
    
    # Compute acceleration response:
    A = []
    for i in range(dof):
        A.append( utils.FiniteDiff(V[i], dx=dt, d=1) )
    A = np.stack((A))
                         
    return t_eval[:-1], np.stack((D,V,A))
    

# Configure the system parameters:
T = 1
dt = 0.001
record_dt = 0.001
states = 77
x0 = 0.1 * np.concatenate((np.linspace(0,1,states-1), np.zeros(1), np.zeros(states)))

# Generate the response:
t, xt = solve_76dof(M77, C77, K77, ic=x0, T=T, dt=dt, action=np.zeros(12))
t = t[::int(record_dt/dt)]

burn_steps = int(0.25/dt)
t, xt = t[burn_steps:], xt[..., burn_steps:] 


# %%
floor = [5,35,52,74]

figure1, ax = plt.subplots(nrows=len(floor), ncols=3, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35, wspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(t, xt[0, floor[i],:], label='DOF:{}th'.format(floor[i]))
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

