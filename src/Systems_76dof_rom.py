#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for generating response for the 76 DOF structure
- using Reduced-order modelling
"""

import sys
import os

import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from functools import partial

current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, 'EQD'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import utils

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

# Tuned-mass augmented system:
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

id_ = [2, 5, 9, 12, 15, 19, 22, 25, 29, 32, 35, 39, 
        42, 45, 49, 52, 55, 59, 62, 65, 69, 72, 75, 76]

# Perform Model-order reduction:
def rom(m,k,c,order=None):
    
    # Get the natural frequencies and modes:
    eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(m),k))
    
    # Sort the eigen-values in ascending order:
    val = np.sqrt( np.sort(eigval) )
    
    # If order is not given, find the minimum orde using energy criteria:
    if order == None: 
        energy = 0.5    # 50\% enrgy of the total system
        sum_ = 0
        order = 0
        for value in val:
            sum_ += value
            order += 1
            if sum_ > energy * np.sum(val):
                break
    
    # Model-order reduction equation:
    rom_system = lambda matrix, phi: np.matmul( np.matmul(phi.T, matrix), phi )
    
    # Truncate the modes:
    if isinstance(order, list):
        val = -np.sort(-val)[order] 
        vec = eigvec[:, order]
    else:
        val = -np.sort(-val[:order])
        vec = eigvec[:, -order:]
    
    # Get the reduced-order matrices:
    m_rom = rom_system(m, vec) 
    k_rom = rom_system(k, vec) 
    c_rom = rom_system(c, vec) 
    return (m_rom, k_rom, c_rom, vec)


# %%
def solve_76dof(M,C,K,ic,T,dt,action,eigvec,mt,ct,kt):
    # Statespace matrix
    sys = np.row_stack(( np.column_stack(( np.zeros(M.shape), np.eye(M.shape[0]) )), \
                         np.column_stack(( -np.matmul(np.linalg.inv(M),K), 
                                           -np.matmul(np.linalg.inv(M),C) )) ))
        
    def dydt(x,t,statespace,sigma,force,H,action,phi,iM):
        
        # Calculate the wind force distribution over the surface
        # Transform to reduced dimension:
        force = np.matmul(phi.T, iM @ np.matmul(sigma,force))
        force = np.concatenate(( np.zeros_like(force), force ))
        
        # Define control force
        control_force = np.matmul(phi.T, np.matmul(H,action))
        control_force = np.concatenate(( np.zeros_like(control_force), control_force ))
        
        # Evaluate the derivatives
        alpha = iM @ (-1e3 * np.ones(phi.shape[0]))
        xrom = np.matmul(phi, x[:24])
        nonlin = np.zeros(phi.shape[0])
        nonlin[35] = alpha[35] * (xrom[35])**3
        nonlin[52] = alpha[52] * (xrom[52])**3
        nonlin[75] = alpha[75] * (xrom[75])**3
    
        nonlin = np.matmul(phi.T, nonlin)
        nonlin = np.concatenate(( np.zeros_like(nonlin), nonlin ))

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
    dof = mt.shape[0]
    sigma = np.eye(dof)
    sigma[-1,-1] = 0
    
    # Inverse of mass matrix
    iM = inv(mt)
    
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
        sol = odeint(func=dydt, t=[i*dt, i*dt+dt], y0=ic, mxstep=1000, args=(sys,sigma,ft,H,action,eigvec,iM))
        solx = np.vstack(sol[-1,:])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        ic = np.ravel(solx)
        force.append(ft)
    
    # Extract the displacement and velocities:
    states = len(ic)//2
    D = eigvec @ xt[:states, :-1]
    V = eigvec @ xt[states:, :-1]
    force = iM @ np.column_stack(( force ))
     
    A = []
    for i in range(dof):
        A.append( utils.FiniteDiff(V[i], dx=dt, d=1) )
    A = np.stack((A))
    return t_eval[:-1], np.stack(( D,V,A ))
    

# %%
# Configure the system parameters:
T = 1
dt = 0.001
record_dt = 0.001
states = 77         # ROM-states dimension 
D0 = 0.1 * np.concatenate((np.linspace(0,1,states-1), np.zeros(1)))
V0 = np.zeros(states)

# Get the reduced-order model:
m_rom, k_rom, c_rom, vec = rom(m=M77,k=K77,c=C77,order=id_)

D0 = vec.T @ D0
V0 = vec.T @ V0
x0 = np.concatenate((D0, V0))

# Generate the response:
t, xt = solve_76dof(m_rom, c_rom, k_rom, ic=x0, T=T, dt=dt, action=np.zeros(12), eigvec=vec,
                    mt=M77, ct=C77, kt=K77)
t = t[::int(record_dt/dt)]
xt = xt[:,:,::int(record_dt/dt)]


# %%
floor = [5,8,11,74]

figure1, ax = plt.subplots(nrows=len(floor), ncols=3, figsize =(18,10), dpi=100)
plt.subplots_adjust(hspace=0.35,wspace=0.35)

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

