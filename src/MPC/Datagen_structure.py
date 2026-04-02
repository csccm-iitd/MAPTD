# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:27:29 2024

@author: tripu
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from timeit import default_timer
import scipy.io as sio

import env_skyscraper
import utils_mpc

np.random.seed(10)

plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %%
# """ Load the control force data """
path = 'results/structure_mpc.mat'
data = sp.io.loadmat(path)

xSamples = data['xtHistory']
uSamples = data['uHistory']
fSamples = data['fHistory']
t = data['tHistory'].flatten()
dt = t[5] - t[4]
rSamples = data['rHistory'].flatten()

# %%
""" Generating system response """
# The time parameters: 
T, dt     = 4, 1e-2
Nt        = int(T/dt)
Ts        = 0.01               # Sampling time
record_dt = dt            # Recording sampling time
CT        = 20
x         = np.arange(12)

# Generate the env and solve the system
env = env_skyscraper.Skyscraper_rom(T=T, dt=dt, forced=True)

# Generate the response:
t, xt = env.solve()

# %%  
floor = [4,13,17,22]

figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(t, xt[0, floor[i],:], label='DOF:{}th'.format(floor[i]))
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('$X({})$'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    
    ax[i,1].plot(t, xt[1, floor[i],:], label='DOF:{}th'.format(floor[i]))
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('$X({})$'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 

plt.suptitle('Response of 76 DOF Skyscraper', y=0.94)
plt.show()

# %%      
def grf(x): 
    sigma = 4; l = 0.5;
    x = x - np.mean(x)
    kxx = (np.outer(x,x) + l)**2
    y = sigma**2 * np.random.multivariate_normal(mean=np.zeros(x.shape), cov=kxx)
    return np.abs(y) - np.max(np.abs(y)) 

# Run the Control loop:
t0 = default_timer()
    
# Get the forcing function:
Force  = env.force_fun
Intensity = env.intensity

D0n, V0n, A0n = env.reset()             # Initial condition    
rHistory = env.D_ref  
xH, xdH, xddH = D0n, V0n, A0n           # Stores state history
fH = Intensity(u_ref = Force(0*dt))
tH = 0                    # Stores time history
        
for run in range(int(CT/Ts)-1):
    t1 = default_timer()

    # action = uSamples[:,run+1]
    uopt0 = grf(x)
    
    D0n, V0n, A0n = env.step(states=(D0n,V0n,A0n), t=run, action=uopt0)
    
    error_d, error_v = utils_mpc.L2norm(D0n[-3:-1], V0n[-3:-1], rHistory[-3:-1])
    
    t2 = default_timer()
    print('Iteration-{}, Time-{:0.4f}, Error: Displacement: {:0.4f}, Velocity: {:0.4f}'.format(
            run, t2-t1, error_d, error_v))
    
    # Store the history:
    xH = np.column_stack((xH, D0n))
    xdH = np.column_stack((xdH, V0n))
    xddH = np.column_stack((xddH, A0n))
    if run == 0:
        uH = uopt0
    else:
        uH = np.column_stack((uH, uopt0))
    fH = np.column_stack((fH, Intensity(u_ref = Force(run*dt)) ))
    tH = np.append(tH, run*Ts)

tT = default_timer()
print('-- Run Complete -- Total Time-{}min.'.format((tT-t0)/60))

# Concatenate the responses:
xtc = np.stack(( xH, xdH, xddH ))

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 16

fig3, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,8), dpi=100,
                        gridspec_kw={'height_ratios':[1,1,0.5]})

ax[0].plot(tH, xH[-3, :], label='Controlled')
# ax[0].plot(tH[:-1], xt[0,-3,:], label='True')
ax[0].set_ylabel('$x(t)$ at DOF-76')

for j in range(env.action_dim):
    ax[1].plot(tH[:-1], uH[j, :], label='@DOF-{}'.format(j+1))
ax[1].set_ylabel('Action-$u(t)$')

ax[2].plot(tH, fH[-2, :])
ax[2].set_ylabel('Excitation-$F(t)$ \n at DOF-76')

for i in range(3):
    if i < 2: ax[i].legend(loc=2, labelspacing=0.1, handletextpad=0.5, handlelength=1, borderaxespad=0.1)
    if i == 2: ax[i].set_xlabel('Time (s)')
    ax[i].grid(True, alpha=0.5)
plt.show()

# %%
floor = [4,13,17,22]
    
t = np.arange(0,1,0.01)

figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(xtc[0, floor[i], :], color='r', label='Controlled')
    ax[i,0].plot(xt[0, floor[i], :], linestyle='--', label='True')
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('DOF-{}'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    ax[i,0].legend()
    
    ax[i,1].plot(xtc[1, floor[i], :], color='r', label='Controlled')
    ax[i,1].plot(xt[1, floor[i], :], linestyle='--', label='True')
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('DOF-{}'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 
    ax[i,1].legend()

ax[0,0].set_title('Displacement')
ax[0,1].set_title('Velocity')
plt.suptitle('Response of 76 DOF Skyscraper (RL ENV)', y=0.94)
plt.show()

# %%
# Save the results,
sio.savemat('results/structure_mpc_data.mat', mdict={'xH':xt,
                                                     'xHc':xtc,
                                                     'uH':uH,
                                                     'fH':fH,
                                                     'tH':tH,
                                                     'ref':rHistory})

