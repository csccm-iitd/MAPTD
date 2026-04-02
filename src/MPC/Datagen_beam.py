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

import env_beam
import utils_mpc

np.random.seed(0)

plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %%
# """ Load the control force data """
path = 'results/beam_mpc.mat'
data = sp.io.loadmat(path)

uHistory = data['uHistory']
fHistory = data['fHistory']
tHistory = data['tHistory'].squeeze()
xHistory = data['xtHistory']
N = uHistory.shape[-1]
dt = tHistory[5] - tHistory[4]
T = N * dt

# %%
""" Generating system response """
# The time parameters: 
# T, dt     = 10, 1e-2
# Nt        = int(T/dt)
Ts        = 0.01               # Sampling time
record_dt = dt            # Recording sampling time
CT        = T
x         = np.arange(0,1,0.01)

# Generate the env and solve the system
env = env_beam.EulerBeam(T=T, dt=dt, forced=True)
xt = env.solve()
        
# Plot the data for visualization:
fig1, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,8), dpi=100)
plt.subplots_adjust(hspace=0.4)

label = ['$u(t)$', '$\dot{u}(t)$', '$\ddot{u}(t)$']
for i in range(3):
    im = ax[i].imshow(xt[i, ::2].T, cmap='jet', origin='lower', aspect='auto'); 
    ax[i].set_ylabel(label[i]); 
    plt.colorbar(im, ax=ax[i], pad=0.01)
plt.suptitle('Uncontrolled Response')
plt.show()

# %%      
def grf(x): 
    y = np.zeros(2*x.shape[0])
    x = x[32:32*3]
    sigma = 1; l = 0.9;
    kxx = sigma**2 * np.exp( -(x[:,None] - x[None,:])**2/(2 * l**2) )
    y[32:32*3] = np.random.multivariate_normal(mean=np.zeros(x.shape), cov=kxx)
    return y * np.linspace(0,1,y.shape[0])
  
# Run the Control loop:
t0 = default_timer()
    
# Get the forcing function:
Force  = env.force_fun

D0n, V0n, A0n = env.reset()             # Initial condition    
uopt0  = np.zeros(env.action_dim)  
rHistory = env.D_ref  
xH, xdH, xddH = D0n, V0n, A0n           # Stores state history
uH = utils_mpc.store_action(uopt0, env.Ne) # Stores control history
fH = utils_mpc.store_force(Force, 0*dt, env.L, env.Ne) 
tH = 0                    # Stores time history
        
for run in range(int(CT/Ts) - 1):
    t1 = default_timer()

    # action = grf(x)
    action = uHistory[:, run+1]
    uopt0 = action[32:32*3]
    
    D0, V0, A0 = env.step(states=(D0n,V0n,A0n), t=run, action=uopt0)
    D0n, V0n, A0n = D0, V0, A0
            
    error_d, error_v = utils_mpc.L2norm(D0n[::2], V0n[::2], rHistory)
    
    t2 = default_timer()
    print('Time step-{}, Time-{:0.4f}, Error: Displacement: {:0.4f}, Velocity: {:0.4f}'.format(
              run, t2-t1, error_d, error_v))
    
    xH = np.column_stack((xH, D0n))
    xdH = np.column_stack((xdH, V0n))
    xddH = np.column_stack((xddH, A0n))
    uH = np.column_stack((uH, utils_mpc.store_action(uopt0, env.Ne) ))
    fH = np.column_stack((fH, utils_mpc.store_force(Force, run*dt, env.L, env.Ne) ))
    tH = np.append(tH, run*Ts)
        
tT = default_timer()
print('End of Force loop, Total Time-{:0.4f}min.'.format((tT-t0)/60))

# Concatenate the responses:
xtc = np.stack(( xH, xdH, xddH ))

# %%
# Plot the data for visualization:
fig1, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,8), dpi=100)
plt.subplots_adjust(hspace=0.4)

label = ['$u(t)$', '$\dot{u}(t)$', '$\ddot{u}(t)$']
for i in range(3):
    im = ax[i].imshow(xtc[i, ::2].T, cmap='jet', origin='lower', aspect='auto'); 
    ax[i].set_ylabel(label[i]); 
    plt.colorbar(im, ax=ax[i], pad=0.01)
plt.suptitle('Controlled Response')
plt.show()

# %%
fig3 = plt.figure(figsize = (10, 6), dpi=100)
plt.plot(tH, uH[50, :], label='Control-$u(t)$')
plt.plot(tH, fH[0, :], label='Excitation-$F(t)$')
plt.plot(tH, xH[-1, :], label='$x(t)$ at free end')
plt.ylabel('State and control force')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

# %%
# Save the results,
# sio.savemat('results/beam_mpc_data.mat', mdict={'xH':xt,
#                                                 'xHc':xtc,
#                                                 'uH':uH,
#                                                 'fH':fH,
#                                                 'tH':tH,
#                                                 'ref':rHistory})

