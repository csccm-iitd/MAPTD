# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). "Model-agnostic stochastic predictive
    control using limited and output only measurements".
   
This code is for the control of 76-DOF system.
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
from timeit import default_timer

import scipy.io as sio
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

import env_skyscraper
import utils_mpc

# %% Response simulation 
T, dt = 4, 1e-2
Nt = int(T/dt)
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

# %% Run for different prediction horizon's:
t0 = default_timer()

Ts     = 0.01             # Sampling time
start  = 0.1
CT     = 0.5                # Run control for 100 time units
Force  = env.force_fun
Intensity = env.intensity

D0n, V0n, A0n = env.reset()             # Initial condition
uopt0  = np.zeros(env.phorizon * env.action_dim)  
bounds = Bounds(env.LB, env.UB)
   
xHistory = D0n                                # Stores state history
xdHistory = V0n                               # Stores state history
xddHistory = A0n                              # Stores state history
uHistory = uopt0[:env.action_dim]             # Stores control history
fHistory = Intensity(u_ref = Force(0*dt))
tHistory = 0                                  # Stores time history
rHistory = env.D_ref                          # Stores reference

time = []
for run in range(int(CT/Ts) - 1):
    t1 = default_timer()
    if run * Ts > start: # control at every 10 step,
        if run*Ts == start+Ts:
            print('\n~~~~~~~~~~~ Control is activated ~~~~~~~~~~~\n')
        
        OBJFUN = lambda u: env.cost(action=u, states=(D0n, V0n, A0n), t=run,
                                    old_action=uopt0[:env.action_dim])
        res = minimize(OBJFUN, uopt0, method='SLSQP', jac="2-point", bounds=bounds)
        uopt0 = res.x
    else:
        uopt0 = np.zeros(env.phorizon * env.action_dim)
    
    D0n, V0n, A0n = env.step(states=(D0n,V0n,A0n), t=run, action=uopt0[:env.action_dim])
    
    error_d, error_v = utils_mpc.L2norm(D0n[-3:-1], V0n[-3:-1], rHistory[-3:-1])
    
    t2 = default_timer()
    time.append(t2-t1)
    print('Iteration-{}, Time-{:0.4f}, Error: Displacement: {:0.4f}, Velocity: {:0.4f}'.format(
            run, t2-t1, error_d, error_v))
    
    # Store the history:
    xHistory = np.column_stack((xHistory, D0n))
    xdHistory = np.column_stack((xdHistory, V0n))
    xddHistory = np.column_stack((xddHistory, A0n))
    uHistory = np.column_stack((uHistory, uopt0[:env.action_dim]))
    fHistory = np.column_stack((fHistory, Intensity(u_ref = Force(run*dt)) ))
    tHistory = np.append(tHistory, run*Ts)

tT = default_timer()
print('-- Run Complete -- Total Time-{}min.'.format((tT-t0)/60))

# Concatenate the responses:
xtHistory = np.stack(( xHistory, xdHistory, xddHistory ))

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16

fig3, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,8), dpi=100,
                        gridspec_kw={'height_ratios':[1,1,0.5]})

ax[0].plot(tHistory, xHistory[-3, :], label='Controlled')
ax[0].plot(tHistory, xt[0,-3,:], label='True')
ax[0].set_ylabel('$x(t)$ at DOF-76')

for j in range(env.action_dim):
    ax[1].plot(tHistory, uHistory[j, :], label='@DOF-{}'.format(j+1))
ax[1].axvline(x=start, color='r', linestyle=':', label='Start time')
ax[1].set_ylabel('Action-$u(t)$')

ax[2].plot(tHistory, fHistory[-2, :])
ax[2].set_ylabel('Excitation-$F(t)$ \n at DOF-76')

for i in range(3):
    if i < 2: ax[i].legend(loc=2, labelspacing=0.1, handletextpad=0.5, handlelength=1, borderaxespad=0.1)
    if i == 2: ax[i].set_xlabel('Time (s)')
    ax[i].grid(True, alpha=0.5)
plt.show()

# fig3.savefig('Figures/Structural_control.png', format='png', dpi=300, bbox_inches='tight')

# %%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

floor = [4,13,17,22]
    
t = np.arange(0,1,0.01)

figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35)

for i in range(len(floor)):
    ax[i,0].plot(xtHistory[0, floor[i], :], color='r', label='Controlled')
    ax[i,0].plot(xt[0, floor[i], :], linestyle='--', label='True')
    if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
    ax[i,0].set_ylabel('DOF-{}'.format(floor[i]))
    ax[i,0].grid(True, alpha=0.25) 
    ax[i,0].legend()
    
    ax[i,1].plot(xtHistory[1, floor[i], :], color='r', label='Controlled')
    ax[i,1].plot(xt[1, floor[i], :], linestyle='--', label='True')
    if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
    ax[i,1].set_ylabel('DOF-{}'.format(floor[i]))
    ax[i,1].grid(True, alpha=0.25) 
    ax[i,1].legend()

ax[0,0].set_title('Displacement')
ax[0,1].set_title('Velocity')
plt.suptitle('Response of 76 DOF Skyscraper (RL ENV)', y=0.94)
plt.show()

# figure1.savefig('Figures/Controlled_response_76dof.png', format='png', dpi=100, bbox_inches='tight')

# %%
# Save the results,
# sio.savemat('results/structure_mpc.mat', mdict={'xt':xt,
#                                                 'xtHistory':xtHistory,
#                                                 'uHistory':uHistory,
#                                                 'fHistory':fHistory,
#                                                 'tHistory':tHistory,
#                                                 'rHistory':rHistory,
#                                                 'time':(tT-t0)})
sio.savemat('results/str_mpc_time.mat', mdict={'time':time}) 

