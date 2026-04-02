#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:27:29 2024

@author: tripu
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

import scipy.io as sio
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.animation as animation

import env_beam
import utils_mpc

plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %%
""" Generating system response """
# The time parameters: 
T, dt = 5, 1e-2
Nt = int(T/dt)
env = env_beam.EulerBeam(T=T, dt=dt, forced=True)

# Solve the system:
xt = env.solve()

# %% Plot the data for visualization:
fig1, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,6), dpi=100)
plt.subplots_adjust(hspace=0.3)

label = ['$u(t)$', '$\dot{u}(t)$', '$\ddot{u}(t)$']
for i in range(3):
    im = ax[i].imshow(xt[i, ::2].T, extent=[0,env.L,0,T], cmap='jet', origin='lower', aspect='auto'); 
    ax[i].set_ylabel(label[i]); 
    plt.colorbar(im, ax=ax[i], pad=0.01)
plt.show()

# %% Run for different prediction horizon's:
t0 = default_timer()

print('Horizon Length-{}'.format(env.phorizon))
Ts     = 0.01               # Sampling time
start  = 0.0
CT     = 0.1
Force  = env.force_fun

D0n, V0n, A0n = env.reset()             # Initial condition
rHistory = env.D_ref
uopt0  = np.zeros(env.phorizon * env.action_dim)  
bounds = Bounds(env.LB, env.UB)
   
xHistory = D0n                                # Stores state history
xdHistory = V0n                               # Stores state history
xddHistory = A0n                              # Stores state history
uHistory = utils_mpc.store_action(uopt0[:env.action_dim], env.Ne) # Stores control history
fHistory = utils_mpc.store_force(Force, 0*dt, env.L, env.Ne)
tHistory = 0                    # Stores time history

time = []
for run in range(int(CT/Ts) - 1):
    t1 = default_timer()
    if run*Ts > start:            # Turn control on
        if run*Ts == start+Ts:
            print('\n~~~~~~~~~~~ Control is activated ~~~~~~~~~~~\n')
        
        OBJFUN = lambda u: env.cost(action=u, states=(D0n, V0n, A0n), t=run,
                                    old_action=uopt0[:env.action_dim])
        res = minimize(OBJFUN, uopt0, method='SLSQP', jac="2-point", bounds=bounds)
        uopt0 = res.x
    else:
        uopt0 = np.zeros(env.phorizon * env.action_dim)
    
    D0n, V0n, A0n = env.step(states=(D0n,V0n,A0n), t=run, action=uopt0[:env.action_dim])
    
    error_d, error_v = utils_mpc.L2norm(D0n[::2], V0n[::2], rHistory)
    
    t2 = default_timer()
    time.append(t2-t1)
    print('Iteration-{}, Time-{:0.4f}, Error: Displacement: {:0.4f}, Velocity: {:0.4f}'.format(
            run, t2-t1, error_d, error_v))
    
    # Store the history:
    xHistory = np.column_stack((xHistory, D0n))
    xdHistory = np.column_stack((xdHistory, V0n))
    xddHistory = np.column_stack((xddHistory, A0n))
    uHistory = np.column_stack((uHistory, utils_mpc.store_action(uopt0[:env.action_dim], env.Ne) ))
    fHistory = np.column_stack((fHistory, utils_mpc.store_force(Force, run*dt, env.L, env.Ne) ))
    tHistory = np.append(tHistory, run*Ts)

tT = default_timer()
print('-- Run Complete -- Total Time-{}min.'.format((tT-t0)/60))

# Concatenate the responses:
xtHistory = np.stack(( xHistory, xdHistory, xddHistory ))
ppp

# %%
# Plot the results
fig2, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,6), dpi=100)
plt.subplots_adjust(hspace=0.3)

label = ['$u(t)$', '$\dot{u}(t)$', '$\ddot{u}(t)$']
for i in range(3):
    im = ax[i].imshow(xtHistory[i, ::2, :].T, extent=[0,1,0,T], cmap='jet', origin='lower', aspect='auto'); 
    ax[i].set_ylabel(label[i]); 
    ax[i].set_xlabel('Space')
    plt.colorbar(im, ax=ax[i], pad=0.01)
plt.show()

# fig2.savefig('Figures/Beam_control_contour.png', format='png', dpi=300, bbox_inches='tight')

# %%
fig3 = plt.figure(figsize = (10, 6), dpi=100)
plt.plot(tHistory, uHistory[32, :], label='Control-$u(t)$')
plt.plot(tHistory, fHistory[0, :], label='Excitation-$F(t)$')
plt.plot(tHistory, xHistory[-1, :], label='$x(t)$ at free end')
plt.ylabel('State and control force')
plt.xlabel('Time (s)')
plt.axvline(x=start, color='r', linestyle=':')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

# fig3.savefig('Figures/Beam_control_tip_displacement.png', format='png', dpi=300, bbox_inches='tight')

# %%
# Save the results,
# sio.savemat('results/beam_mpc2.mat', mdict={'xt':xt,
#                                            'xtHistory':xtHistory,
#                                            'uHistory':uHistory,
#                                            'fHistory':fHistory,
#                                            'tHistory':tHistory,
#                                            'rHistory':rHistory,
#                                            'time':(tT-t0)})
sio.savemat('results/beam_mpc_time.mat', mdict={'time':time}) 

# %%
# # 2D animation:
# nframes = xHistory.shape[-1]

# fig8, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), dpi=100)
# def update(x,y,ref,k):
#     print('Animation Generating, Frame-{}'.format(k))
#     ax.cla()
#     ax.plot(x,y, color='k', label='$u(t)$')
#     ax.set_xlabel('Time (s)', labelpad = 20, fontweight='bold');
#     ax.set_ylabel('u(x,t)', labelpad = 10, fontweight='bold')
#     ax.set_title('EB Beam: $(u_{tt} = (EI)/(\mu) u_{xxxx})$\nTime: %0.2f' % ((k+1)*dt),
#                  fontweight='bold'); 
#     # ax.set_ylim([np.floor(np.min(xHistory)), np.ceil(np.max(xHistory))])
#     ax.set_ylim([-1*np.ceil(np.max(xHistory)), np.ceil(np.max(xHistory))])
#     ax.plot(x,ref, '--', color='r', label='UB')
#     ax.plot(x,-1* ref, '--', color='b', label='LB')
#     ax.legend(loc=2, labelspacing=0.15, handletextpad=0.15)
#     ax.grid(True, alpha=0.35)
#     ax.margins(0)

# def animate_2d(k):
#     update(xgrid, xHistory[:,k], xref, k)
    
# anim = animation.FuncAnimation(fig=fig8, func=animate_2d, interval=1, frames=nframes, repeat=False)
# anim.save("Beam_control_2d.mp4", writer='ffmpeg', fps=30)

# # %%
# # 3D animation using surface plot:
# xx, tt = np.meshgrid(tHistory, xgrid)
# xxr, ttr, xur = xx[:, ::5], tt[:, ::5], xHistory[:, ::5]

# fig9 = plt.figure(figsize=(8,8), dpi=100)
# cmin, cmax = np.min(xHistory), np.max(xHistory)

# ax = plt.axes(projection='3d')

# def update_3d(x,y,z,k):
#     print('3D Animation Generating, Frame-{}'.format(k))
#     ax.plot_surface(x, y, z, cmap='gist_ncar', vmin=cmin, vmax=cmax, antialiased=True)
#     ax.view_init(35,35)
#     ax.set_xlabel('Time (s)', labelpad = 10, fontweight='bold');
#     ax.set_ylabel('Space ($x$)', labelpad = 10, fontweight='bold');
#     ax.set_zlabel('u(x,t)', fontweight='bold')
#     ax.set_title('EB Beam: $(u_{tt} = (EI)/(\mu) u_{xxxx})$\nTime: %0.2f' % ((k+1)*dt),
#                  fontweight='bold', y=1); 

# def animate(k):
#     update_3d(xxr[:, :k], ttr[:, :k], xur[:, :k], k)
    
# anim = animation.FuncAnimation(fig=fig9, func=animate, interval=1, frames=xxr.shape[-1], repeat=False)
# anim.save("Beam_control_3d.mp4", writer='ffmpeg', fps=30)

