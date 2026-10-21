#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for generating response for the 76 DOF structure
"""

import scipy.io as sio
import numpy as np
from numpy.linalg import inv
from scipy.linalg import eigh
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
eigval, eigvec = eigh(K76, M76)     # real eigenvalues in ascending order

# Get the first modal mass:
idx = 0
phi = normalize(eigvec)
modal_m = (phi[:,idx].T @ M76 @ phi[:,idx]).squeeze()

# Modal mass ratio:
md = 500
mu = md/modal_m  # mass ratio

# Optimum damper parameters:
fopt = 1/(1+mu)
zopt = np.sqrt((3*mu)/(8*(1+mu)))

# Find the optimum stiffness and damping
omega = np.sqrt(eigval)             # natural frequencies (rad/s)
kd = fopt**2 * omega[idx]**2 * md
cda = 2 * zopt * fopt * omega[idx] * md

M77 = np.zeros([77,77])
K77 = np.zeros([77,77])
C77 = np.zeros([77,77])
M77[:76,:76] = M76
M77[-1,-1] = md

K77[:76,:76] = K76
K77[-2,-2] = K76[-1,-1] + kd
K77[-2,-1] = -kd
K77[-1,-2] = -kd
K77[-1,-1] = kd

C77[:76,:76] = C76
C77[-2,-2] = C76[-1,-1] + cda
C77[-2,-1] = -cda
C77[-1,-2] = -cda
C77[-1,-1] = cda

def solve_76dof(M,C,K,alpha,nl_idx,ic,T,dt,action):
	# Statespace matrix
	sys = np.vstack(( np.column_stack(( np.zeros(M.shape),
                                       np.eye(M.shape[0]) )), \
                      np.column_stack(( -np.matmul(np.linalg.inv(M),K),
                                        -np.matmul(np.linalg.inv(M),C) )) ))

	def dydt(x,t,statespace,alpha,nl_idx,iM_nl,force):
		dof = len(x)//2

		# Nonlinearity (cubic springs at nl_idx, mapped by M^-1)
		f_nl = alpha * x[nl_idx]**3

		# Evaluate the derivatives; 'force' is M^-1 (wind + control force)
		dydt = np.matmul(statespace,x)
		dydt[dof:] += force - iM_nl @ f_nl
		return dydt

	# Wind load calculation:
	L = 42
	u_ref = 13.5                    # mean wind speed at z_ref = 10 m (m/s)
	I_u = 0.2                       # turbulence intensity
	f_w = omega[0]/2/np.pi          # gust frequency = first-mode frequency (Hz)
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

	z_ref = 10                      # reference height (m)
	h = np.gradient(z)              # tributary height of each floor (m)

	# Define the forcing function (kN, consistent with tonnes and kN/m):
	fun = lambda rho, L, h, cd, uv, z_ref, z: 0.5*rho*L*h*cd*uv**2*(z/z_ref)**0.365 / 1000

	# Compressing the forcing function:
	intensity = partial(fun, rho=rho, L=L, h=h, cd=cd, z_ref=z_ref, z=z)

	# Define the excitation influence matrix:
	dof = M.shape[0]
	sigma = np.eye(dof)
	sigma[-1,-1] = 0

	# Inverse of mass matrix
	iM = inv(M)

	# Define control influence matrix:
	c_idx = np.array([5, 10, 15, 25, 35, 45, 55, 60, 65, 72, 75])
	H = np.zeros((77, len(c_idx)))                      # Influence matrix
	H[(c_idx, np.arange(len(c_idx)))] = 1

	# Precompute the load matrices mapped by M^-1:
	alpha = np.asarray(alpha, dtype=float)
	iM_w = iM @ sigma               # wind
	iM_h = iM @ H                   # control
	iM_nl = iM[:, nl_idx]           # nonlinear forces

	# Time integration:
	t_eval = np.arange(0, T+dt, dt)
	xt = np.zeros((len(ic), len(t_eval)))
	xt[:, 0] = ic
	for i in range(len(t_eval)-1):
		ft = np.concatenate(( intensity(uv = u_ref * (1 + I_u*np.sin(2*np.pi*f_w*i*dt))), np.zeros(1) ))
		force = iM_w @ ft + iM_h @ action

		sol = odeint(func=dydt, t=[i*dt, i*dt+dt], y0=ic, mxstep=1000, args=(sys,alpha,nl_idx,iM_nl,force))
		ic = sol[-1,:]
		xt[:, i+1] = ic

	# Extract the displacement and velocities:
	states = len(ic)//2
	D = xt[:states, :-1]
	V = xt[states:, :-1]

	# Compute acceleration response:
	A = utils.FiniteDiff_vec(V, dx=dt, d=1) # along time, for all DOFs

	return t_eval[:-1], np.stack((D,V,A))


# Configure the system parameters:
T = 60
dt = 0.01
record_dt = 0.01
dof = 77
alpha = [4.6e7, 3.7e6, 3.7e5]
nl_idx = [35, 52, 75]

# Initial condition in the full (physical) coordinates:
D0 = 0.0 * np.concatenate((np.linspace(0,1,dof-1), np.zeros(1)))
V0 = np.zeros(dof)
x0 = np.concatenate((D0, V0))

# Generate the response:
t, xt = solve_76dof(M77, C77, K77, alpha, nl_idx, ic=x0, T=T, dt=dt, action=np.zeros(11))
t, xt = t[::int(record_dt/dt)], xt[..., ::int(record_dt/dt)]

# burn_steps = int(0.25/record_dt)
# t, xt = t[burn_steps:], xt[..., burn_steps:]

# %%
""" RMS of the response and degree of nonlinearity """
rms = lambda y: np.sqrt(np.mean(y**2, axis=-1))
alpha = np.asarray(alpha, dtype=float)

# Degree of nonlinearity: cubic secant stiffness (alpha x^2) relative to the
# building's point stiffness at each nonlinear DOF (k_i = 1/[K^-1]_ii):
k_point = 1/np.diag(inv(K77))[nl_idx]
print('Nonlinearity  alpha*x_rms^2 / k_point :', np.round(alpha*rms(xt[0, nl_idx])**2/k_point, 4))

# Compare against the linear system (alpha = 0):
compare_linear = True
if compare_linear:
	_, xt_lin = solve_76dof(M77, C77, K77, 0*alpha, nl_idx, ic=x0, T=T, dt=dt, action=np.zeros(11))
	xt_lin = xt_lin[..., ::int(record_dt/dt)]

for name, k in zip(['Displacement', 'Velocity', 'Acceleration'], range(3)):
	print('\nRMS {}:'.format(name))
	for d in nl_idx:
		line = '  DOF {:2d}: nonlinear = {:.4e}'.format(d, rms(xt[k, d]))
		if compare_linear:
			line += ', linear = {:.4e}, change = {:+.2f} %'.format(
                rms(xt_lin[k, d]), 100*(rms(xt[k, d]) - rms(xt_lin[k, d]))/rms(xt_lin[k, d]))
		print(line)

# %%
floor = [5,25,70,74]

figure1, ax = plt.subplots(nrows=len(floor), ncols=3, figsize =(16,10), dpi=100)
plt.subplots_adjust(hspace=0.35, wspace=0.35)

for i in range(len(floor)):
	ax[i,0].plot(t, xt[0, floor[i],:], label='DOF:{}th'.format(floor[i]))
	if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
	ax[i,0].set_ylabel('$X({})$'.format(floor[i]))
	ax[i,0].grid(True, alpha=0.25)

	ax[i,1].plot(t, xt[1, floor[i],:], label='DOF:{}th'.format(floor[i]))
	if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
	ax[i,1].set_ylabel(r'$\dot{{X}}({})$'.format(floor[i]))
	ax[i,1].grid(True, alpha=0.25)

	ax[i,2].plot(t, xt[2, floor[i],:], label='DOF:{}th'.format(floor[i]))
	if i == len(floor)-1: ax[i,2].set_xlabel('Time (Sec)')
	ax[i,2].set_ylabel(r'$\ddot{{X}}({})$'.format(floor[i]))
	ax[i,2].grid(True, alpha=0.25)

plt.suptitle('Response of 76 DOF Skyscraper', y=0.94)
plt.show()

