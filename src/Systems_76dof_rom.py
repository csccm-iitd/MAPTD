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
from scipy.linalg import eigh
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

id_ = [2, 5, 9, 12, 15, 19, 22, 25, 29, 32, 35, 39,
        42, 45, 49, 52, 55, 59, 62, 65, 69, 72, 75, 76]

# Perform Model-order reduction:
def rom(m,k,c,order=None,mass_ratio=0.9):
	"""
    Reduced-order model x = T q, with  m_rom = T' m T,  k_rom = T' k T,  c_rom = T' c T.

    order : list -> DOFs (floors) to retain; Guyan (static) condensation, q = x[order]
            int  -> lowest 'order' modes; mass-normalised modal truncation, m_rom = I
            None -> lowest modes whose cumulative effective (lateral) mass >= mass_ratio
    Returns (m_rom, k_rom, c_rom, T); T is (n x r).
    """
	n = m.shape[0]

	# Model-order reduction equation:
	rom_system = lambda matrix, phi: np.matmul( np.matmul(phi.T, matrix), phi )

	if isinstance(order, (list, tuple, np.ndarray)):
		# Guyan condensation: slave DOFs follow the master DOFs statically
		master = np.asarray(order)
		slave = np.setdiff1d(np.arange(n), master)
		vec = np.zeros((n, len(master)))
		vec[master, np.arange(len(master))] = 1
		vec[slave] = -np.linalg.solve(k[np.ix_(slave, slave)], k[np.ix_(slave, master)])

	else:
		# Mass-normalised modes in ascending order of frequency (phi' m phi = I):
		eigval, eigvec = eigh(k, m)

		# If order is not given, use the modal mass participation criterion:
		if order is None:
			r = np.ones(n)                                  # lateral influence vector
			eff_mass = (eigvec.T @ m @ r)**2 / (r @ m @ r)
			order = int(np.searchsorted(np.cumsum(eff_mass), mass_ratio) + 1)

		vec = eigvec[:, :order]

	# Get the reduced-order matrices:
	m_rom = rom_system(m, vec)
	k_rom = rom_system(k, vec)
	c_rom = rom_system(c, vec)
	return (m_rom, k_rom, c_rom, vec)


# %%
def solve_76dof(M,C,K,alpha,nl_idx,ic,T,dt,action,eigvec,mt,ct=None,kt=None,method='rk4',substeps=None):
	"""
    Reduced-order response, x = eigvec @ q:
        M q'' + C q' + K q = eigvec' (F_wind + H a - f_nl(eigvec @ q))
    M, C, K : reduced matrices from rom();  mt : full mass matrix (sets the full dimension)
    method  : 'rk4' (fixed-step Runge-Kutta, wind and action held over each dt) or 'odeint'
    substeps: RK4 substeps per dt; default from the stability limit, omega_max * dt/substeps <= 2
    Returns the response in the full (physical) coordinates.
    """
	dof = mt.shape[0]           # full dimension
	r = M.shape[0]              # reduced dimension
	alpha = np.asarray(alpha, dtype=float)

	# Inverse of the reduced mass matrix and the force projection, q'' = iM T' F:
	iM = inv(M)
	P = iM @ eigvec.T

	# Statespace matrix
	sys = np.vstack(( np.column_stack(( np.zeros(M.shape), np.eye(r) )), \
                      np.column_stack(( -iM @ K, -iM @ C )) ))

	def dydt(x,t,statespace,alpha,T_nl,P_nl,force):

		# Nonlinearity (cubic springs on the physical DOFs, then projected)
		f_nl = alpha * (T_nl @ x[:r])**3

		# Evaluate the derivatives; 'force' is the projected wind + control force
		dydt = np.matmul(statespace,x)
		dydt[r:] += force - P_nl @ f_nl
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

	# Define the excitation influence matrix (no wind on the TMD):
	sigma = np.eye(dof)
	sigma[-1,-1] = 0

	# Define control influence matrix:
	c_idx = np.array([5, 10, 15, 25, 35, 45, 55, 60, 65, 72, 75])
	H = np.zeros((dof, len(c_idx)))                     # Influence matrix
	H[(c_idx, np.arange(len(c_idx)))] = 1

	# Precompute the projected load matrices (reduced dimension):
	Pw = P @ sigma                  # wind
	Ph = P @ H                      # control
	T_nl = eigvec[nl_idx, :]        # physical displacement at the nonlinear DOFs
	P_nl = P[:, nl_idx]             # projection of the nonlinear forces
	args = (sys,alpha,T_nl,P_nl)

	# RK4 substeps from the stability limit of the highest retained frequency:
	if method == 'rk4' and substeps is None:
		omega_max = np.sqrt(eigh(K, M, eigvals_only=True)[-1])
		substeps = max(1, int(np.ceil(omega_max * dt / 2.0)))

	# Time integration:
	t_eval = np.arange(0, T+dt, dt)
	xt = np.zeros((2*r, len(t_eval)))
	xt[:, 0] = ic
	x = np.array(ic, dtype=float)
	for i in range(len(t_eval)-1):
		ft = np.concatenate(( intensity(uv = u_ref * (1 + I_u*np.sin(2*np.pi*f_w*i*dt))), np.zeros(1) ))
		force = Pw @ ft + Ph @ action

		if method == 'rk4':
			hs = dt / substeps
			for _ in range(substeps):
				k1 = dydt(x, 0, *args, force)
				k2 = dydt(x + 0.5*hs*k1, 0, *args, force)
				k3 = dydt(x + 0.5*hs*k2, 0, *args, force)
				k4 = dydt(x + hs*k3, 0, *args, force)
				x = x + hs/6 * (k1 + 2*k2 + 2*k3 + k4)
		else:
			sol = odeint(func=dydt, t=[i*dt, i*dt+dt], y0=x, mxstep=1000, args=(*args, force))
			x = sol[-1,:]
		xt[:, i+1] = x

	# Extract the displacement and velocities in physical coordinates:
	D = eigvec @ xt[:r, :-1]
	V = eigvec @ xt[r:, :-1]

	# Compute acceleration response:
	A = utils.FiniteDiff_vec(V, dx=dt, d=1) # along time, for all DOFs

	return t_eval[:-1], np.stack(( D,V,A ))


# %%
# Configure the system parameters:
T = 60
dt = 0.01
record_dt = 0.01
dof = 77            # full-model dimension (76 floors + TMD)
alpha = [4.6e7, 3.7e6, 3.7e5]   # cubic stiffness at nl_idx (kN/m^3)
nl_idx = [35, 52, 75]
order = id_         # list: floors kept (Guyan), int: number of modes, None: mass participation

# Initial condition in the full (physical) coordinates:
D0 = 0.0 * np.concatenate((np.linspace(0,1,dof-1), np.zeros(1)))
V0 = np.zeros(dof)

# Get the reduced-order model:
m_rom, k_rom, c_rom, vec = rom(m=M77,k=K77,c=C77,order=order)
states = vec.shape[1]   # ROM-states dimension
print('ROM dimension: {}'.format(states))

# Project the initial condition onto the reduced coordinates:
if isinstance(order, (list, tuple, np.ndarray)):
	# Guyan: reduced coordinates are the displacements of the retained floors
	D0, V0 = D0[order], V0[order]
else:
	# Mass-normalised modes: q = phi' M x
	D0, V0 = vec.T @ M77 @ D0, vec.T @ M77 @ V0
x0 = np.concatenate((D0, V0))

# Generate the response:
t, xt = solve_76dof(m_rom, c_rom, k_rom, alpha, nl_idx, ic=x0, T=T, dt=dt, action=np.zeros(11),
                    eigvec=vec, mt=M77, method='odeint')
t = t[::int(record_dt/dt)]
xt = xt[:,:,::int(record_dt/dt)]


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

