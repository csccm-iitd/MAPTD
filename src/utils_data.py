#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified script for generating data
"""

import numpy as np
import beam_solver
import scipy.io as sio
from functools import partial
from scipy.integrate import odeint
from numpy.linalg import inv
from scipy.linalg import eigh
import utils

"""
Codes for free vibration of a cantilever
"""
def cantilever(sparams,tparams,force):
	rho, E, b, d, A, I, L, mode, alpha, beta = sparams   # alpha, beta: Rayleigh damping
	Ne, T, dt = tparams
	t_eval = dt*np.arange(int(round(T/dt)))
	xx = (L/Ne)*np.arange(1, Ne+1)

	[Ma, Ka, _, _] = beam_solver.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')

	Ca = (alpha*Ma + beta*Ka)
	F = force

	Lambda = np.array([1.875104069, 4.694091133, 7.854757438, 10.99554073,
                       14.13716839, 17.27875953])/L

	h1 = np.cosh(Lambda[mode]*xx) -np.cos(Lambda[mode]*xx) -(np.cos(Lambda[mode]*L)+np.cosh(Lambda[mode]*L)) \
        /(np.sin(Lambda[mode]*L)+np.sinh(Lambda[mode]*L))*(np.sinh(Lambda[mode]*xx)-np.sin(Lambda[mode]*xx))
	h2 = Lambda[mode]*(np.sinh(Lambda[mode]*xx)+np.sin(Lambda[mode]*xx))-(np.cos(Lambda[mode]*L)+np.cosh(Lambda[mode]*L)) \
        /(np.sin(Lambda[mode]*L)+np.sinh(Lambda[mode]*L))*(np.cosh(Lambda[mode]*xx)-np.cos(Lambda[mode]*xx))*Lambda[mode]

	D0 = np.zeros(2*Ne)
	D0[0::2] = h1
	D0[1::2] = h2
	V0 = np.zeros(2*Ne)
	xt = beam_solver.Newmark(Ma,Ca,Ka,F,D0,V0,dt,T)

	Dis = xt[0, 0::2]
	Vel = xt[1, 0::2]
	Acc = xt[2, 0::2]
	return t_eval, np.stack(( Dis, Vel, Acc ))


"""
Code to solve the 76 DOF structure
"""
def normalize(mode):
	nmode = np.zeros(mode.shape)
	for i in range(len(mode[0])):
		nmode[:,i] = mode[:,i]/mode[-1,i]
	return nmode

# The 76 storey building properties
def skyscraper():
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

	# Tuned-mass augmented system:
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

	return M77,K77,C77,id_

def solve_76dof(M,C,K,ic,T,dt,alpha=(4.6e7, 3.7e6, 3.7e5),nl_idx=(35, 52, 75),
                u_ref=13.5,I_u=0.2,f_w=None):
	"""
    Response of the TMD-augmented 76 DOF building under along-wind load.
    Units: tonnes, kN/m, kN.
    
    alpha  : cubic stiffness at each DOF in nl_idx, f_nl = alpha * x^3 (alpha > 0: hardening)
    u_ref  : mean wind speed at z_ref = 10 m (m/s); u_ref = 0 gives free vibration
    I_u    : turbulence intensity, u(t) = u_ref * (1 + I_u*sin(2*pi*f_w*t))
    f_w    : gust frequency (Hz); defaults to the first natural frequency
    """
	dof = M.shape[0]
	alpha = np.asarray(alpha, dtype=float)
	nl_idx = list(nl_idx)

	# Inverse of mass matrix
	iM = inv(M)

	# Statespace matrix
	sys = np.vstack(( np.column_stack(( np.zeros(M.shape), np.eye(dof) )), \
                      np.column_stack(( -iM @ K, -iM @ C )) ))

	# Wind load calculation:
	L = 42
	z_ref = 10                      # reference height (m)
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
	h = np.gradient(z)              # tributary height of each floor (m)

	if f_w is None:
		f_w = np.sqrt(eigh(K[:-1,:-1], M[:-1,:-1], eigvals_only=True)[0])/2/np.pi   # building without TMD

	# Define the forcing function (kN, consistent with tonnes and kN/m):
	fun = lambda rho, L, h, cd, uv, z_ref, z: 0.5*rho*L*h*cd*uv**2*(z/z_ref)**0.365 / 1000

	# Compressing the forcing function:
	intensity = partial(fun, rho=rho, L=L, h=h, cd=cd, z_ref=z_ref, z=z)
	wind = lambda t: np.concatenate(( intensity(uv = u_ref*(1 + I_u*np.sin(2*np.pi*f_w*t))), np.zeros(1) ))

	# Define the excitation influence matrix (no wind on the TMD):
	sigma = np.eye(dof)
	sigma[-1,-1] = 0

	# Precompute the load matrices mapped by M^-1:
	iM_w = iM @ sigma               # wind
	iM_nl = iM[:, nl_idx]           # nonlinear forces

	def dydt(x,t,statespace,alpha,nl_idx,iM_w,iM_nl):

		# Nonlinearity (cubic springs at nl_idx)
		f_nl = alpha * x[nl_idx]**3

		# Evaluate the derivatives; wind and nonlinear forces mapped by M^-1
		dydt = np.matmul(statespace,x)
		dydt[dof:] += iM_w @ wind(t) - iM_nl @ f_nl
		return dydt

	# Solve the system:
	t_eval = np.arange(0,T+dt,dt)
	sol = odeint(func=dydt, y0=ic, t=t_eval, mxstep=1000, args=(sys,alpha,nl_idx,iM_w,iM_nl))
	sol = sol.transpose(1,0)

	# Wind force history (kN):
	ft = np.column_stack(( [wind(t) for t in t_eval] ))[:, :-1]

	# Extract the displacement and velocities:
	states = len(ic)//2
	D = sol[:states, :-1]
	V = sol[states:, :-1]

	A = utils.FiniteDiff_vec(V, dx=dt, d=1) # along time, for all DOFs
	return t_eval[:-1], np.stack(( D,V,A )), ft
