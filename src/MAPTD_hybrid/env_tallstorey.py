# -*- coding: utf-8 -*-
"""
This script defines the environment for the Skyscraper
"""

from functools import partial
import numpy as np
import scipy.io as sio
from numpy.linalg import inv
from scipy.linalg import eigh
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from dm_env import specs, TimeStep, StepType
from collections import OrderedDict
from collections import defaultdict

"""
Codes for free vibration of a cantilever
"""
class Skyscraper:
	def __init__(self, LB=None, UB=None, forced=True, action_repeat=1):

		# Initialize the scyscraper properties
		self.forced = forced
		self.action_repeat = action_repeat     # env steps per agent step in one_step (planner model)
		self.building()
		self.windpressure()

		self.action_dim = len(self.c_idx)
		self.D_ref = np.zeros(self.dof)    # reference: building at rest
		self.V_ref = np.zeros(self.dof)
		self.A_ref = np.zeros(self.dof)
		self.Qx = np.eye(self.dof)
		self.Qx[-1, -1] = 0     # don't penalise the TMD's own motion
		self.Qu = [12, 17, 15]                                # displacement, velocity, acceleration
		self.Ru = 1e-7 * np.eye(self.action_dim)             # control, u in kN

		# Normalised actions in [-1, 1]; the environment applies u = u_max * action (kN)
		self.u_max = 250
		self.LB = LB if LB is not None else -1*np.ones(self.action_dim)
		self.UB = UB if UB is not None else 1*np.ones(self.action_dim)

	def action_spec(self):
		return specs.BoundedArray(shape=self.LB.shape,
			   dtype=self.LB.dtype,
			   minimum=self.LB,
			   maximum=self.UB,
			   name='action')

	def observation_spec(self):
		state = specs.Array(shape=(2*self.dof,), dtype=self.LB.dtype, name='states')

		dic = OrderedDict([('states', state)])
		return dic

	def normalize(self,mode):
		# Normalizes the mode shapes of a structure
		# By deafult, it normalizes with respect to the top floor
		nmode = np.zeros(mode.shape)
		for i in range(len(mode[0])):
			nmode[:,i] = mode[:,i]/mode[-1,i]
		return nmode

	def building(self):
		# The 76 storey building properties (units: tonnes, kN/m, kN)
		data = sio.loadmat('data/B76_inp.mat')
		M76 = data['M76']
		K76 = data['K76']
		C76 = data['C76']

		# Get the natural frequencies and modes:
		eigval, eigvec = eigh(K76, M76)     # real eigenvalues in ascending order

		# Get the first modal mass:
		idx = 0
		phi = self.normalize(eigvec)
		modal_m = (phi[:,idx].T @ M76 @ phi[:,idx]).squeeze()
		self.phi1 = phi[:,idx]              # first-mode shape, 1 at the roof

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
		self.omega = omega

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

		self.M = M77
		self.C = C77
		self.K = K77
		self.dof = self.M.shape[0]

		# Time step
		self.dt = 1e-2
		self.T = 60

		# Inverse of mass matrix
		iM = inv(M77)

		# Statespace matrix
		self.sys = np.vstack(( np.column_stack(( np.zeros(M77.shape), np.eye(self.dof) )), \
                               np.column_stack(( -iM @ K77, -iM @ C77 )) ))

		# Nonlinearity: cubic springs f_nl = alpha * x^3 at nl_idx (alpha > 0: hardening)
		self.alpha = np.array([4.6e7, 3.7e6, 3.7e5])     # kN/m^3
		self.nl_idx = [35, 52, 75]

		# Define the excitation influence matrix (no wind on the TMD):
		sigma = np.eye(self.dof)
		sigma[-1,-1] = 0

		# Define control influence matrix (actuator DOFs, 0-based):
		# self.c_idx = np.concatenate(( np.arange(5,30,10), np.arange(30,75,5), np.array([72, 74, 76]) )) - 1
		self.c_idx = np.array([5, 10, 15, 25, 35, 45, 55, 60, 65, 72, 75])

		H = np.zeros((self.dof, len(self.c_idx)))           # Influence matrix
		H[(self.c_idx, np.arange(len(self.c_idx)))] = 1

		self.sigma = sigma
		self.H = H

		# Precompute the load matrices mapped by M^-1:
		self.iM_w = iM @ sigma              # wind
		self.iM_h = iM @ H                  # control
		self.iM_nl = iM[:, self.nl_idx]     # nonlinear forces


	def windpressure(self):
		# Wind load calculation:
		L = 42          # width of building
		z_ref = 10      # reference height (m)
		rho = 1.22      # air density
		cd = 1.2        # drag coefficient
		z = np.concatenate((
                            np.array([10]),
                            10 + np.linspace(4.5,9,2),
                            10 + 9 + np.linspace(3.9,136.5,35),
                            10 + 9 + 136.5 + np.linspace(4.5,9,2),
                            10 + 9 + 136.5 + 9 + np.linspace(3.9,132.6,34),
                            10 + 9 + 136.5 + 9 + 132.6 + np.linspace(4.5,9,2)
                           ))       # height of the skyscraper
		h = np.gradient(z)          # tributary height of each floor (m)

		# Wind speed history: mean u_ref at z_ref with a gust at the first-mode frequency
		u_ref = 13.5                    # mean wind speed at z_ref = 10 m (m/s)
		I_u = 0.2                       # turbulence intensity
		f_w = self.omega[0]/2/np.pi     # gust frequency (Hz)
		self.force_fun = lambda arg : 0 if self.forced == False else u_ref*(1 + I_u*np.sin(2*np.pi*f_w*arg))

		# Define the forcing function (kN, consistent with tonnes and kN/m):
		fun = lambda rho, L, h, cd, u_ref, z_ref, z: 0.5*rho*L*h*cd*u_ref**2*(z/z_ref)**0.365 / 1000

		# Compressing the forcing function:
		self.intensity = partial(fun, rho=rho, L=L, h=h, cd=cd, z_ref=z_ref, z=z)


	def reset(self):
		# Intial state vector: first-mode shape (0.1 m at the roof), at rest;
		# the TMD starts with the roof, so its spring is unstretched
		D0 = 0.1*np.concatenate(( self.phi1, self.phi1[-1:] ))
		V0 = np.zeros(self.dof)
		A0 = np.zeros(self.dof)

		# Statespace vector containing only diplacement and velocity
		self.x0 = np.concatenate(( D0, V0 ))

		# Store the system matrices:
		self.D = D0
		self.V = V0
		self.A = A0
		self.t = 0
		self.r = 0

		dic = OrderedDict([('states', self.x0)])

		obs = TimeStep(step_type=StepType.FIRST,
                       reward=None,
                       discount=None,
                       observation=dic)
		return obs

	# first-order statespace equation:
	def dydt(self,x,t,statespace,force):
		# Nonlinearity (cubic springs at nl_idx)
		f_nl = self.alpha * x[self.nl_idx]**3

		# Evaluate the derivatives; 'force' is M^-1 (wind + control force)
		dydt = np.matmul(statespace,x)
		dydt[self.dof:] += force - self.iM_nl @ f_nl
		return dydt

	def integrate(self, states, t, action):
		# One time step from time index t, wind and action held over the step
		ft = np.concatenate(( self.intensity(u_ref = self.force_fun(t*self.dt)), np.zeros(1) ))
		force = self.iM_w @ ft + self.iM_h @ action      # 'action' is the control force in kN

		sol = odeint(func=self.dydt, t=[t*self.dt, t*self.dt+self.dt], y0=states, mxstep=1000,
                     args=(self.sys,force))
		x1 = sol[-1,:]

		# Extract the displacement and velocities:
		D1 = x1[:self.dof]
		V1 = x1[self.dof:]

		# Compute acceleration response from the equation of motion:
		A1 = self.dydt(x1, t*self.dt+self.dt, self.sys, force)[self.dof:]
		return x1, D1, V1, A1

	def control_force(self, action):
		# Normalised action in [-1, 1] -> control force (kN)
		return self.u_max * np.clip(action, self.LB, self.UB)

	def step(self,action):
		# Time integration:
		action = self.control_force(action)
		self.x0, D1, V1, A1 = self.integrate(self.x0, self.t, action)

		self.D = D1
		self.V = V1
		self.A = A1
		self.t += 1
		self.action = action

		dic = OrderedDict([('states', self.x0)])

		obs = TimeStep(step_type=StepType.MID,
                       reward=self.reward(),
                       discount=1.0,
                       observation=dic)
		return obs

	def matrix_norm(self,x,weight):
		return np.matmul(np.matmul(np.transpose(x), weight), x)

	def reward(self):
		# Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
		J = self.Qu[0] * self.matrix_norm(self.D - self.D_ref, self.Qx) + \
            self.Qu[1] * self.matrix_norm(self.V - self.V_ref, self.Qx) + \
            self.Qu[2] * self.matrix_norm(self.A - self.A_ref, self.Qx)

		# Control effort: Add a penalty for large or inefficient control inputs.
		J += np.dot(np.dot(self.action, self.Ru), self.action)

		self.r = -J # Return -ve of cost function as a reward
		return self.r

	def one_step(self, states, t, action):
		# One agent step from agent step index t: the action is held for action_repeat
		# environment steps and the rewards are summed, as in ActionRepeatWrapper
		action = self.control_force(action)
		self.action = action
		x, r = states, 0
		for k in range(self.action_repeat):
			x, self.D, self.V, self.A = self.integrate(x, t*self.action_repeat + k, action)
			r += self.reward()
		self.x0 = x

		return self.x0, np.array([r])

	def done(self):
		return True if self.t > int(self.T/self.dt) else False

	def info(self):
		return defaultdict(float)


class Skyscraper_rom:
	def __init__(self, LB=None, UB=None, forced=True, action_repeat=1):

		# Initialize the scyscraper properties
		self.forced = forced
		self.action_repeat = action_repeat     # env steps per agent step in one_step (planner model)
		self.building()
		self.windpressure()

		self.action_dim = len(self.c_idx)
		self.D_ref = np.zeros(self.dof)    # reference: building at rest
		self.V_ref = np.zeros(self.dof)
		self.A_ref = np.zeros(self.dof)
		self.Qx = np.eye(self.dof)
		self.Qx[-1, -1] = 0     # don't penalise the TMD's own motion
		self.Qu = [12, 17, 15]                                # displacement, velocity, acceleration
		self.Ru = 1e-7 * np.eye(self.action_dim)             # control, u in kN

		# Normalised actions in [-1, 1]; the environment applies u = u_max * action (kN)
		self.u_max = 250
		self.LB = LB if LB is not None else -1*np.ones(self.action_dim)
		self.UB = UB if UB is not None else 1*np.ones(self.action_dim)

	def action_spec(self):
		return specs.BoundedArray(shape=self.LB.shape,
			   dtype=self.LB.dtype,
			   minimum=self.LB,
			   maximum=self.UB,
			   name='action')

	def observation_spec(self):
		state = specs.Array(shape=(2*self.dof,), dtype=self.LB.dtype, name='states')

		dic = OrderedDict([('states', state)])
		return dic

	def normalize(self,mode):
		# Normalizes the mode shapes of a structure
		# By deafult, it normalizes with respect to the top floor
		nmode = np.zeros(mode.shape)
		for i in range(len(mode[0])):
			nmode[:,i] = mode[:,i]/mode[-1,i]
		return nmode

	def building(self):
		# The 76 storey building properties (units: tonnes, kN/m, kN)
		data = sio.loadmat('data/B76_inp.mat')
		M76 = data['M76']
		K76 = data['K76']
		C76 = data['C76']

		# Get the natural frequencies and modes:
		eigval, eigvec = eigh(K76, M76)     # real eigenvalues in ascending order

		# Get the first modal mass:
		idx = 0
		phi = self.normalize(eigvec)
		modal_m = (phi[:,idx].T @ M76 @ phi[:,idx]).squeeze()
		self.phi1 = phi[:,idx]              # first-mode shape, 1 at the roof

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
		self.omega = omega

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

		# Floors retained in the reduced-order model (Guyan condensation):
		self.id_ = [2, 5, 9, 12, 15, 19, 22, 25, 29, 32, 35, 39,
                    42, 45, 49, 52, 55, 59, 62, 65, 69, 72, 75, 76]

		m_rom, k_rom, c_rom = self.rom(m=M77,k=K77,c=C77,order=self.id_)

		self.M = m_rom
		self.C = c_rom
		self.K = k_rom
		self.Mt = M77
		self.Ct = C77
		self.Kt = K77

		# Time step
		self.dt = 1e-2
		self.T = 60

		self.dof = self.Mt.shape[0]
		self.rom_dof = self.M.shape[0]

		# Inverse of the reduced mass matrix and the force projection, q'' = iM T' F:
		iM = inv(self.M)
		P = iM @ self.vec.T

		# Statespace matrix (reduced coordinates)
		self.sys = np.vstack(( np.column_stack(( np.zeros(self.M.shape), np.eye(self.rom_dof) )), \
                               np.column_stack(( -iM @ self.K, -iM @ self.C )) ))

		# Nonlinearity: cubic springs f_nl = alpha * x^3 at nl_idx (alpha > 0: hardening)
		self.alpha = np.array([4.6e7, 3.7e6, 3.7e5])     # kN/m^3
		self.nl_idx = [35, 52, 75]

		# Define the excitation influence matrix (no wind on the TMD):
		self.sigma = np.eye(self.dof)
		self.sigma[-1,-1] = 0

		# Define control influence matrix (actuator DOFs, 0-based):
		self.c_idx = np.array([5, 10, 15, 25, 35, 45, 55, 60, 65, 72, 75])
		H = np.zeros((self.dof, len(self.c_idx)))           # Influence matrix
		H[(self.c_idx, np.arange(len(self.c_idx)))] = 1
		self.H = H

		# Precompute the projected load matrices (reduced dimension):
		self.P_w = P @ self.sigma                   # wind
		self.P_h = P @ H                            # control
		self.T_nl = self.vec[self.nl_idx, :]        # physical displacement at the nonlinear DOFs
		self.P_nl = P[:, self.nl_idx]               # projection of the nonlinear forces

		# RK4 substeps from the stability limit of the highest retained frequency:
		omega_max = np.sqrt(eigh(self.K, self.M, eigvals_only=True)[-1])
		self.substeps = max(1, int(np.ceil(omega_max * self.dt / 2.0)))

	# Perform Model-order reduction:
	def rom(self,m,k,c,order):
		"""
        Guyan (static) condensation onto the retained DOFs 'order', x = vec @ q with q = x[order].
        """
		n = m.shape[0]

		# Model-order reduction equation:
		self.rom_system = lambda matrix, phi: np.matmul( np.matmul(phi.T, matrix), phi )

		# Slave DOFs follow the master DOFs statically:
		master = np.asarray(order)
		slave = np.setdiff1d(np.arange(n), master)
		vec = np.zeros((n, len(master)))
		vec[master, np.arange(len(master))] = 1
		vec[slave] = -np.linalg.solve(k[np.ix_(slave, slave)], k[np.ix_(slave, master)])
		self.vec = vec

		# Get the reduced-order matrices:
		m_rom = self.rom_system(m, self.vec)
		k_rom = self.rom_system(k, self.vec)
		c_rom = self.rom_system(c, self.vec)
		return (m_rom, k_rom, c_rom)

	def windpressure(self):
		# Wind load calculation:
		L = 42          # width of building
		z_ref = 10      # reference height (m)
		rho = 1.22      # air density
		cd = 1.2        # drag coefficient
		z = np.concatenate((
                            np.array([10]),
                            10 + np.linspace(4.5,9,2),
                            10 + 9 + np.linspace(3.9,136.5,35),
                            10 + 9 + 136.5 + np.linspace(4.5,9,2),
                            10 + 9 + 136.5 + 9 + np.linspace(3.9,132.6,34),
                            10 + 9 + 136.5 + 9 + 132.6 + np.linspace(4.5,9,2)
                           ))       # height of the skyscraper
		h = np.gradient(z)          # tributary height of each floor (m)

		# Wind speed history: mean u_ref at z_ref with a gust at the first-mode frequency
		u_ref = 13.5                    # mean wind speed at z_ref = 10 m (m/s)
		I_u = 0.2                       # turbulence intensity
		f_w = self.omega[0]/2/np.pi     # gust frequency (Hz)
		self.force_fun = lambda arg : 0 if self.forced == False else u_ref*(1 + I_u*np.sin(2*np.pi*f_w*arg))

		# Define the forcing function (kN, consistent with tonnes and kN/m):
		fun = lambda rho, L, h, cd, u_ref, z_ref, z: 0.5*rho*L*h*cd*u_ref**2*(z/z_ref)**0.365 / 1000

		# Compressing the forcing function:
		self.intensity = partial(fun, rho=rho, L=L, h=h, cd=cd, z_ref=z_ref, z=z)

	def reset(self):
		# Intial state vector: first-mode shape (0.1 m at the roof), at rest;
		# the TMD starts with the roof, so its spring is unstretched
		D0 = 0.1*np.concatenate(( self.phi1, self.phi1[-1:] ))
		V0 = np.zeros(self.dof)
		A0 = np.zeros(self.dof)

		# Keep the state in the reduced-order subspace, x = vec @ x[id_]:
		D0 = self.vec @ D0[self.id_]

		# Statespace vector containing only diplacement and velocity
		self.x0 = np.concatenate(( D0, V0 ))

		# Store the system matrices:
		self.D = D0
		self.V = V0
		self.A = A0
		self.t = 0
		self.r = 0

		dic = OrderedDict([('states', self.x0)])

		obs = TimeStep(step_type=StepType.FIRST,
                       reward=None,
                       discount=None,
                       observation=dic)
		return obs

	# first-order statespace equation (reduced coordinates):
	def dydt(self,x,force):
		# Nonlinearity (cubic springs on the physical DOFs, then projected)
		f_nl = self.alpha * (self.T_nl @ x[:self.rom_dof])**3

		# Evaluate the derivatives; 'force' is the projected wind + control force
		dydt = np.matmul(self.sys,x)
		dydt[self.rom_dof:] += force - self.P_nl @ f_nl
		return dydt

	def integrate(self, states, t, action):
		# One time step from time index t with RK4, wind and action held over the step
		D, V = states[:self.dof], states[self.dof:]
		x = np.concatenate(( D[self.id_], V[self.id_] ))       # Guyan: q = x[id_]

		ft = np.concatenate(( self.intensity(u_ref = self.force_fun(t*self.dt)), np.zeros(1) ))
		force = self.P_w @ ft + self.P_h @ action           # 'action' is the control force in kN

		hs = self.dt / self.substeps
		for _ in range(self.substeps):
			k1 = self.dydt(x, force)
			k2 = self.dydt(x + 0.5*hs*k1, force)
			k3 = self.dydt(x + 0.5*hs*k2, force)
			k4 = self.dydt(x + hs*k3, force)
			x = x + hs/6 * (k1 + 2*k2 + 2*k3 + k4)

		# Displacement, velocity and acceleration in physical coordinates:
		D1 = self.vec @ x[:self.rom_dof]
		V1 = self.vec @ x[self.rom_dof:]
		A1 = self.vec @ self.dydt(x, force)[self.rom_dof:]
		return np.concatenate(( D1, V1 )), D1, V1, A1

	def control_force(self, action):
		# Normalised action in [-1, 1] -> control force (kN)
		return self.u_max * np.clip(action, self.LB, self.UB)

	def step(self,action):
		# Time integration:
		action = self.control_force(action)
		self.x0, D1, V1, A1 = self.integrate(self.x0, self.t, action)

		self.D = D1
		self.V = V1
		self.A = A1
		self.t += 1
		self.action = action

		dic = OrderedDict([('states', self.x0)])

		obs = TimeStep(step_type=StepType.MID,
                       reward=self.reward(),
                       discount=1.0,
                       observation=dic)
		return obs

	def matrix_norm(self,x,weight):
		return np.matmul(np.matmul(np.transpose(x), weight), x)

	def reward(self):
		# Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
		J = self.Qu[0] * self.matrix_norm(self.D - self.D_ref, self.Qx) + \
            self.Qu[1] * self.matrix_norm(self.V - self.V_ref, self.Qx) + \
            self.Qu[2] * self.matrix_norm(self.A - self.A_ref, self.Qx)

		# Control effort: Add a penalty for large or inefficient control inputs.
		J += np.dot(np.dot(self.action, self.Ru), self.action)

		self.r = -J # Return -ve of cost function as a reward
		return self.r

	def one_step(self, states, t, action):
		# One agent step from agent step index t: the action is held for action_repeat
		# environment steps and the rewards are summed, as in ActionRepeatWrapper
		action = self.control_force(action)
		self.action = action
		x, r = states, 0
		for k in range(self.action_repeat):
			x, self.D, self.V, self.A = self.integrate(x, t*self.action_repeat + k, action)
			r += self.reward()
		self.x0 = x

		return self.x0, np.array([r])

	def done(self):
		return True if self.t > int(self.T/self.dt) else False

	def info(self):
		return defaultdict(float)


# %%
if __name__=='__main__':
	"""
    View the uncontrolled response, e.g.:
        python env_tallstorey.py --model rom --T 30 --dt 0.01
        python env_tallstorey.py --model full --T 10 --floors 35 52 75 76 --unforced
    """
	import argparse
	import timeit

	parser = argparse.ArgumentParser(description='Response of the 76 DOF skyscraper environment')
	parser.add_argument('--model', choices=['rom', 'full'], default='rom', help='Skyscraper_rom or Skyscraper')
	parser.add_argument('--T', type=float, default=30, help='simulation time (s)')
	parser.add_argument('--dt', type=float, default=0.01, help='time step (s)')
	parser.add_argument('--floors', type=int, nargs='+', default=[5, 35, 52, 75], help='DOFs to plot')
	parser.add_argument('--action', type=float, default=0.0, help='constant normalised action in [-1, 1]')
	parser.add_argument('--unforced', action='store_true', help='switch off the wind')
	args = parser.parse_args()

	# Call the environment:
	env = Skyscraper_rom(forced=not args.unforced) if args.model == 'rom' else Skyscraper(forced=not args.unforced)
	env.dt, env.T = args.dt, args.T
	if hasattr(env, 'substeps'):
		# RK4 substeps for the new time step (stability limit of the highest ROM frequency)
		omega_max = np.sqrt(eigh(env.K, env.M, eigvals_only=True)[-1])
		env.substeps = max(1, int(np.ceil(omega_max * env.dt / 2.0)))

	# Initialize the states:
	n = int(round(env.T/env.dt))
	m = env.dof
	D, V, A, R = np.zeros([m,n+1]), np.zeros([m,n+1]), np.zeros([m,n+1]), np.zeros(n)

	obs = env.reset()
	D[:, 0], V[:, 0], A[:, 0] = env.D, env.V, env.A

	# Time integration:
	t0 = timeit.default_timer()
	action = args.action*np.ones(env.action_dim)
	for i in range(n):
		xt = env.step(action=action)
		D[:,i+1], V[:,i+1], A[:,i+1], R[i] = env.D, env.V, env.A, xt.reward

		if i % int(max(1, n//10)) == 0: print('Step: {}, Reward: {:.4f}'.format(i, xt.reward))

	print('Model: {}, T = {} s, dt = {} s, {} steps, time: {:.2f} s'.format(
          args.model, env.T, env.dt, n, timeit.default_timer() - t0))

	# %%
	floor = args.floors
	t = env.dt*np.arange(n+1)

	figure1, ax = plt.subplots(nrows=len(floor), ncols=3, figsize =(18,10), dpi=100, squeeze=False)
	plt.subplots_adjust(hspace=0.35, wspace=0.35)

	for i in range(len(floor)):
		for j, (y, lab) in enumerate(zip([D, V, A], ['X', r'\dot{{X}}', r'\ddot{{X}}'])):
			ax[i,j].plot(t, y[floor[i],:], label='DOF:{}th'.format(floor[i]))
			if i == len(floor)-1: ax[i,j].set_xlabel('Time (Sec)')
			ax[i,j].set_ylabel(r'${}({})$'.format(lab, floor[i]))
			ax[i,j].grid(True, alpha=0.25)

	plt.suptitle('Response of 76 DOF Skyscraper (RL ENV: {})'.format(args.model), y=0.94)
	plt.show()
