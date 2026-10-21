# -*- coding: utf-8 -*-
"""
This script defines the environment for the Beam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from dm_env import specs, TimeStep, StepType
from collections import OrderedDict
from collections import defaultdict

"""
Codes for free vibration of a cantilever
"""
class EulerBeam:
	def __init__(self, LB=None, UB=None, forced=False, action_repeat=1):

		# Both the real env and the planner model are built with defaults (dynamics_gym),
		# so change the default of 'forced' here to switch the forcing on for both
		self.forced = forced
		self.action_repeat = action_repeat     # env steps per agent step in one_step (planner model)
		self.action_dim = 32*2
		self.Ne = 100   # actually, 2 times Ne, one for deflection and one for rotation
		self.D_ref = np.zeros(self.Ne)      # target: zero deflection
		self.V_ref = np.zeros(self.Ne)      # target: zero velocity
		self.A_ref = np.zeros(self.Ne)
		self.Qu = [1, 1, 0]
		self.Ru = 1e-3*np.eye(self.action_dim)      # control cost stays ~5-10% of the uncontrolled state cost
		self.LB = LB if LB is not None else -1*np.ones(self.action_dim)     # normalized action bounds
		self.UB = UB if UB is not None else 1*np.ones(self.action_dim)
		self.action_scale = 0.01    # maps normalized action [-1,1] to physical force (scaled with init_amp)
		self.init_amp = 0.02     # initial tip deflection (m), L/50 to stay within small-deflection theory
		self.force_amp = 5e-4    # amplitude of the external force, scaled with init_amp (was 0.05 for 2 m)

		self.get_properties(forced=self.forced)
		self.newmark()

	def action_spec(self):
		return specs.BoundedArray(shape=self.LB.shape,
			   dtype=self.LB.dtype,
			   minimum=self.LB,
			   maximum=self.UB,
			   name='action')

	def observation_spec(self):
		dis_state = specs.Array(shape=(2*self.Ne,), dtype=self.LB.dtype, name='displacement')
		vel_state = specs.Array(shape=(2*self.Ne,), dtype=self.LB.dtype, name='velocity')
		acc_state = specs.Array(shape=(2*self.Ne,), dtype=self.LB.dtype, name='Acceleration')

		dic = OrderedDict([('displacement', dis_state),
                            ('velocity', vel_state),
                            ('acceleration', acc_state)])
		# dic = OrderedDict([('displacement', dis_state)])
		return dic

	def Beam3(self,rho,A,E,I,Le,n,bc):
		# Euler-Bernoulli Beam
		# spatial discretization by finite element method
		# n is the NUMBER of NODES including the left end
		# Le is the LENGTH of each ELEMENT

		# element mass and stiffness matrix
		Me = rho*A*Le/420*np.array([[156,    22*Le,   54,     -13*Le],
                                    [22*Le,  4*Le**2,  13*Le,  -3*Le**2],
                                    [54,     13*Le,   156,    -22*Le],
                                    [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])

		Ke = E*I/(Le**3)*np.array([[12,    6*Le,    -12,    6*Le],
                                   [6*Le,  4*Le**2,  -6*Le,  2*Le**2],
                                   [-12,   -6*Le,   12,     -6*Le],
                                   [6*Le,  2*Le**2,  -6*Le,  4*Le**2]])

		# global mass and stiffness matrix
		Ma = np.zeros([2*n,2*n])
		Ka = np.zeros([2*n,2*n])
		for i in range(0, 2*n-3, 2):
			Ma[i:i+4,i:i+4] = Ma[i:i+4,i:i+4] + Me
			Ka[i:i+4,i:i+4] = Ka[i:i+4,i:i+4] + Ke

		# boundary conditions !
		# bcs = 'general' and 'simply-supported';
		if bc == 'cantilever':
			# the left end is clamped !
			Ma = np.delete(Ma, [0,1], 1) # column delete
			Ma = np.delete(Ma, [0,1], 0) # row delete
			Ka = np.delete(Ka, [0,1], 1)
			Ka = np.delete(Ka, [0,1], 0)

		elif bc == 'simply-supported':
			# simply supported at two ends
			Ma = np.delete(Ma, [0,-2], 1) # first and second last column
			Ma = np.delete(Ma, [0,-2], 0) # first and second last row
			Ka = np.delete(Ka, [0,-2], 1)
			Ka = np.delete(Ka, [0,-2], 0)

		else:
			raise Exception('Boundary Condition Not Implemented')

		return Ma, Ka

	def get_properties(self, forced=False):
		# Material properties of beam:
		rho, E = 7800, 2e11
		b, d, self.L = 0.0254, 0.002, 1
		A = b*d
		I = b*d**3/12
		self.grid = np.arange(self.L/self.Ne, self.L+self.L/self.Ne, self.L/self.Ne)

		self.mode = 0
		c1, c2 = 0, 0

		# Get system matrices:
		Ma, Ka = self.Beam3(rho,A,E,I,(self.L/self.Ne),(self.Ne+1),'cantilever')
		Ca = (c1*Ma + c2*Ka)

		self.M = Ma
		self.C = Ca
		self.K = Ka

		# Force function:
		self.force_fun = lambda arg : 0 if forced == False else self.force_amp*np.sin(2*np.pi*arg)

	def reset(self):
		Lambda = np.array([1.875104069, 4.694091133, 7.854757438, 10.99554073,
                           14.13716839, 17.27875953]) / self.L
		beta = Lambda[self.mode]
		fraction = (np.cos(beta*self.L)+np.cosh(beta*self.L)) / (np.sin(beta*self.L)+np.sinh(beta*self.L))

		delta = np.cosh(beta*self.grid) - np.cos(beta*self.grid) - fraction*(np.sinh(beta*self.grid) - np.sin(beta*self.grid))
		theta = beta*(np.sinh(beta*self.grid) + np.sin(beta*self.grid)) - fraction*(np.cosh(beta*self.grid) - np.cos(beta*self.grid))*beta

		# Intial Displacement:
		D0 = np.zeros(2*self.Ne)
		D0[0::2] = delta
		D0[1::2] = theta
		D0 = D0 * self.init_amp / delta[-1]     # normalize mode shape: tip deflection = init_amp

		# Intial Velocity:
		V0 = np.zeros(2*self.Ne)

		# Intial Acceleration:
		A0 = np.linalg.solve(self.M, self.get_force(0) - np.dot(self.K,D0) - np.dot(self.C,V0))      # initial acceleration

		# Store the system matrices:
		self.D = D0
		self.V = V0
		self.A = A0
		self.t = 0
		self.r = 0

		dic = OrderedDict([('displacement', D0),
                            ('velocity', V0),
                            ('acceleration', A0)])

		obs = TimeStep(step_type=StepType.FIRST,
                       reward=None,
                       discount=None,
                       observation=dic)
		return obs

	def newmark(self,newmark_Beta=1/4,newmark_Gamma=1/2):
		# Time step
		self.dt = 1e-2
		self.T = 1

		# integration constant
		self.nc1 = 1/newmark_Beta/self.dt**2
		self.nc2 = newmark_Gamma/newmark_Beta/self.dt
		self.nc3 = 1/newmark_Beta/self.dt
		self.nc4 = 1/2/newmark_Beta - 1
		self.nc5 = newmark_Gamma/newmark_Beta - 1
		self.nc6 = (newmark_Gamma/2/newmark_Beta - 1)*self.dt
		self.nc7 = (1 - newmark_Gamma)*self.dt
		self.nc8 = newmark_Gamma*self.dt

		Kbar = self.nc1*self.M + self.nc2*self.C + self.K          # linear time-invariant system
		self.Kbar_f = cho_factor(Kbar)                            # factor once, reuse every step

	def scale_action(self, action):
		# Clip the normalized action to [LB, UB] and scale it to physical force
		return self.action_scale * np.clip(action, self.LB, self.UB)

	def get_force(self, t):
		# External force at time step t (applied on all DOFs):
		force = np.zeros(2*self.Ne)
		force[::2] = self.force_fun(t*self.dt) * self.L / 2
		force[1::2] = self.force_fun(t*self.dt) * self.L**2 / 12
		return force

	def integrate(self, D, V, A, t, action):
		# One Newmark step from env time step t to t+1
		# Location of control forces:
		u = np.zeros(2*self.Ne)
		u[32:32*3] = self.scale_action(action)  # Control using Piezoelectric patch at 1/8th to 3/8th length

		# Force at the new time step:
		force = self.get_force(t+1)

		Fbar = force + np.dot(self.M, (self.nc1*D + self.nc3*V + self.nc4*A)) \
                     + np.dot(self.C, (self.nc2*D + self.nc5*V + self.nc6*A)) + u

		D1 = cho_solve(self.Kbar_f, Fbar)
		A1 = self.nc1*(D1 - D) - self.nc3*V - self.nc4*A
		V1 = V + self.nc7*A + self.nc8*A1
		return D1, V1, A1

	def step(self,action):
		# Time integration:
		D1, V1, A1 = self.integrate(self.D, self.V, self.A, self.t, action)

		self.D = D1
		self.V = V1
		self.A = A1
		self.t += 1
		self.action = action
		self.r = self.reward()

		dic = OrderedDict([('displacement', D1),
                            ('velocity', V1),
                            ('acceleration', A1)])
		# dic = OrderedDict([('displacement', D1)])

		obs = TimeStep(step_type=StepType.MID,
                       reward=self.r,
                       discount=1.0,
                       observation=dic)
		return obs

	def reward(self, D=None, V=None, A=None, action=None):
		# Defaults to the current environment states
		D = self.D if D is None else D
		V = self.V if V is None else V
		A = self.A if A is None else A
		action = self.action if action is None else action

		# Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
		J = self.Qu[0] * np.sum((D[::2] - self.D_ref)**2) + \
            self.Qu[1] * np.sum((V[::2] - self.V_ref)**2) + \
            self.Qu[2] * np.sum((A[::2] - self.A_ref)**2)

		# Control effort: Add a penalty for large or inefficient control inputs.
		J += np.dot(np.dot(action, self.Ru), action)

		return -J # Return -ve of cost function as a reward

	def one_step(self, states, t, action):
		# One agent step from agent step index t: the action is held for action_repeat
		# environment steps and the rewards are summed, as in ActionRepeatWrapper
		# Extract the states from concatenated observations
		D, V, A = states[:2*self.Ne], states[2*self.Ne:4*self.Ne], states[4*self.Ne:]

		r = 0
		for k in range(self.action_repeat):
			D, V, A = self.integrate(D, V, A, t*self.action_repeat + k, action)
			r += self.reward(D, V, A, action)

		# Do not overwrite the environment states (used by the planner)
		return np.concatenate([D, V, A]), np.array([r])

	def done(self):
		return True if self.t > int(round(self.T/self.dt)) else False

	def info(self):
		return defaultdict(float)


# %%
if __name__=='__main__':
	# Call the environment:
	env = EulerBeam()

	# %%
	# Initialize the states:
	n = int(round(env.T/env.dt))
	m = env.Ne

	D = np.zeros([m,n])
	V = np.zeros([m,n])
	A = np.zeros([m,n])

	obs = env.reset()
	D[:, 0] = obs.observation['displacement'][::2]
	V[:, 0] = obs.observation['velocity'][::2]
	A[:, 0] = obs.observation['acceleration'][::2]

	# Time integration:
	for i in range(n-1):
		xt = env.step(action=np.zeros(env.action_dim))

		if i % 20 == 0: print('Step: {}, Reward: {}'.format(i,xt.reward))

		D[:,i+1] = xt.observation['displacement'][::2]
		V[:,i+1] = xt.observation['velocity'][::2]
		A[:,i+1] = xt.observation['acceleration'][::2]

	# %%
	t = np.arange(n) * env.dt
	extent = [0, env.L, t[0], t[-1]]      # x: space (m), y: time (s)

	fig2, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,9), dpi=100, sharex=True)
	plt.subplots_adjust(hspace=0.3)
	for k, (Y, name) in enumerate([(D, 'Displacement (m)'), (V, 'Velocity (m/s)'), (A, 'Acceleration (m/s$^2$)')]):
		im = ax[k].imshow(Y, cmap='jet', origin='lower', aspect='auto', extent=extent)
		ax[k].set_title(name);
		ax[k].set_xlabel('Time (sec)')
		plt.colorbar(im, ax=ax[k], pad=0.01)
	ax[-1].set_ylabel('Space (x)')
	plt.show()

