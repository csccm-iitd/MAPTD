# -*- coding: utf-8 -*-
"""
This script defines the environment for the Beam with a Wavelet Neural Operator surrogate
"""

import numpy as np
import matplotlib.pyplot as plt

import os
directory = os.path.abspath(os.path.join(os.path.dirname('NO'), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_convolution import WaveConv1d
from env_beam import EulerBeam as EulerBeam_physics

"""
Codes for free vibration of a cantilever using Neural Operator
"""

class WNO1d(nn.Module):
	def __init__(self,width,level,layers,size,wavelet,in_channel,grid_range,omega):
		super(WNO1d, self).__init__()

		self.level = level
		self.width = width
		self.layers = layers
		self.size = size
		self.wavelet = wavelet
		self.omega = omega
		self.in_channel = in_channel
		self.grid_range = grid_range

		self.conv0, self.w0 = nn.ModuleList(), nn.ModuleList()
		self.conv1, self.w1 = nn.ModuleList(), nn.ModuleList()
		self.conv2, self.w2 = nn.ModuleList(), nn.ModuleList()

		self.fc10 = nn.Linear(self.in_channel[0], self.width[0]) # input channel is 2: (a(x), x)
		self.fc11 = nn.Linear(self.in_channel[0], self.width[0]) # input channel is 2: (a(x), x)
		self.fc12 = nn.Linear(self.in_channel[0], self.width[0]) # input channel is 2: (a(x), x)

		self.ft = nn.Linear(self.in_channel[1], self.width[0]) # input channel is 2: (f(x), u(x))

		for i in range( self.layers - 1 ):
			self.conv0.append( WaveConv1d(3*self.width[i]+self.width[0], self.width[i+1], self.level,
                                         self.size, self.wavelet, omega=self.omega) )
			self.w0.append( nn.Conv1d(3*self.width[i]+self.width[0], self.width[i+1], 1) )

			self.conv1.append( WaveConv1d(3*self.width[i]+self.width[0], self.width[i+1], self.level,
                                         self.size, self.wavelet, omega=self.omega) )
			self.w1.append( nn.Conv1d(3*self.width[i]+self.width[0], self.width[i+1], 1) )

			self.conv2.append( WaveConv1d(3*self.width[i]+self.width[0], self.width[i+1], self.level,
                                         self.size, self.wavelet, omega=self.omega) )
			self.w2.append( nn.Conv1d(3*self.width[i]+self.width[0], self.width[i+1], 1) )

		self.fc21 = nn.Sequential(nn.Linear(self.width[-1], 128), nn.Mish(), nn.Linear(128, 1))
		self.fc22 = nn.Sequential(nn.Linear(self.width[-1], 128), nn.Mish(), nn.Linear(128, 1))
		self.fc23 = nn.Sequential(nn.Linear(self.width[-1], 128), nn.Mish(), nn.Linear(128, 1))

	def forward(self, x, f, u):
		grid = self.get_grid(x.shape, x.device)

		# append the grid:
		x0 = torch.cat((x[:,:,0:1], grid), dim=-1)
		x1 = torch.cat((x[:,:,1:2], grid), dim=-1)
		x2 = torch.cat((x[:,:,2:3], grid), dim=-1)
		fu = torch.cat((f,u), dim=-1)

		# get the latent space:
		x0 = self.fc10(x0)              # Shape: Batch * x * Channel
		x1 = self.fc11(x1)              # Shape: Batch * x * Channel
		x2 = self.fc12(x2)              # Shape: Batch * x * Channel
		fu = self.ft(fu)

		x0 = x0.permute(0, 2, 1)       # Shape: Batch * Channel * x
		x1 = x1.permute(0, 2, 1)       # Shape: Batch * Channel * x
		x2 = x2.permute(0, 2, 1)       # Shape: Batch * Channel * x
		fu = fu.permute(0, 2, 1)

		for index, (cl0, wl0, cl1, wl1, cl2, wl2) in enumerate( zip(self.conv0, self.w0,
                                                                    self.conv1, self.w1,
                                                                    self.conv2, self.w2) ):
			if index == 0:
				z = torch.cat((x0,x1,x2,fu), dim=1)
				v0 = cl0(z) + wl0(z)
				v1 = cl1(z) + wl1(z)
				v2 = cl2(z) + wl2(z)
			else:
				z = torch.cat((v0,v1,v2,fu), dim=1)
				v0 = cl0(z) + wl0(z)
				v1 = cl1(z) + wl1(z)
				v2 = cl2(z) + wl2(z)

			if index != self.layers - 1:   # Final layer has no activation
				v0 = F.mish(v0)            # Shape: Batch * Channel * x
				v1 = F.mish(v1)            # Shape: Batch * Channel * x
				v2 = F.mish(v2)            # Shape: Batch * Channel * x

		v0 = v0.permute(0, 2, 1)       # Shape: Batch * x * Channel
		v1 = v1.permute(0, 2, 1)       # Shape: Batch * x * Channel
		v2 = v2.permute(0, 2, 1)       # Shape: Batch * x * Channel

		v0 = self.fc21(v0)    # Shape: Batch * x * Channel
		v1 = self.fc22(v1)    # Shape: Batch * x * Channel
		v2 = self.fc23(v2)    # Shape: Batch * x * Channel
		return torch.cat((v0,v1,v2), dim=-1)

	def get_grid(self, shape, device):
		# The grid of the solution
		batchsize, size_x = shape[0], shape[1]
		gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
		gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
		return gridx.to(device)


class EulerBeam(EulerBeam_physics):
	"""
    Surrogate of env_beam.EulerBeam: the beam, initial state, forcing, action scaling and reward
    weights are inherited, and the response over a time step is predicted by the WNO
    """
	def __init__(self, lr=1e-3, device=torch.device('cpu'), LB=None, UB=None, forced=False, action_repeat=1,
                 path=directory + '/NO/model/WNO_beam_control2'):
		super().__init__(LB=LB, UB=UB, forced=forced, action_repeat=action_repeat)

		self.device = device
		self.lr = lr
		self.model = torch.load(path, map_location=self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

	def force_input(self, t, batch):
		# WNO force input for the transition from env step t to t+1: the force at the new
		# time step, as applied by the real environment (env_beam.py). t: scalar or per sample
		t = np.broadcast_to(np.asarray(t), (batch,))
		force = np.stack([self.get_force(tk+1) for tk in t])
		return torch.tensor(force[:,:,None], dtype=torch.float32).to(self.device)

	def reward(self, D, V, A, action):
		# Batched reward of the real environment: states (batch x 2Ne), normalized actions (batch x action_dim)
		J = self.Qu[0] * np.sum((D[:,::2] - self.D_ref)**2, axis=-1, keepdims=True) + \
            self.Qu[1] * np.sum((V[:,::2] - self.V_ref)**2, axis=-1, keepdims=True) + \
            self.Qu[2] * np.sum((A[:,::2] - self.A_ref)**2, axis=-1, keepdims=True)

		# Control effort: Add a penalty for large or inefficient control inputs.
		J += np.einsum('bi,ij,bj->b', action, self.Ru, action)[:, None]

		return -J # Return -ve of cost function as a reward

	def one_step(self, states, t, action):
		# One agent step from agent step index t: the action is held for action_repeat
		# environment steps (one WNO call each) and the rewards are summed, as in ActionRepeatWrapper
		batch = states.shape[0]
		states = states.reshape(batch, 2*self.Ne, -1, order='F')

		# Location of control forces:
		u = np.zeros((batch, 2*self.Ne))
		u[:, 32:32*3] = self.scale_action(action)  # Control using Piezoelectric patch at 1/8th to 3/8th length

		x = torch.tensor(states, dtype=torch.float32).to(self.device)
		uin = torch.tensor(u[:,:,None], dtype=torch.float32).to(self.device)

		r = 0
		self.model.eval()
		with torch.no_grad():
			for k in range(self.action_repeat):
				x = self.model(x, self.force_input(t*self.action_repeat + k, batch), uin)
				out = x.cpu().numpy()
				r += self.reward(out[:,:,0], out[:,:,1], out[:,:,2], action)

		# Do not overwrite the environment states (used by the planner)
		return out.reshape(batch, -1, order='F'), r.astype('float32')

	def update(self, xx, yt, t, action):
		# One gradient step on observed agent-step transitions xx -> yt from agent step index t
		# (scalar or one per sample): the WNO is rolled out for action_repeat env steps
		batch = xx.shape[0]
		Dx, Vx, Ax = xx[:,:2*self.Ne], xx[:,2*self.Ne:4*self.Ne], xx[:,4*self.Ne:]
		xx = torch.stack((Dx, Vx, Ax), axis=-1).to(self.device)

		D, V, A = yt[:,:2*self.Ne], yt[:,2*self.Ne:4*self.Ne], yt[:,4*self.Ne:]
		yt = torch.stack((D, V, A), axis=-1).to(self.device)

		# Location of control forces:
		action = torch.as_tensor(self.scale_action(action.detach().cpu().numpy()), dtype=torch.float32)
		u = torch.zeros((batch, 2*self.Ne))
		u[:, 32:32*3] = action  # Control using Piezoelectric patch at 1/8th to 3/8th length
		u = u[:,:,None].to(self.device)

		self.model.train()
		loss = 0
		self.optimizer.zero_grad(set_to_none=True)

		out = xx
		for k in range(self.action_repeat):
			out = self.model(out, self.force_input(t*self.action_repeat + k, batch), u)
		loss = torch.norm(yt - out)/torch.norm(yt)

		loss.backward()
		self.optimizer.step()


# %%
if __name__=='__main__':
	# Call the environment:
	env = EulerBeam()

	# Initialize the states:
	n = int(round(env.T/env.dt))
	m = 2 * env.Ne

	D = np.zeros([m,n])
	V = np.zeros([m,n])
	A = np.zeros([m,n])

	obs = env.reset()
	D[:, 0] = obs.observation['displacement']
	V[:, 0] = obs.observation['velocity']
	A[:, 0] = obs.observation['acceleration']

	# Time integration:
	for i in range(n-1):
		D0 = D[:, i]
		V0 = V[:, i]
		A0 = A[:, i]

		states, r = env.one_step(states=np.concatenate((D0,V0,A0))[None, :],
                                  t=i, action=np.zeros(env.action_dim)[None, :])

		if i % 20 == 0: print('Step: {}, Reward: {}'.format(i,r[0,0]))

		D[:, i+1] = states[0, :2*env.Ne]
		V[:, i+1] = states[0, 2*env.Ne:4*env.Ne]
		A[:, i+1] = states[0, 4*env.Ne:]

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

