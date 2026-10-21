# -*- coding: utf-8 -*-
"""
This script defines the environment for the Skyscraper with a neural network surrogate
"""

import numpy as np

import os
directory = os.path.abspath(os.path.join(os.path.dirname('MPC'), '..'))

import torch
import torch.nn as nn
from env_tallstorey import Skyscraper_rom as Skyscraper_rom_physics

class dnn(nn.Module):
	def __init__(self, x_dim, f_dim, u_dim, emd_dim, out_dim, mlps_net=[512,384,256,128]):
		super(dnn, self).__init__()

		"""
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        in_channel: scalar, channels in input including grid
        """

		self.states = x_dim[0]
		self.x_dim = x_dim[1]
		self.in_dim = np.prod(x_dim)
		self.f_dim = f_dim
		self.u_dim = u_dim
		self.emd_dim = emd_dim
		self.out_dim = np.prod(out_dim)

		self.net00 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)
		self.net01 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)

		self.net10 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)
		self.net11 = self.net(4*self.emd_dim, self.emd_dim, mlps_net)

		self.enc_x0 = self.encoder(self.x_dim, self.emd_dim)
		self.enc_x1 = self.encoder(self.x_dim, self.emd_dim)

		self.enc_f = self.encoder(self.f_dim, self.emd_dim)
		self.enc_u = self.encoder(self.u_dim, self.emd_dim)

		self.dec0 = self.decoder(self.emd_dim, self.x_dim)
		self.dec1 = self.decoder(self.emd_dim, self.x_dim)
		self.dec2 = self.decoder(self.emd_dim, self.x_dim)

	def net(self, in_dim, out_dim, mlp, act=nn.ReLU()):
		layers = len(mlp)
		net = []
		net.append( nn.Linear(in_dim, mlp[0]) )
		net.append( act )
		for i in range( layers-1 ):
			net.append( nn.Linear(in_features=mlp[i], out_features=mlp[i+1]) )
			net.append( act )
		net.append( nn.Linear(in_features=mlp[-1], out_features=out_dim) )
		return nn.Sequential(*net)

	def encoder(self, in_dim, out_dim):
		return nn.Linear(in_features=in_dim, out_features=out_dim)

	def decoder(self, in_dim, out_dim):
		# fc = nn.Sequential(nn.Linear(in_features=in_dim, out_features=in_dim),
		#                    nn.ReLU(),
		#                    nn.Linear(in_features=in_dim, out_features=out_dim))
		fc = nn.Linear(in_features=in_dim, out_features=out_dim)
		return fc

	def forward(self, x, f, u):
		# pre-processing
		x0, x1 = x[:,0], x[:,1]

		# get the latent space
		x0, x1 = self.enc_x0(x0), self.enc_x1(x1)

		f = self.enc_f(f)
		u = self.enc_u(u)

		z = torch.cat((x0,x1,f,u), dim=-1)
		x0, x1 = self.net00(z), self.net01(z)
		z = torch.cat((x0,x1,f,u), dim=-1)
		x0, x1 = self.net10(z), self.net11(z)

		# decode the latent space
		x0 = self.dec0(x0)
		x1 = self.dec1(x1)
		return torch.stack((x0,x1)).permute(1,0,2)


class Skyscraper_rom(Skyscraper_rom_physics):
	"""
    Surrogate of env_tallstorey.Skyscraper_rom: the building, wind, initial state, action scaling
    and reward weights are inherited, and the response over a time step is predicted by the network
    """
	def __init__(self, lr=1e-3, device=torch.device('cpu'), LB=None, UB=None, forced=True, action_repeat=1,
                 path=directory + '/NO/model/WNO_structure_main'):
		super().__init__(LB=LB, UB=UB, forced=forced, action_repeat=action_repeat)

		self.device = device
		self.lr = lr
		self.model = torch.load(path, map_location=self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

	def force_input(self, t, batch):
		# Wind input for the step from env step t to t+1: the wind at time t, held over the
		# step as in the real environment (env_tallstorey.py). t: scalar or per sample
		t = np.broadcast_to(np.asarray(t), (batch,))
		return np.stack([self.intensity(u_ref = self.force_fun(tk*self.dt)) for tk in t])

	def acceleration(self, D, V, ft, u):
		# Acceleration from the equation of motion for batched states, as in the real environment
		x = np.concatenate(( D[:, self.id_], V[:, self.id_] ), axis=1)       # Guyan: q = x[id_]
		ft = np.concatenate(( ft, np.zeros((ft.shape[0], 1)) ), axis=1)     # no wind on the TMD
		force = ft @ self.P_w.T + u @ self.P_h.T
		f_nl = self.alpha * (x[:, :self.rom_dof] @ self.T_nl.T)**3
		acc = (x @ self.sys.T)[:, self.rom_dof:] + force - f_nl @ self.P_nl.T
		return acc @ self.vec.T

	def reward(self, D, V, A, u):
		# Batched reward of the real environment (u: control force in kN)
		J = self.Qu[0] * np.einsum('bi,ij,bj->b', D - self.D_ref, self.Qx, D - self.D_ref) + \
            self.Qu[1] * np.einsum('bi,ij,bj->b', V - self.V_ref, self.Qx, V - self.V_ref) + \
            self.Qu[2] * np.einsum('bi,ij,bj->b', A - self.A_ref, self.Qx, A - self.A_ref)

		# Control effort: Add a penalty for large or inefficient control inputs.
		J += np.einsum('bi,ij,bj->b', u, self.Ru, u)
		return -J[:, None] # Return -ve of cost function as a reward

	def one_step(self, states, t, action):
		# One agent step from agent step index t: the action is held for action_repeat
		# environment steps (one network call each) and the rewards are summed, as in ActionRepeatWrapper
		batch = states.shape[0]
		u = self.control_force(action)

		# Extract the states from concatenated observations
		D, V = states[:,:self.dof], states[:,self.dof:2*self.dof]
		x = torch.tensor(np.stack((D, V), axis=1), dtype=torch.float32).to(self.device)
		uin = torch.tensor(u, dtype=torch.float32).to(self.device)

		r = 0
		self.model.eval()
		with torch.no_grad():
			for k in range(self.action_repeat):
				ft = self.force_input(t*self.action_repeat + k, batch)
				x = self.model(x, torch.tensor(ft, dtype=torch.float32).to(self.device), uin)
				out = x.cpu().numpy()
				D1, V1 = out[:,0,:], out[:,1,:]
				r += self.reward(D1, V1, self.acceleration(D1, V1, ft, u), u)

		# Do not overwrite the environment states (used by the planner)
		return out.reshape(batch, -1), r.astype('float32')

	def update(self, xx, yt, t, action):
		# One gradient step on observed agent-step transitions xx -> yt from agent step index t
		# (scalar or one per sample): the network is rolled out for action_repeat env steps
		batch = xx.shape[0]

		Dx, Vx = xx[:,:self.dof], xx[:,self.dof:]
		xx = torch.stack((Dx, Vx), axis=1).to(self.device)

		D, V = yt[:,:self.dof], yt[:,self.dof:]
		yt = torch.stack((D, V), axis=1).to(self.device)

		# Control force (kN):
		u = torch.as_tensor(self.control_force(action.detach().cpu().numpy()), dtype=torch.float32).to(self.device)

		self.model.train()
		loss = 0
		self.optimizer.zero_grad(set_to_none=True)

		out = xx
		for k in range(self.action_repeat):
			fin = torch.tensor(self.force_input(t*self.action_repeat + k, batch), dtype=torch.float32).to(self.device)
			out = self.model(out, fin, u)
		loss = torch.norm(yt - out)/torch.norm(yt)

		loss.backward()
		self.optimizer.step()


# %%
if __name__=='__main__':

	import timeit
	t0 = timeit.default_timer()

	# Call the environment:
	env = Skyscraper_rom()

	# Uncontrolled response predicted by the network:
	n = int(round(env.T/env.dt))
	env.reset()
	x = env.x0[None, :]
	for i in range(n):
		x, r = env.one_step(states=x, t=i, action=np.zeros((1, env.action_dim)))
		if i % int(max(1, n//10)) == 0: print('Step: {}, Reward: {:.4f}'.format(i, r[0,0]))

	print('{} steps, time: {:.2f} s'.format(n, timeit.default_timer() - t0))
