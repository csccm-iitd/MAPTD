# -*- coding: utf-8 -*-
"""
This script defines the environment for the Skyscraper
"""

from functools import partial
import numpy as np
import scipy.io as sio
from numpy.linalg import inv
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from dm_env import specs, TimeStep, StepType
from collections import OrderedDict
from collections import defaultdict

import os
directory = os.path.abspath(os.path.join(os.path.dirname('MPC'), '..'))

import torch
import torch.nn as nn

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


"""
Codes for free vibration of a cantilever
"""
class Skyscraper_rom:
    def __init__(self, LB=None, UB=None, forced=True, lr=1e-3, device=torch.device('cpu'),
                 path=directory + '/NO/model/WNO_structure_main'):
        
        # Initialize the scyscraper properties
        self.building()
        self.windpressure()
        
        self.action_dim = 12
        self.D_ref = 0.001*np.ones(self.dof)
        self.V_ref = 0.010*np.ones(self.dof)
        self.A_ref = 1.000*np.ones(self.dof)
        self.Qu = [1, 1, 1]
        # self.Qx = np.diag( np.linspace(1/self.dof,1,self.dof) )
        self.Qx = np.eye( self.dof )
        self.R = 0.1*np.eye(self.action_dim)
        self.Ru = 0.1*np.eye(self.action_dim)
        self.LB = LB if LB != None else -25000*np.ones(self.action_dim)
        self.UB = UB if UB != None else 25000*np.ones(self.action_dim)
        self.forced = forced
        
        self.device = device
        self.lr = lr
        self.model = torch.load(path, map_location=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
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
        # The 76 storey building properties
        data = sio.loadmat('data/B76_inp.mat')
        M76 = data['M76']
        K76 = data['K76']
        C76 = data['C76']

        # Get the natural frequencies and modes:
        eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(M76),K76))

        # Get the first modal mass:
        idx = np.where(eigval == np.min(eigval))
        phi = self.normalize(eigvec)
        modal_m = np.matmul( np.dot(phi[:,idx[0]].T, M76), phi[:,idx[0]] ).squeeze()

        # Modal mass ratio:
        md = 500 
        mu = md/modal_m  # mass ratio

        # Optimum damper parameters:
        fopt = 1/(1+mu)
        zopt = np.sqrt((3*mu)/(8*(1+mu)))

        # Find the optimum stiffness and damping
        omega = np.sqrt(eigval)/2/np.pi
        kd = (fopt**2 * omega[idx]**2 * md).squeeze()
        cda = (2 * zopt * fopt * omega[idx] * md).squeeze()

        M77 = np.zeros([77,77])
        K77 = np.zeros([77,77])
        C77 = np.zeros([77,77])
        M77[:76,:76] = M76
        M77[-1,-1] = md

        K77[:76,:76] = K76
        K77[-2,-2] = K76[-2,-2] + kd
        K77[-2,-1] = -kd
        K77[-1,-2] = -kd
        K77[-1,-1] = kd

        C77[:76,:76] = C76
        C77[-2,-2] = C76[-2,-2] + cda
        C77[-2,-1] = -cda
        C77[-1,-2] = -cda
        C77[-1,-1] = cda
        
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
        self.T = 0.5
        
        # Statespace matrix
        self.sys = np.row_stack(( np.column_stack(( np.zeros(self.M.shape), np.eye(self.M.shape[0]) )), \
                                  np.column_stack(( -np.matmul(np.linalg.inv(self.M),self.K), 
                                                    -np.matmul(np.linalg.inv(self.M),self.C) )) ))
        
        # Force function: 
        self.force_fun = lambda arg : 0 if self.forced == False else np.sin(np.pi*arg) 
        
        # Define the excitation influence matrix:
        self.dof = self.Mt.shape[0]
        self.rom_dof = self.M.shape[0]
        
        self.sigma = np.eye(self.dof)
        self.sigma[-1,-1] = 0
        
        # Define control influence matrix:        
        c_idx = np.array([5, 10, 15, 25, 35, 45, 55, 60, 65, 72, 75, 76])
        H = np.zeros((77, len(c_idx)))                      # Influence matrix
        H[(c_idx, np.arange(len(c_idx)))] = 1
        
        self.H = H
        
    # Perform Model-order reduction:
    def rom(self,m,k,c,order=None):
        
        # Get the natural frequencies and modes:
        eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(m),k))
        
        # Sort the eigen-values in ascending order:
        val = np.sqrt( np.sort(eigval) )
        
        # If order is not given, find the minimum orde using energy criteria:
        if order == None: 
            energy = 0.5    # 50\% enrgy of the total system
            sum_ = 0
            order = 0
            for value in val:
                sum_ += value
                order += 1
                if sum_ > energy * np.sum(val):
                    break
        
        # Model-order reduction equation:
        self.rom_system = lambda matrix, phi: np.matmul( np.matmul(phi.T, matrix), phi )
        
        # Truncate the modes:
        if isinstance(order, list):
            self.val = -np.sort(-val)[order] 
            self.vec = eigvec[:, order]
        else:
            self.val = -np.sort(-val[:order])
            self.vec = eigvec[:, -order:]
        
        # Get the reduced-order matrices:
        m_rom = self.rom_system(m, self.vec) 
        k_rom = self.rom_system(k, self.vec) 
        c_rom = self.rom_system(c, self.vec) 
        return (m_rom, k_rom, c_rom)
    
    def windpressure(self):
        # Wind load calculation:
        L = 42          # width of building
        z_ref = 10      # atmospheric roughness length
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
        
        # Define the forcing function:
        fun = lambda rho, L, cd, u_ref, z_ref, z: 0.5*rho*L*cd*u_ref**2*(z/z_ref)**0.365
        
        # Compressing the forcing function:
        self.intensity = partial(fun, rho=rho, L=L, cd=cd, z_ref=z_ref, z=z)
        
    def reset(self):
        # Intial state vector:
        D0 = 0 * np.concatenate(( np.ones(self.dof-1), np.zeros(1) ))
        V0 = 0 * np.zeros(self.dof)
        A0 = 0 * np.zeros(self.dof)
        
        # Statespace vector containing only diplacement and velocity
        # self.x0 = np.concatenate(( D0, V0 ))
        
        # Store the system matrices:
        self.D = D0
        self.V = V0
        self.A = A0
        self.t = 0
        self.r = 0
        
        dic = OrderedDict([('displacement', D0),
                            ('velocity', V0)])
        # dic = OrderedDict([('states', self.x0)])
        
        obs = TimeStep(step_type=StepType.FIRST,
                       reward=None,
                       discount=None,
                       observation=dic)
        return obs
    
    # first-order statespace equation:
    def dydt(self,x,t,force,action):
        # Calculate the wind force distribution over the surface
        # Transform to reduced dimension:
        force = np.matmul(self.vec.T, np.matmul(self.sigma,force))
        force = np.concatenate(( np.zeros_like(force), force ))
        
        # Define control force        
        control_force = np.matmul(self.vec.T, np.matmul(self.H,action))
        control_force = np.concatenate(( np.zeros_like(control_force), control_force ))
        
        # Evaluate the derivatives
        alpha = -1e6 * np.ones(self.dof)
        xrom = np.matmul(self.vec, x[:24])
        nonlin = np.zeros(self.dof)
        nonlin[35] = alpha[35] * (xrom[35])**3
        nonlin[52] = alpha[52] * (xrom[52])**3
        nonlin[75] = alpha[75] * (xrom[75])**3
        
        nonlin = np.matmul(self.vec.T, nonlin)
        nonlin = np.concatenate(( np.zeros_like(nonlin), nonlin ))
        
        dydt = np.matmul(self.sys,x) + nonlin + control_force + force
        return dydt
    
    def reward(self):
        # Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
        J = self.Qu[0] * np.sum((self.D - self.D_ref)**2, axis=-1, keepdims=True) + \
            self.Qu[1] * np.sum((self.V - self.V_ref)**2, axis=-1, keepdims=True) #+ \
            # self.Qu[2] * np.sum((self.A - self.A_ref)**2, axis=-1, keepdims=True)
        
        # Control effort: Add a penalty for large or inefficient control inputs.
        J += np.einsum('bi,ij,bj->b', self.action, self.Ru, self.action)[:, None]

        self.r = -J # Return -ve of cost function as a reward
        return self.r
    
    def one_step(self, states, t, action):
        self.batch = states.shape[0]
        # Extract the states from concatenated observations
        D, V = states[:,:self.dof], states[:,self.dof:2*self.dof]
        states = np.stack((D, V), axis=1)
        
        # Time integration:
        ft = self.intensity(u_ref = self.force_fun(t*self.dt))
        ft = np.repeat(ft[None, :], axis=0, repeats=self.batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        fin = torch.tensor(ft, dtype=torch.float32).to(self.device)
        uin = torch.tensor(action, dtype=torch.float32).to(self.device)
        with torch.no_grad(): 
            out = self.model(states, fin, uin).cpu().numpy()
        
        D1 = out[:,0,:]
        V1 = out[:,1,:]

        # Compute acceleration response:
        A1 = -np.matmul(D1, np.matmul(inv(self.Mt),self.Kt).T) \
                -np.matmul(V1, np.matmul(inv(self.Mt),self.Ct).T) \
                +np.matmul(np.concatenate((ft, np.zeros((self.batch, 1))), axis=1), np.matmul(inv(self.Mt),self.sigma).T) \
                +np.matmul(action, np.matmul(inv(self.Mt),self.H).T)
                
        self.D = D1
        self.V = V1
        self.A = A1
        self.action = action
        
        out = out.reshape(self.batch, -1)
        return out, self.reward()        
    
    def update(self, xx, yt, t, action):
        batch = xx.shape[0]
        
        Dx, Vx = xx[:,:self.dof], xx[:,self.dof:]
        xx = torch.stack((Dx, Vx), axis=1).to(self.device)
        
        D, V = yt[:,:self.dof], yt[:,self.dof:]
        yt = torch.stack((D, V), axis=1).to(self.device)
        
        # Location of control forces:
        action = action.to(self.device)
        
        # Force parameters: 
        fin = self.intensity(u_ref = self.force_fun(t*self.dt))        
        fin = np.repeat(fin[None, :], axis=0, repeats=batch)        
        fin = torch.tensor(fin, dtype=torch.float32).to(self.device)
        
        self.model.train()
        loss = 0
        self.optimizer.zero_grad(set_to_none=True)
        
        out = self.model(xx, fin, action)
        loss = torch.norm(yt - out)/torch.norm(yt)
        
        loss.backward()
        self.optimizer.step()
        
    def done(self):
        return True if self.t > int(self.T/self.dt) else False
    
    def info(self):
        return defaultdict(float)
    

# %%
if __name__=='__main__':
    
    import timeit
    t0 = timeit.default_timer()
    
    # Call the environment:
    env = Skyscraper_rom()
     
    # Initialize the states:
    n = int(env.T/env.dt)
    m = env.dof
    
    D = np.zeros([m,n])
    V = np.zeros([m,n])
    A = np.zeros([m,n])
    
    obs = env.reset()
    D[:, 0] = env.D
    V[:, 0] = env.V
    A[:, 0] = env.A
    
    # Time integration:
    for i in range(n-1):  
        D0 = D[:, i]
        V0 = V[:, i]
        A0 = A[:, i]
        
        states, r = env.one_step(states=np.concatenate((D0,V0,A0))[None, :],
                                 t=i, action=0*np.ones(env.action_dim)[None, :])
        
        env.update(np.concatenate((D0,V0))[None, :], np.concatenate((D0,V0))[None, :],
                                 t=i, action=0*np.ones(env.action_dim)[None, :])
                
        D[:,i+1] = env.D
        V[:,i+1] = env.V
        A[:,i+1] = env.A

