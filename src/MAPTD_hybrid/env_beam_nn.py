# -*- coding: utf-8 -*-
"""
This script defines the environment for the Beam
"""

import numpy as np
import matplotlib.pyplot as plt
from dm_env import specs, TimeStep, StepType
from collections import OrderedDict
from collections import defaultdict

import os
directory = os.path.abspath(os.path.join(os.path.dirname('NO'), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_convolution import WaveConv1d

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


class EulerBeam:
    def __init__(self, LB=None, UB=None, device=torch.device('cpu'),
                 path=directory + '/NO/model/WNO_beam_control2'):
        
        self.action_dim = 32*2
        self.Ne = 100   # actually, 2 times Ne, one for deflection and one for rotation
        self.D_ref = 0.01*np.ones(self.Ne)
        self.V_ref = 0.10*np.ones(self.Ne)
        self.A_ref = 1.00*np.ones(self.Ne)
        self.Qu = [1, 1, 0]
        self.Ru = 0.1*np.eye(self.action_dim)
        self.LB = LB if LB != None else -10000*np.ones(self.action_dim)
        self.UB = UB if UB != None else 10000*np.ones(self.action_dim)
        self.dt = 1e-2
        self.T = 1
        self.device = device
        
        self.get_properties(forced=False)
        self.newmark()
        
        self.device = device
        self.model = torch.load(path, map_location=self.device)        
        
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
        self.force_fun = lambda arg : 0 if forced == False else 0.05*np.sin(2*np.pi*arg)  
            
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
        
        # Intial Velocity:
        V0 = np.zeros(2*self.Ne)
        
        # Intial Acceleration:
        A0 = np.dot( np.linalg.inv(self.M), (0 - np.dot(self.K,D0)- np.dot(self.C,V0)) )      # initial acceleration
        
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
        # integration constant
        newmark_Beta = 1/4
        newmark_Gamma = 1/2
        
        self.nc1 = 1/newmark_Beta/self.dt**2
        self.nc2 = newmark_Gamma/newmark_Beta/self.dt
        self.nc3 = 1/newmark_Beta/self.dt
        self.nc4 = 1/2/newmark_Beta - 1
        self.nc5 = newmark_Gamma/newmark_Beta - 1
        self.nc6 = (newmark_Gamma/2/newmark_Beta - 1)*self.dt
        self.nc7 = (1 - newmark_Gamma)*self.dt
        self.nc8 = newmark_Gamma*self.dt
        
    def reward(self):
        # Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
        J = self.Qu[0] * np.sum((self.D[:,::2] - self.D_ref[None,:].repeat(self.batch,0))**2, axis=-1, keepdims=True) + \
            self.Qu[1] * np.sum((self.V[:,::2] - self.V_ref[None,:].repeat(self.batch,0))**2, axis=-1, keepdims=True) + \
            self.Qu[2] * np.sum((self.A[:,::2] - self.A_ref[None,:].repeat(self.batch,0))**2, axis=-1, keepdims=True)
        
        # Control effort: Add a penalty for large or inefficient control inputs.
        J += np.einsum('bi,ij,bj->b', self.action, self.Ru, self.action)[:, None]

        self.r = -J # Return -ve of cost function as a reward
        return self.r
    
    def one_step(self, states, t, action):
        self.batch = states.shape[0]
        # D, V, A = states[:,:2*self.Ne], states[:,2*self.Ne:4*self.Ne], states[:,4*self.Ne:]
        # states = np.stack((D, V, A), axis=-1)
        states = states.reshape(self.batch, 2*self.Ne, -1, order='F')
        
        # Location of control forces:
        u = np.zeros((self.batch, 2*self.Ne))
        u[:, 32:32*3] = action  # Control using Piezoelectric patch at 1/8th to 3/8th length
        
        # Force parameters: 
        force = np.zeros(2*self.Ne)
        force[::2] = self.force_fun(t*self.dt) * self.L / 2
        force[1::2] = self.force_fun(t*self.dt) * self.L**2 / 12
        self.force = np.repeat(force[None, :], axis=0, repeats=self.batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        fin = torch.tensor(self.force[:,:,None], dtype=torch.float32).to(self.device)
        uin = torch.tensor(u[:,:,None], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            out = self.model(states, fin, uin).cpu().numpy()
            
        self.D = out[:,:,0]
        self.V = out[:,:,1]
        self.A = out[:,:,2]
        self.action = action
        
        return out.reshape(self.batch, -1, order='F'), self.reward()
    
    def done(self):
        return True if self.t > int(self.T/self.dt) else False
    
    def info(self):
        return defaultdict(float)
    

# %%    
if __name__=='__main__':    
    # Call the environment:
    env = EulerBeam()
    
    # Initialize the states:
    n = int(env.T/env.dt)
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
        
        D[:, i+1] = states[0, :2*env.Ne]
        A[:, i+1] = states[0, 2*env.Ne:4*env.Ne]
        V[:, i+1] = states[0, 4*env.Ne:]

    
