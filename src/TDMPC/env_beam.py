# -*- coding: utf-8 -*-
"""
This script defines the environment for the Beam
"""

import numpy as np
import matplotlib.pyplot as plt
from dm_env import specs, TimeStep, StepType
from collections import OrderedDict
from collections import defaultdict

"""
Codes for free vibration of a cantilever
"""
class EulerBeam:
    def __init__(self, LB=None, UB=None):
        
        self.action_dim = 32*2
        self.Ne = 100   # actually, 2 times Ne, one for deflection and one for rotation
        self.D_ref = 0.01*np.ones(self.Ne)
        self.V_ref = 0.10*np.ones(self.Ne)
        self.A_ref = 1.00*np.ones(self.Ne)
        self.Qu = [1, 1, 0]
        self.Ru = 0.1*np.eye(self.action_dim)
        self.LB = LB if LB != None else -1*np.ones(self.action_dim)
        self.UB = UB if UB != None else 1*np.ones(self.action_dim)
        
        self.get_properties(forced=False)
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
        # Time step
        self.dt = 1e-2
        self.T = 1
        
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

    def step(self,action):
        Kbar = self.nc1*self.M + self.nc2*self.C + self.K          # linear time-invariant system
        
        # Location of control forces:
        u = np.zeros(2*self.Ne)
        u[32:32*3] = action  # Control using Piezoelectric patch at 1/8th to 3/8th length
        
        # Force parameters: 
        self.force = np.zeros(2*self.Ne)
        self.force[::2] = self.force_fun(self.t*self.dt) * self.L / 2
        self.force[1::2] = self.force_fun(self.t*self.dt) * self.L**2 / 12
        
        Fbar = self.force + np.dot(self.M, (self.nc1*self.D + self.nc3*self.V + self.nc4*self.A)) \
                          + np.dot(self.C, (self.nc2*self.D + self.nc5*self.V + self.nc6*self.A)) + u
                          
        D1 = np.matmul(np.linalg.inv(Kbar), Fbar)
        A1 = self.nc1*(D1 - self.D) - self.nc3*self.V - self.nc4*self.A
        V1 = self.V + self.nc7*self.A + self.nc8*A1 
        
        self.D = D1
        self.V = V1
        self.A = A1
        self.t += 1
        self.action = action
        
        dic = OrderedDict([('displacement', D1),
                            ('velocity', V1),
                            ('acceleration', A1)])
        # dic = OrderedDict([('displacement', D1)])
        
        obs = TimeStep(step_type=StepType.MID,
                       reward=self.reward(),
                       discount=1.0,
                       observation=dic)
        return obs
    
    def reward(self):
        # Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
        J = self.Qu[0] * np.sum((self.D[::2] - self.D_ref)**2) + \
            self.Qu[1] * np.sum((self.V[::2] - self.V_ref)**2) + \
            self.Qu[2] * np.sum((self.A[::2] - self.A_ref)**2)
        
        # Control effort: Add a penalty for large or inefficient control inputs.
        J += np.dot(np.dot(self.action, self.Ru), self.action)

        self.r = -J # Return -ve of cost function as a reward
        return self.r
    
    def one_step(self, states, t, action):
        # Extract the states from concatenated observations
        D, V, A = states[:2*self.Ne], states[2*self.Ne:4*self.Ne], states[4*self.Ne:]
        
        Kbar = self.nc1*self.M + self.nc2*self.C + self.K          # linear time-invariant system
        
        # Location of control forces:
        u = np.zeros(2*self.Ne)
        u[32:32*3] = action  # Control using Piezoelectric patch at 1/8th to 3/8th length
        
        # Force parameters: 
        self.force = np.zeros(2*self.Ne)
        self.force[::2] = self.force_fun(t*self.dt) * self.L / 2
        self.force[1::2] = self.force_fun(t*self.dt) * self.L**2 / 12
        
        Fbar = self.force + np.dot(self.M, (self.nc1*D + self.nc3*V + self.nc4*A)) \
                          + np.dot(self.C, (self.nc2*D + self.nc5*V + self.nc6*A)) + u
                          
        D1 = np.matmul(np.linalg.inv(Kbar), Fbar)
        A1 = self.nc1*(D1 - D) - self.nc3*V - self.nc4*A
        V1 = V + self.nc7*A + self.nc8*A1 
        
        self.D = D1
        self.V = V1
        self.A = A1
        self.action = action
        
        return np.concatenate([D1, V1, A1]), np.array([self.reward()])
    
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
    fig2, ax = plt.subplots(ncols=1, nrows=3, figsize=(12,6), dpi=100)
    im = ax[0].imshow(D.T, cmap='jet', origin='lower', aspect='auto'); 
    ax[0].set_ylabel('$u(t)$'); plt.colorbar(im, ax=ax[0], pad=0.01)
    
    im = ax[1].imshow(V.T, cmap='jet', origin='lower', aspect='auto'); 
    ax[1].set_ylabel('$u(t)$'); plt.colorbar(im, ax=ax[1], pad=0.01)
    
    im = ax[2].imshow(A.T, cmap='jet', origin='lower', aspect='auto'); 
    ax[2].set_ylabel('$u(t)$'); plt.colorbar(im, ax=ax[2], pad=0.01)
    plt.show()

    
