# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:32:47 2024

Codes for free vibration of a cantilever
"""

import numpy as np
import matplotlib.pyplot as plt

class EulerBeam:
    def __init__(self, T=1, dt=0.001, forced=False, LB=None, UB=None):
        
        self.dt = dt
        self.T = T
        self.Nt = int(self.T/self.dt)
        self.action_dim = 32*2
        self.phorizon = 5
        self.Ne = 100   # actually, 2 times Ne, one for deflection and one for rotation
        self.D_ref = 1e-5*np.ones(self.Ne)
        self.V_ref = 1e-5*np.ones(self.Ne)
        self.A_ref = 1e-5*np.ones(self.Ne)
        self.Qu = [0.75, 0.25, 0]
        self.R = 0.5
        self.Ru = 0.5*np.eye(self.action_dim)
        self.LB = LB if LB != None else -2*np.ones(self.phorizon * self.action_dim)
        self.UB = UB if UB != None else 2*np.ones(self.phorizon * self.action_dim)
        self.forced = forced
        
        self.get_properties()
        self.newmark()
        
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
    
    def get_properties(self):
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
        self.force_fun = lambda arg : 0 if self.forced == False else 0.05*np.sin(2*np.pi*arg) 
            
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
        
        return (D0, V0, A0)
    
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

    def cost(self, action, states, t, old_action):        
        # Set initial plant states, controller output and cost:
        Dk, Vk, Ak = states
        action = action.reshape(self.phorizon, -1)
        uk = action[0,:]
        J = 0
        for ct in range(self.phorizon):     # Loop through each prediction step        
            # Obtain plant state at next prediction step:
            Dk1, Vk1, Ak1 = self.step(states=(Dk, Vk, Ak), t=t+ct, action=uk)
            
            # Accumulate state tracking cost from x(k+1) to x(k+N):
            J = J + self.Qu[0] * np.linalg.norm(np.abs(Dk1[::2]) - self.D_ref) + \
                    self.Qu[1] * np.linalg.norm(np.abs(Vk1[::2]) - self.V_ref)
            
            # Accumulate MV rate of change cost from u(k) to u(k+N-1):
            if ct == 0:
                J = J + np.dot(np.dot(np.transpose(uk-old_action), self.R), (uk-old_action)) + \
                    np.dot(np.dot(np.transpose(uk), self.Ru), uk)
            else:
                J = J + np.dot(np.dot(np.transpose(uk-action[ct-1,:]), self.R), (uk-action[ct-1,:])) + \
                    np.dot(np.dot(np.transpose(uk), self.Ru), uk)
                
            # Update xk and uk for the next prediction step:
            Dk, Vk, Ak = Dk1, Vk1, Ak1
            if ct < (self.phorizon-1):
                uk = action[ct+1,:]
                
        self.J = J
        return self.J
    
    def step(self, states, t, action):
        # Extract the states from concatenated observations
        D, V, A = states
        
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
        
        return (D1, V1, A1)
    
    def solve(self, action=0):
        # -------------------------------------------------------------------------
        D = np.zeros([2*self.Ne, self.Nt])
        V = np.zeros([2*self.Ne, self.Nt])
        A = np.zeros([2*self.Ne, self.Nt])
        
        D0, V0, A0 = self.reset()
        
        D[:, 0] = D0
        V[:, 0] = V0
        A[:, 0] = A0
        # -------------------------------------------------------------------------
        for ii in range(self.Nt-1):
            D0 = D[:, ii]
            V0 = V[:, ii]
            A0 = A[:, ii]
            
            D1, V1, A1 = self.step(states=(D0, V0, A0), t=ii, action=action)
            
            D[:, ii+1] = D1
            A[:, ii+1] = A1
            V[:, ii+1] = V1
        
        return np.stack(( D, V, A ))
    

# %%
if __name__=='__main__':    
    # The time parameters: 
    T, dt = 1, 1e-2
    Nt = int(T/dt)
    
    # Call the environment:
    env = EulerBeam(T=T, dt=dt, forced=False)
    
    # Solve the system:
    xt = env.solve()
    
    # %% Plot the data for visualization:
    fig1, ax = plt.subplots(nrows=3, ncols=1, figsize=(10,6), dpi=100)
    plt.subplots_adjust(hspace=0.3)
    
    label = ['$u(t)$', '$\dot{u}(t)$', '$\ddot{u}(t)$']
    for i in range(3):
        im = ax[i].imshow(xt[i, ::2].T, extent=[0,env.L,0,T], cmap='jet', origin='lower', aspect='auto'); 
        ax[i].set_ylabel(label[i]); 
        plt.colorbar(im, ax=ax[i], pad=0.01)
    plt.show()


