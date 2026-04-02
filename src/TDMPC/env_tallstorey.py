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

"""
Codes for free vibration of a cantilever
"""
class Skyscraper:
    def __init__(self, LB=None, UB=None, forced=True):
        
        # Initialize the scyscraper properties
        self.building()
        self.windpressure()
        
        self.action_dim = 16
        self.D_ref = 0.1*np.ones(self.dof)
        self.V_ref = 0.1*np.ones(self.dof)
        self.A_ref = 1*np.ones(self.dof)
        self.Qu = [0.5, 0.5, 0]
        # self.Qx = np.diag( np.linspace(1/self.dof,1,self.dof) )
        self.Qx = np.eye( self.dof )
        self.Ru = 0.5*np.eye(self.action_dim)
        self.LB = LB if LB != None else -250*np.ones(self.action_dim)
        self.UB = UB if UB != None else 250*np.ones(self.action_dim)
        self.forced = forced
        
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
        
        self.M = M77
        self.C = C77
        self.K = K77
        
        # Time step
        self.dt = 1e-2
        self.T = 10
        
        # Statespace matrix
        self.sys = np.row_stack(( np.column_stack(( np.zeros(M77.shape), np.eye(M77.shape[0]) )), \
                                  np.column_stack(( -np.matmul(np.linalg.inv(M77),K77), 
                                                    -np.matmul(np.linalg.inv(M77),C77) )) ))
        
        # Force function: 
        self.force_fun = lambda arg : 0 if self.forced == False else 0.2*np.sin(np.pi*arg)
        
        # Define the excitation influence matrix:
        self.dof = self.M.shape[0]
        sigma = np.concatenate(( np.zeros((self.dof,self.dof)), np.eye(self.dof) ))
        sigma[-1,-1] = 0
        
        # Define control influence matrix:        
        # c_idx = 77 + np.concatenate(( np.array([16]), np.arange(30,77,10), np.array([76, 77]) ))
        c_idx = 77 + np.concatenate(( np.arange(5,30,10), np.arange(30,75,5), np.array([72, 74, 76, 77]) ))
        H = np.zeros((154, len(c_idx)))             # Influence matrix
        H[(c_idx-1, np.arange(len(c_idx)))] = 1
        
        self.sigma = sigma
        self.H = H
        
    
    def windpressure(self):
        # Wind load calculation:
        L = 42          # width of building
        z_ref = 0.5     # atmospheric roughness length
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
        fun = lambda rho, L, cd, u_ref, z_ref, z: 0.5*rho*L*cd*u_ref**2*(z/z_ref)**0.32
        
        # Compressing the forcing function:
        self.intensity = partial(fun, rho=rho, L=L, cd=cd, z_ref=z_ref, z=z)
        
    
    def reset(self):
        # Intial state vector:
        D0 = 0.25 * np.concatenate(( np.ones(self.dof-1), np.zeros(1) ))
        V0 = 0.25 * np.zeros(self.dof)
        A0 = 0.25 * np.zeros(self.dof)
        
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
    def dydt(self,t,x,statespace,sigma,force,H,action):
        # Calculate the wind force distribution over the surface
        force = np.matmul(sigma,force)
        
        # Define control force        
        control_force = np.matmul(H,action) 
        
        # Evaluate the derivatives
        dydt = np.matmul(statespace,x) + control_force + force
        return dydt
    
    def step(self,action):
        # Time integration:
        ft = np.concatenate(( self.intensity(u_ref = self.force_fun(self.t*self.dt)), np.zeros(1) ))
        sol = solve_ivp(fun=self.dydt, t_span=[self.t*self.dt, self.t*self.dt+self.dt], y0=self.x0,
                        method='RK45', t_eval= None, 
                        args=(self.sys,self.sigma,ft,self.H,action))
        
        # Update the statespace vector:
        self.x0 = sol.y[:,-1]
        
        # Extract the displacement and velocities:
        D1 = self.x0[:self.dof]
        V1 = self.x0[self.dof:]
        
        # Compute acceleration response:
        A1 = -np.matmul(np.matmul(inv(self.M),self.K),D1) \
                -np.matmul(np.matmul(inv(self.M),self.C),V1) \
                +np.matmul(np.matmul(inv(self.M),self.sigma[self.dof:,:]),ft) \
                +np.matmul(np.matmul(inv(self.M),self.H[self.dof:,:]),action)
        
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
        # Time integration:
        ft = np.concatenate(( self.intensity(u_ref = self.force_fun(t*self.dt)), np.zeros(1) ))
        sol = solve_ivp(fun=self.dydt, t_span=[t*self.dt, t*self.dt+self.dt], y0=states,
                        method='RK45', t_eval= None, 
                        args=(self.sys,self.sigma,ft,self.H,action))
        
        # Update the statespace vector:
        self.x0 = sol.y[:,-1]
        
        # Extract the displacement and velocities:
        D1 = self.x0[:self.dof]
        V1 = self.x0[self.dof:]
        
        # Compute acceleration response:
        A1 = -np.matmul(np.matmul(inv(self.M),self.K),D1) \
                -np.matmul(np.matmul(inv(self.M),self.C),V1) \
                +np.matmul(np.matmul(inv(self.M),self.sigma[self.dof:,:]),ft) \
                +np.matmul(np.matmul(inv(self.M),self.H[self.dof:,:]),action)
        
        self.D = D1
        self.V = V1
        self.A = A1
        self.action = action
        
        return self.x0, np.array([self.reward()])
    
    def done(self):
        return True if self.t > int(self.T/self.dt) else False
    
    def info(self):
        return defaultdict(float)
    
    
class Skyscraper_rom:
    def __init__(self, LB=None, UB=None, forced=True):
        
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
        self.LB = LB if LB != None else -250*np.ones(self.action_dim)
        self.UB = UB if UB != None else 250*np.ones(self.action_dim)
        self.forced = forced
        
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
        self.T = 1
        
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
        alpha = -1e3
        nonlin = np.zeros(24)
        nonlin[10] = alpha * (x[10])**3
        nonlin[15] = alpha * (x[15])**3
        nonlin[22] = alpha * (x[22])**3
        
        nonlin = np.matmul(self.vec.T, np.matmul(self.vec, nonlin))
        nonlin = np.concatenate(( np.zeros_like(nonlin), nonlin ))
        
        dydt = np.matmul(self.sys,x) + nonlin + control_force + force
        return dydt
    
    def step(self,action):
        # Time integration:
        x0 = np.concatenate(( self.vec.T @ self.D, self.vec.T @ self.V ))
        
        ft = np.concatenate(( self.intensity(u_ref = self.force_fun(self.t*self.dt)), np.zeros(1) ))
        sol = odeint(func=self.dydt, t=[self.t*self.dt, self.t*self.dt+self.dt],
                     mxstep=1000, y0=x0, args=(ft,action))
        
        D1 = self.vec @ sol[-1,:self.rom_dof]
        V1 = self.vec @ sol[-1,self.rom_dof:]
        
        # Compute acceleration response:
        A1 = -np.matmul(np.matmul(inv(self.Mt),self.Kt),D1) \
                -np.matmul(np.matmul(inv(self.Mt),self.Ct),V1) \
                +np.matmul(np.matmul(inv(self.Mt),self.sigma),ft) \
                +np.matmul(np.matmul(inv(self.Mt),self.H),action)
        
        self.D = D1
        self.V = V1
        self.A = A1
        self.t += 1
        self.action = action
        
        dic = OrderedDict([('displacement', D1),
                            ('velocity', V1)])
        # dic = OrderedDict([('states', self.x0)])
        
        obs = TimeStep(step_type=StepType.MID,
                       reward=self.reward(),
                       discount=1.0,
                       observation=dic)
        return obs
    
    def matrix_norm(self,x,weight):
        return np.matmul(np.matmul(np.transpose(x), weight), x)
    
    def reward(self):
        # Minimize deflection: Reward the agent for keeping the beam's deflection close to zero.
        J = self.Qu[0] * np.sum((self.D - self.D_ref)**2) + \
            self.Qu[1] * np.sum((self.V - self.V_ref)**2) + \
            self.Qu[2] * np.sum((self.A - self.A_ref)**2)
        
        # Control effort: Add a penalty for large or inefficient control inputs.
        J += np.dot(np.dot(self.action, self.Ru), self.action)

        self.r = -J # Return -ve of cost function as a reward
        return self.r
    
    def one_step(self, states, t, action):
        # Extract the states from concatenated observations
        D, V = states[:self.dof], states[self.dof:] 
        x0 = np.concatenate(( self.vec.T @ D, self.vec.T @ V ))
        
        # Time integration:
        ft = np.concatenate(( self.intensity(u_ref = self.force_fun(t*self.dt)), np.zeros(1) ))
        sol = odeint(func=self.dydt, t=[t*self.dt, t*self.dt+self.dt], 
                     mxstep=1000, y0=x0, args=(ft,action))
        
        D1 = self.vec @ sol[-1,:self.rom_dof]
        V1 = self.vec @ sol[-1,self.rom_dof:]
        
        # Compute acceleration response:
        A1 = -np.matmul(np.matmul(inv(self.Mt),self.Kt),D1) \
                -np.matmul(np.matmul(inv(self.Mt),self.Ct),V1) \
                +np.matmul(np.matmul(inv(self.Mt),self.sigma),ft) \
                +np.matmul(np.matmul(inv(self.Mt),self.H),action)
        
        self.D = D1
        self.V = V1
        self.A = A1
        self.action = action
        
        return np.concatenate([D1, V1]), np.array([self.reward()])
    
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
        xt = env.step(action=0*np.ones(env.action_dim))
        
        if i % 20 == 0: print('Step: {}, Reward: {}'.format(i,xt.reward))
        
        D[:,i+1] = env.D
        V[:,i+1] = env.V
        A[:,i+1] = env.A
    
    print('Time: {}'.format(timeit.default_timer() - t0))
    
    # %%
    floor = [5,11,18,74]
    t = np.arange(0,env.T,env.dt)
    
    figure1, ax = plt.subplots(nrows=len(floor), ncols=2, figsize =(16,10), dpi=100)
    plt.subplots_adjust(hspace=0.35)
    
    for i in range(len(floor)):
        ax[i,0].plot(t, D[floor[i],:], label='DOF:{}th'.format(floor[i]))
        if i == len(floor)-1: ax[i,0].set_xlabel('Time (Sec)')
        ax[i,0].set_ylabel('$X({})$'.format(floor[i]))
        ax[i,0].grid(True, alpha=0.25) 
        
        ax[i,1].plot(t, V[floor[i],:], label='DOF:{}th'.format(floor[i]))
        if i == len(floor)-1: ax[i,1].set_xlabel('Time (Sec)')
        ax[i,1].set_ylabel('$X({})$'.format(floor[i]))
        ax[i,1].grid(True, alpha=0.25) 
    
    plt.suptitle('Response of 76 DOF Skyscraper (RL ENV)', y=0.94)
    plt.show()

