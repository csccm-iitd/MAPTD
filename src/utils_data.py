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
import utils

"""
Code for string vibration in FINITE DIFFERENCE
"""
def string(L,stopTime,c,dx=0.01,dt=0.001):    
    # dx = 0.01   # Spacing of points on string
    # dt = 0.001  # Size of time step
    # c = 5  # Speed of wave propagation
    # L = 10 # Length of string
    # stopTime = 5 # Time to run the simulation
    
    r = c*dt/dx 
    n = int(L/dx + 1)
    t  = np.arange(0, stopTime+dt, dt)
    mesh = np.arange(0, L+dx, dx)
    sol = np.zeros([len(mesh), len(t)])
    
    # Set current and past to the graph of a plucked string
    current = 0.1 - 0.1*np.cos( 2*np.pi/L*mesh ) 
    past = current
    sol[:, 0] = current
    
    for i in range(len(t)):
        future = np.zeros(n)
    
        # Calculate the future position of the string
        future[0] = 0 
        future[1:n-2] = r**2*( current[0:n-3]+ current[2:n-1] ) + 2*(1-r**2)*current[1:n-2] - past[1:n-2]
        future[n-1] = 0 
        sol[:, i] = current
        
        # Settings up for the next time step
        past = current 
        current = future 
    
    Vel = np.zeros([sol.shape[0], sol.shape[1]])
    for i in range(1, sol.shape[1]-1):
        Vel[:,i] = (sol[:,i+1] - sol[:,i-1])/(2*dt)
    Vel[:,0] = (-3.0/2*sol[:,0] + 2*sol[:,1] - sol[:,2]/2) / dt
    Vel[:,sol.shape[1]-1] = (3.0/2*sol[:,sol.shape[1]-1] - 2*sol[:,sol.shape[1]-2] + sol[:,sol.shape[1]-3]/2) / dt
    
    xt = np.zeros([2*sol.shape[0],sol.shape[1]])
    xt[::2] = sol
    xt[1::2] = Vel
    return xt


"""
Codes for free vibration of a cantilever
"""
def cantilever(sparams,tparams,force,Ne=100):
    rho, E, b, d, A, I, L, mode, c1, c2 = sparams
    Ne, T, dt = tparams
    t_eval = np.arange(0, T, dt)
    xx = np.arange(L/Ne, L+L/Ne, L/Ne)
    
    [Ma, Ka, _, _] = beam_solver.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')
    
    Ca = (c1*Ma + c2*Ka)
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

def kcmat(k,dof):
    mat = np.zeros([dof, dof])
    for i in range(dof):
        if i == 0:
            mat[i, 0] = (k[0]+k[1])
            mat[i, 1] = -k[1]
        elif i == dof-1:
            mat[i, i-1] = -k[-1]
            mat[i, i] = k[-1]
        else:
            mat[i, i-1] = -k[i]
            mat[i, i] = (k[i]+k[i+1])
            mat[i, i+1] = -k[i+1]
    return mat

# Perform Model-order reduction:
def rom(m,k,c,order=None):
    
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
    rom_system = lambda matrix, phi: np.matmul( np.matmul(phi.T, matrix), phi )
    
    # Truncate the modes:
    if isinstance(order, list):
        val = -np.sort(-val)[order] 
        vec = eigvec[:, order]
    else:
        val = -np.sort(-val[:order])
        vec = eigvec[:, -order:]
    
    # Get the reduced-order matrices:
    m_rom = rom_system(m, vec) 
    k_rom = rom_system(k, vec) 
    c_rom = rom_system(c, vec) 
    return (m_rom, k_rom, c_rom, vec)

# The 76 storey building properties 
def skyscraper():
    data = sio.loadmat('data/B76_inp.mat')
    M76 = data['M76']
    K76 = data['K76']
    C76 = data['C76']
    
    # Get the natural frequencies and modes:
    eigval, eigvec = np.linalg.eig(np.matmul(np.linalg.inv(M76),K76))
    
    # Get the first modal mass:
    ms = 153000 
    idx = np.where(eigval == np.min(eigval))
    phi = normalize(eigvec)
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
    
    # Tuned-mass augmented system:
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
    
    id_ = [2, 5, 9, 12, 15, 19, 22, 25, 29, 32, 35, 39, 
            42, 45, 49, 52, 55, 59, 62, 65, 69, 72, 75, 76]
    
    return M77,K77,C77,id_

def solve_76dof(M,C,K,ic,T,dt):
    # Statespace matrix
    sys = np.row_stack(( np.column_stack(( np.zeros(M.shape), np.eye(M.shape[0]) )), \
                         np.column_stack(( -np.matmul(np.linalg.inv(M),K), 
                                           -np.matmul(np.linalg.inv(M),C) )) ))
        
    def dydt(x,t,statespace,non,sigma,force,iM):
        
        # Calculate the wind force distribution over the surface
        force = np.concatenate(( intensity(u_ref = np.sin(np.pi*t)), np.zeros(1) ))
        
        # Transform to reduced dimension:
        force = iM @ sigma @ force
        force = np.concatenate(( np.zeros(force.shape[0]), force ))
        
        # Evaluate the nonlinearity
        nonlin = np.matmul(iM, non) @ x[:77]**3
        nonlin = np.concatenate(( np.zeros(nonlin.shape[0]), nonlin ))

        dydt = np.matmul(statespace,x) + nonlin + force
        return dydt
        
    # Wind load calculation:
    L = 42
    z_ref = 10
    # u_ref = 1
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
    
    # Define the forcing function:
    fun = lambda rho, L, cd, u_ref, z_ref, z: 0.0*rho*L*cd*u_ref**2*(z/z_ref)**0.365
    
    # Compressing the forcing function:
    intensity = partial(fun, rho=rho, L=L, cd=cd, z_ref=z_ref, z=z)
    
    # Define the excitation influence matrix:
    dof = M.shape[0]
    sigma = np.eye(dof)
    sigma[-1,-1] = 0
    
    # Inverse of mass matrix
    iM = inv(M)
    
    # Non-linearity:
    alpha = -1e13
    nonlin = np.zeros(77) 
    nonlin[[35,52,70]] = alpha
    nonlin = np.diag(nonlin)
    
    # Solve the system:
    t_eval = np.arange(0,T+dt,dt)
    sol = odeint(func=dydt, y0=ic, t=t_eval, mxstep=1000, args=(sys,nonlin,sigma,intensity,iM))
    sol = sol.transpose(1,0)
    
    ft = [ np.concatenate(( intensity(u_ref = np.sin(np.pi*t)), np.zeros(1) )) for t in t_eval ]
    ft = np.column_stack(( ft ))[:, :-1]
    
    # Extract the displacement and velocities:
    states = len(ic)//2
    D = sol[:states, :-1]
    V = sol[states:, :-1]
       
    A = []
    for i in range(dof):
        A.append( utils.FiniteDiff(V[i], dx=dt, d=1) )
    A = np.stack((A))
    return t_eval[:-1], np.stack(( D,V,A )), ft
    
