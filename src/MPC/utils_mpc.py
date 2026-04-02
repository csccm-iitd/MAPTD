# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:40:25 2024

@author: tripu
"""

import numpy as np
# from omegaconf import OmegaConf as omg

def beam_objective(u, arguments):
    """
    It generates the Cost function to optimize:

    Parameters
    ----------
    u : vector of control variables, from time k to time k+N-1 
        Size: N, length of prediction horizon.
        
    arguments : list of list. 
        It contains the following lists:
            [sysfun, sparams, x0, F, control_params]
        
        Variables in input lists:
            Ma: Mass matrix of beam
            Ca: Damping matrix of beam
            Ka: Stiffness matrix of beam
            
            xref:   state references, constant from time k+1 to k+N
            u0:     previous controller output at time k-1
            N:      prediction horizon
            Q:      particification factor of displacement and velocity
            R:      control weight of control variables
            Ru:     control weight of change in control variables
            
            Dk: Initial displacement 
            Vk: Initial velocity
            Ak: Intial acceleration

    Returns
    -------
    J : function handler for cost function.

    """
    # Unload the variables:
    sysfun, sparams, x0, F, control_params = arguments
    Ma, Ca, Ka, L = sparams
    xref, u0, N, Q, R, Ru = control_params
    Dk, Vk, Ak = x0
    
    # Set initial plant states, controller output and cost:
    uk = u[0]
    J = 0
    for ct in range(N):     # Loop through each prediction step        
        # Obtain plant state at next prediction step:
        Dk1, Vk1, Ak1 = sysfun(Ma,Ca,Ka,L,F,uk,Dk,Vk,Ak) 
        
        # Accumulate state tracking cost from x(k+1) to x(k+N):
        J = J + Q[0] * np.sum((Dk1[::2] - xref)**2) + Q[1] * np.sum((Vk1[::2] - xref)**2)
        
        # Accumulate MV rate of change cost from u(k) to u(k+N-1):
        if ct == 0:
            J = J + np.dot(np.dot(np.transpose(uk-u0), R), (uk-u0)) + \
                np.dot(np.dot(np.transpose(uk), Ru), uk)
        else:
            J = J + np.dot(np.dot(np.transpose(uk-u[ct-1]), R), (uk-u[ct-1])) + \
                np.dot(np.dot(np.transpose(uk), Ru), uk)
            
        # Update xk and uk for the next prediction step:
        Dk, Vk, Ak = Dk1, Vk1, Ak1
        if ct < (N-1):
            uk = u[ct+1]
    return J


def L2norm(D,xref,V=None):
    
    error_d = np.linalg.norm(D - xref)/np.linalg.norm(xref)
    if V.any() == None:
        return error_d
    else:
        error_v = np.linalg.norm(V - xref)/np.linalg.norm(xref)
        return error_d, error_v

# def parse_yaml(path):
#     base = omg.load(path)

#     # Algebraic expressions
#     for k,v in base.items():
#         if isinstance(v, str):
#             base[k] = eval(v)
#     return base

def store_force(fun, arg, L, Ne):
    # Force parameters: 
    force = np.zeros(2*Ne)
    force[::2] = fun(arg) * L / 2
    force[1::2] = fun(arg) * L**2 / 12  
    return force

def store_action(arg, Ne):
    # Control parameters:
    action = np.zeros(2*Ne)
    action[32:32*3] = arg
    return action


