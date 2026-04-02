#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:35:49 2024

@author: user
"""

import torch
import numpy as np
import pickle

"""
For numerical derivative using 4th order accuarcy
"""
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=4)
    Works but with poor accuracy for d > 4
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n)
    
    if d == 1:
        ux[1:n-1] = (u[2:n]-u[0:n-2]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        ux[1:n-1] = (u[2:n]-2*u[1:n-1]+u[0:n-2]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        ux[2:n-2] = (u[4:n]/2-u[3:n-1]+u[1:n-3]-u[0:n-4]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d == 4:
        ux[2:n-2] = (u[4:n]-4*u[3:n-1]+6*u[2:n-2]-4*u[1:n-3]+u[0:n-4]) / dx**4
        
        ux[0] = (3*u[0]-14*u[1]+26*u[2]-24*u[3]+11*u[4]-2*u[5]) / dx**4
        ux[1] = (3*u[1]-14*u[2]+26*u[3]-24*u[4]+11*u[5]-2*u[6]) / dx**4
        ux[n-1] = (3*u[n-1]-14*u[n-2]+26*u[n-3]-24*u[n-4]+11*u[n-5]-2*u[n-6]) / dx**4
        ux[n-2] = (3*u[n-2]-14*u[n-3]+26*u[n-4]-24*u[n-5]+11*u[n-6]-2*u[n-7]) / dx**4
        return ux
    
    if d > 4:
        return FiniteDiff(FiniteDiff(u,dx,4), dx, d-4)
    
def FiniteDiff_torch(u, dx, d, device='cpu'):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=4)
    Works but with poor accuracy for d > 4
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.shape[0]
    ux = torch.zeros(n, dtype=torch.float64, device=device)
    
    if d == 1:
        ux[1:n-1] = (u[2:n]-u[0:n-2]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        ux[1:n-1] = (u[2:n]-2*u[1:n-1]+u[0:n-2]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        ux[2:n-2] = (u[4:n]/2-u[3:n-1]+u[1:n-3]-u[0:n-4]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d == 4:
        ux[2:n-2] = (u[4:n]-4*u[3:n-1]+6*u[2:n-2]-4*u[1:n-3]+u[0:n-4]) / dx**4
        
        ux[0] = (3*u[0]-14*u[1]+26*u[2]-24*u[3]+11*u[4]-2*u[5]) / dx**4
        ux[1] = (3*u[1]-14*u[2]+26*u[3]-24*u[4]+11*u[5]-2*u[6]) / dx**4
        ux[n-1] = (3*u[n-1]-14*u[n-2]+26*u[n-3]-24*u[n-4]+11*u[n-5]-2*u[n-6]) / dx**4
        ux[n-2] = (3*u[n-2]-14*u[n-3]+26*u[n-4]-24*u[n-5]+11*u[n-6]-2*u[n-7]) / dx**4
        return ux
    
    if d > 4:
        return FiniteDiff_torch(FiniteDiff_torch(u,dx,4,device=device), dx, d-4,device=device)


def savemodel(obj, filename):
    """
    Saves the Bayes Gibbs sampler object:
    E.g. savemodel(model, "discovery_ebeam.obj")

    Parameters
    ----------
    obj : python object, the bayes object to be saved.
    filename : string, name of the files.
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file) # prints "Serialised"
        file.close()
    return None

def loadmodel(filename):
    """
    Loads a saved Bayes Gibbs type object:
    E.g. saved = loadmodel("discovery_ebeam.obj")

    Parameters
    ----------
    filename : string, the file to load.

    Returns
    -------
    saved_model : pickle dictionary, saved attributes of Gibbs sampler.
    """
    with open(filename, "rb") as file:
        saved_model = pickle.load(file)
        file.close()
        
    saved_model.keys = dir(saved_model) 
    saved_model.modeldict = saved_model.__dict__
    return saved_model
