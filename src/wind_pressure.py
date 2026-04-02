#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trial script for generating wind pressure for the 76 DOF example
"""

from functools import partial
import numpy as np
import matplotlib.pyplot as plt

fun = lambda rho, L, cd, u_ref, z_ref, z: 0.5*rho*L*cd*u_ref**2*(z/z_ref)**0.32

L = 42
z_ref = 0.5
u_ref = 1
rho = 1.22
cd = 1.2
z = np.concatenate((
                    np.linspace(0,10,2),
                    10 + np.linspace(4.5,9,2),
                    10 + 9 + np.linspace(3.9,136.5,35),
                    10 + 9 + 136.5 + np.linspace(4.5,9,2),
                    10 + 9 + 136.5 + 9 + np.linspace(3.9,132.6,34),
                    10 + 9 + 136.5 + 9 + 132.6 + np.linspace(4.5,9,2)
                   ))

intensity = partial(fun, rho=rho, L=L, cd=cd, z_ref=z_ref, z=z)

t = np.arange(0,5,0.01)
x = np.sin(np.pi*t)

force = intensity(u_ref=0.15*np.sin(np.pi*0.5))

# %%
fig1 = plt.figure(figsize=(6,10), dpi=100)
plt.plot(force, np.arange(len(z)), '-o')
plt.xlabel('Wind Intensity $f(z)$')
plt.ylabel('Height of Building ($z$)')
plt.yticks(np.arange(0,len(z),4))
plt.xticks(np.arange(0,np.ceil(np.max(force)+1),1))
plt.grid(True, alpha=0.4)
plt.show()
