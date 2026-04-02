#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main program to validate the codes the presented example is a free vibration of a cantilever
the exact solution is in first order mode spatial discretization by finite element method
time integration by Newmark
"""

import numpy as np
import matplotlib.pyplot as plt
import beam_solver
from timeit import default_timer

# Steel properties:
rho = 7800
E = 2e11
b, d = 0.0254, 0.002
A = b*d 
I = b*d**3/12 
L = 1
c1, c2 = 0, 0
Ne = 128
xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)

Ma, Ka, omegaL, lambdaL = beam_solver.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')
Ca = (c1*Ma + c2*Ka)
# F = lambda t : 0
F = lambda t : 2*np.sin(2*np.pi*t*(np.sin(8*np.pi*t))) 

# % ------------------------------------------------
mode = 0
Lambda = np.array([1.875104069, 4.694091133, 7.854757438, 10.99554073,
                   14.13716839, 17.27875953])/L

omega = Lambda[mode]**2*np.sqrt(E*I/rho/A)
h1 = np.cosh(Lambda[mode]*xx) -np.cos(Lambda[mode]*xx) -(np.cos(Lambda[mode]*L)+np.cosh(Lambda[mode]*L)) \
    /(np.sin(Lambda[mode]*L)+np.sinh(Lambda[mode]*L))*(np.sinh(Lambda[mode]*xx)-np.sin(Lambda[mode]*xx))
h2 = Lambda[mode]*(np.sinh(Lambda[mode]*xx)+np.sin(Lambda[mode]*xx))-(np.cos(Lambda[mode]*L)+np.cosh(Lambda[mode]*L)) \
    /(np.sin(Lambda[mode]*L)+np.sinh(Lambda[mode]*L))*(np.cosh(Lambda[mode]*xx)-np.cos(Lambda[mode]*xx))*Lambda[mode]

D0 = np.zeros(2*Ne)
D0[0::2] = h1
D0[1::2] = h2
V0 = np.zeros(2*Ne)
T, dt = 5, 1e-2
t_eval = np.arange(0, T, dt)

t0 = default_timer()
xt = beam_solver.Newmark(Ma,Ca,Ka,F,D0,V0,dt,T)
t1 = default_timer()
print('Mean time: {}'.format((t1-t0)))

# %% -----------------------------------------------------
""" exact solutions """
ExactSol = lambda x,t : (np.cosh(Lambda[mode]*x) - np.cos(Lambda[mode]*x) 
                 -(np.cos(Lambda[mode]*L)+np.cosh(Lambda[mode]*L))/(np.sin(Lambda[mode]*L)+np.sinh(Lambda[mode]*L)) \
                     *(np.sinh(Lambda[mode]*x)-np.sin(Lambda[mode]*x)))*np.cos(omega*t)

dn = xt[0, Ne-2, :]     # displacement of mid-point by Newmark
de = np.array([ExactSol(L/2, t_eval[i]) for i in range(len(t_eval))])

# %% -------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

fig1 = plt.figure(figsize=(10,6))
plt.plot(t_eval, de, 'r', label='Exact')
plt.plot(t_eval, dn, '--b', linewidth=2, label='Newmark')
plt.title('Response of Mid-Point')
plt.xlabel('T (sec)')
plt.ylabel('Displacement')
plt.legend()
plt.margins(0)

# %% -------------------------------------------------
Dis = xt[0, 0::2]
Vel = xt[1, 0::2]
Acc = xt[2, 0::2]

# %%
fig2 = plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.imshow(Dis.T, cmap='jet', origin='lower', aspect='auto'); 
plt.ylabel('$u(t)$'); plt.colorbar(pad=0.01)
plt.subplot(3,1,2)
plt.imshow(Vel.T, cmap='jet', origin='lower', aspect='auto');
plt.ylabel('$\dot{u}(t)$'); plt.colorbar(pad=0.01)
plt.subplot(3,1,3)
plt.imshow(Acc.T, cmap='jet', origin='lower', aspect='auto');
plt.ylabel('$\ddot{u}(t)$'); plt.xlabel('Time (s)'); plt.colorbar(pad=0.01)
plt.show()
