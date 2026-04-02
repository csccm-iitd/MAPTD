# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 23:21:36 2021

@author: Tapas Tripura
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg as LA
from torch.distributions.multivariate_normal import MultivariateNormal as mvrnd
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli as bern
from torch.distributions.gamma import Gamma as gamma

import timeit
import utils

"""
Sparse-Least square regression
"""
def sparsifyDynamics(library,target,lam,iteration=10,ode=False):
    if ode == False:
        if target.ndim > 1:
            target = target.flatten()[None, :]
    Xi = torch.matmul(torch.linalg.pinv(library), target.mT) # initial guess: Least-squares
    for k in range(iteration):
        smallinds = torch.where(torch.abs(Xi) < lam)   # find small coefficients
        Xi[smallinds] = 0
        for ind in range(Xi.shape[1]):
            biginds = torch.where(torch.abs(Xi[:,ind]) > lam)
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds[0], ind] = torch.matmul(torch.linalg.pinv(library[:, biginds[0]]), target[ind, :]) 
    return Xi


"""
The Dictionary creation part:
"""
def library(xt, force=None, polyn=3, harmonic=False, modulus=False, device='cpu'):
    """
    Creates the library from the system states

    Parameters
    ----------
    xt : 2D tensor, states x history,
        State matrix.
    force : 1D tensor, optional
        Deterministic Force vector. The default is None.
    polyn : scalar, optional
        Degree of poynomials to consider. The default is 3.
    harmonic : boolean, optional
        Indicator for including harmonic basses. The default is False.
    modulus : boolean, optional
        Indicator for including modulus basses. The default is False.

    Returns
    -------
    D : matrix or 2D tensor,
        Candidate library.
    nb : scalar,
        number of library functions.

    """
    if polyn > 6:
        raise Exception("Right now the library supports up to 6 polynomial degrees") 
    if polyn == 0:
        polyn = 1
    
    # poly order 0
    ns, nt = [*xt.shape]
    D = torch.ones(nt,1, device=device, dtype=torch.float64)
    
    # poly order 1
    if polyn >= 1:
        for i in range(ns):
            D = torch.cat((D, xt[i,:][:,None]), axis=1)
    # ploy order 2
    if polyn >= 2: 
        for i in range(ns):
            D = torch.cat((D, (xt[i,:]**2)[:,None]), axis=1) 
    # ploy order 3
    if polyn >= 3:    
        for i in range(ns):
            D = torch.cat((D, (xt[i,:]**3)[:,None]), axis=1) 
    # ploy order 4
    if polyn >= 4:
        for i in range(ns):
            D = torch.cat((D, (xt[i,:]**4)[:,None]), axis=1) 
    # ploy order 5
    if polyn >= 5:
        for i in range(ns):
            D = torch.cat((D, (xt[i,:]**5)[:,None]), axis=1) 
    # ploy order 6
    if polyn >= 6:
        for i in range(ns):
            D = torch.cat((D, (xt[i,:]**6)[:,None]), axis=1) 
    if modulus:
        # for the signum or sign operator
        for i in range(ns):
            D = torch.cat((D, torch.sign(xt[i,:])[:,None]), axis=1) 
        # for the modulus operator
        for i in range(ns):
            D = torch.cat((D, torch.abs(xt[i,:])[:,None]), axis=1) 
    if harmonic:
        # for sin(x)
        for i in range(ns):
            D = torch.cat((D, torch.sin(xt[i,:])[:,None]), axis=1) 
        # for cos(x)
        for i in range(ns):
            D = torch.cat((D, torch.cos(xt[i,:])[:,None]), axis=1) 
    # The force u
    for i in range(force.shape[0]):
        D = torch.cat((D, force[i,:][:, None]), axis=1) 
    # Total number of library 
    nb = D.shape[1]  
    
    return D, nb

def library_pde(xt, dx, force=None, polyn=1, order=6, harmonic=False, modulus=False, device='cpu'):
    """
    Creates the library from the system states

    Parameters
    ----------
    xt : 2D tensor, states x history,
        State matrix.
    force : 1D tensor, optional
        Deterministic Force vector. The default is None.
    polyn : scalar, optional
        Degree of poynomials to consider. The default is 3.
    harmonic : boolean, optional
        Indicator for including harmonic basses. The default is False.
    modulus : boolean, optional
        Indicator for including modulus basses. The default is False.

    Returns
    -------
    D : matrix or 2D tensor,
        Candidate library.
    nb : scalar,
        number of library functions.

    """
    if polyn > 6:
        print("Polynomial degrees beyond 6 will make the library very big") 
    if polyn == 0:
        polyn = 1
    
    # poly order 0
    _, nx, nt = [*xt.shape]
    D = torch.ones(nx*nt,1, device=device, dtype=torch.float64)
    # Time derivative of u
    D = torch.cat((D, xt[1, ...].flatten()[:,None]), axis=1)
    # Poly order of u
    for i in range(polyn):
        D = torch.cat((D, (xt[0, ...]**(i+1)).flatten()[:,None]), axis=1)

    # Compute the derivatives:
    dudx = torch.zeros(order, nx, nt, device=device, dtype=torch.float64)
    for i in range(order):
        for j in range(nt):
            dudx[i, :, j] = utils.FiniteDiff_torch(xt[0,:,j], dx, d=(i+1), device=device)

    # Space derivative of u
    for i in range(order):
        for j in range(polyn):
            D = torch.cat((D, (dudx[i, ...]**(j+1)).flatten()[:,None]), axis=1) 
    # u times space derivate of u:
    for i in range(order):
        for j in range(polyn):
            D = torch.cat((D, (xt[0,:] * dudx[i, ...]**(j+1)).flatten()[:,None]), axis=1) 

    if modulus:
        # for the signum or sign operator
        D = torch.cat((D, torch.sign(xt[0,:]).flatten()[:,None]), axis=1) 
        # for the modulus operator
        D = torch.cat((D, torch.abs(xt[0,:]).flatten()[:,None]), axis=1) 
        # for the tensor operator
        D = torch.cat((D, torch.mul(xt[0,:], torch.abs(xt[0,:])).flatten()[:,None]), axis=1) 
    if harmonic:
        # for sin(x)
        D = torch.cat((D, torch.sin(xt[0,:]).flatten()[:,None]), axis=1) 
        # for cos(x)
        D = torch.cat((D, torch.cos(xt[0,:]).flatten()[:,None]), axis=1) 
    # The force u
    if force != None:
        D = torch.cat((D, force.flatten()[:, None]), axis=1) 
    # Total number of library 
    nb = D.shape[1]  

    return D, nb


"""
# Gibbs sampling:
"""
class Gibbs(object):
    def __init__(self, ns, nl, nt, ap=0.1, bp=1, av=0.5, bv=0.5, asig=1e-4, bsig=1e-4,
                 p0_init=0.1, vs_init=10, iterations=1000, device='cpu'):
        
        # Hyper-parameters
        self.N = nt
        self.ns = ns
        self.nl = nl
        self.ap = torch.tensor(ap, dtype=torch.float64) 
        self.bp = torch.tensor(bp, dtype=torch.float64) 
        self.av = torch.tensor(av, dtype=torch.float64)
        self.bv = torch.tensor(bv, dtype=torch.float64) 
        self.asig = torch.tensor(asig, dtype=torch.float64)  
        self.bsig = torch.tensor(bsig, dtype=torch.float64)  
        self.MCMC = iterations
        self.burn_in = int(iterations/2) 
        self.device = device
        
        # Parameter Initialisation:
        self.p0 = torch.zeros(self.MCMC, dtype=torch.float64)
        self.vs = torch.zeros(self.MCMC, dtype=torch.float64)
        self.sig = torch.zeros(self.MCMC, dtype=torch.float64)
        self.zstore = torch.zeros(self.nl, self.MCMC-self.burn_in, dtype=torch.float64)
        self.theta = torch.zeros(self.nl, self.MCMC-self.burn_in, dtype=torch.float64)
        self.p0[0] = p0_init
        self.vs[0] = vs_init
    
    def res_var(self, L, y):
        # Residual variance:
        beta = torch.matmul(torch.linalg.pinv(L), y)
        error = y - torch.matmul(L, beta)
        return torch.var(error)
    
    def Bsig_and_Bmu(self, z, L, vs, y, prior="independent"):
        # Theta: Multivariate Normal distribution
        index = torch.where(z != 0)[0]
        Dr = L[:,index] 
        if prior == "independent":
            Aor = torch.eye(len(index), dtype=torch.float64, device=self.device) # independent prior
        else:
            Aor = torch.mul(len(Dr), torch.linalg.inv(torch.matmul(Dr.T, Dr))) # g-prior
            
        SIG = torch.linalg.inv(torch.matmul(Dr.T, Dr) + torch.mul(vs**-1, torch.linalg.inv(Aor)))
        MU = torch.matmul(torch.matmul(SIG, Dr.T), y)
        return MU, SIG, Aor, index

    def pyzv(self, D, ztemp, vs, y, prior="independent"):
        # P(Y|zi=(0|1),z-i,vs)
        rind = torch.where(ztemp != 0)[0]
        Sz = sum(ztemp)
        Dr = D[:, rind] 
        if prior == "independent":
            Aor = torch.eye(len(rind), dtype=torch.float64, device=self.device) # independent prior
        else:
            Aor = torch.mul(self.N, torch.linalg.inv(torch.matmul(Dr.T, Dr))) # g-prior
            
        BSIG = torch.matmul(Dr.T, Dr) + torch.mul(vs**-1, torch.linalg.inv(Aor))
        
        (sign, logdet0) = torch.linalg.slogdet(torch.linalg.inv(Aor))
        (sign, logdet1) = torch.linalg.slogdet(torch.linalg.inv(BSIG))
        
        PZ = torch.lgamma(self.asig + 0.5*self.N) -0.5*self.N*torch.log(2*torch.tensor(torch.pi)) \
            - 0.5*Sz*torch.log(vs) + self.asig*torch.log(self.bsig) - torch.lgamma(self.asig) \
                + 0.5*logdet0 + 0.5*logdet1
        denom_sub = torch.eye(self.N, device=self.device) - torch.matmul(torch.matmul(Dr, torch.linalg.inv(BSIG)), Dr.T)
        denom = (0.5*torch.matmul(torch.matmul(y.T, denom_sub), y))
        PZ = PZ - (self.asig+0.5*self.N) * (torch.log(self.bsig + denom))
        return PZ

    def pyzv0(self, y):
        # P(Y|zi=0,z-i,vs)
        PZ0 = torch.lgamma(self.asig + 0.5*self.N) - 0.5*self.N*torch.log(2*torch.tensor(torch.pi)) \
            + self.asig*torch.log(self.bsig) - torch.lgamma(self.asig) + torch.log(torch.tensor(1)) \
                - (self.asig+0.5*self.N)*torch.log(self.bsig + 0.5*torch.matmul(y.T, y))
        return PZ0
    
    def slab(self, theta, Aor, Sz, sigma):
        # sample 'vs' from inverse Gamma:
        avvs = self.av + 0.5*Sz
        bvvs = self.bv + (torch.matmul(torch.matmul(theta.T, torch.linalg.inv(Aor)), theta))/(2*sigma)
        distribution = gamma(avvs, bvvs)
        return 1 / distribution.sample() # inverse gamma RVs
    
    def binary(self, Sz):
        # sample 'p0' from Beta distribution:
        app0 = self.ap+Sz
        bpp0 = self.bp+self.nl-Sz # Here, P=nl (no. of functions in library)
        distribution = Beta(app0, bpp0)
        return distribution.sample()
    
    def noise_sig(self, MU, BSIG, y):
        # sample 'sig^2' from inverse Gamma:
        asiggamma = self.asig+0.5*self.N
        temp = torch.matmul(torch.matmul(MU.T, torch.linalg.inv(BSIG)), MU)
        bsiggamma = self.bsig+0.5*(torch.matmul(y.T, y) - temp)
        distribution = gamma(asiggamma, bsiggamma) # inverse gamma RVs
        return 1 / distribution.sample()
    
    def weights(self, MU, BSIG, sigma, index):
        # Sample theta from Normal distribution:
        beta = torch.zeros(self.nl, dtype=torch.float64, device=self.device)
        distribution = mvrnd(MU, torch.mul(sigma, BSIG))
        thetar = distribution.sample()
        beta[index] = thetar
        return beta, thetar
    
    def latent_variable(self, D, y, zz, vs, p0):
        # sample z from the Bernoulli distribution:
        for jj in range(self.nl):
            ztemp_0 = zz
            ztemp_0[jj] = 0
            if torch.mean(ztemp_0) == 0:
                PZ0 = self.pyzv0(y)
            else:
                PZ0 = self.pyzv(D, ztemp_0, vs, y)
            
            ztemp_1 = zz
            ztemp_1[jj] = 1      
            PZ1 = self.pyzv(D, ztemp_1, vs, y)
            
            zeta = PZ0 - PZ1  
            zeta = p0/( p0 + torch.exp(zeta)*(1-p0))
            distribution = bern(probs = zeta)
            zz[jj] = distribution.sample()
        return zz
    
    def forward(self, D, yy, verbose=False, verbose_interval=50):
        if yy.ndim > 1:
            yy = yy.flatten() 
            
        self.sig[0] = self.res_var(D, yy)
        zint  = latent(self.ns, self.nl, D, yy)
        zval = zint.clone()
        
        Bmu, BSIG, Aor, index = self.Bsig_and_Bmu(zval, D, self.vs[0], yy)
        _, thetar = self.weights(Bmu, BSIG, self.sig[0], index)
        
        time = 0
        for epoch in range(1, self.MCMC):
            t1 = timeit.default_timer() 
            
            # sample z from the Bernoulli distribution:
            zr = zval.clone()
            zr = self.latent_variable(D, yy, zr, self.vs[epoch-1], self.p0[epoch-1])
            zval = zr
            
            if epoch > self.burn_in-1:
                self.zstore[:, epoch-self.burn_in] = zval 
            
            # sample sig^2 from inverse Gamma:
            self.sig[epoch] = self.noise_sig(Bmu, BSIG, yy) # inverse gamma RVs
            
            # sample vs from inverse Gamma:
            Sz = sum(zval)
            self.vs[epoch] = self.slab(thetar, Aor, Sz, self.sig[epoch])
            
            # sample p0 from Beta distribution:
            self.p0[epoch] = self.binary(Sz)
            
            # Sample theta from Normal distribution:
            Bmu, BSIG, Aor, index = self.Bsig_and_Bmu(zval, D, self.vs[epoch], yy)
            weights, thetar = self.weights(Bmu, BSIG, self.sig[epoch], index)
            if epoch > self.burn_in-1:
                self.theta[:, epoch-self.burn_in] = weights
                
            t2 = timeit.default_timer() 
            time += (t2-t1)
            
            if verbose:
                if epoch % verbose_interval == 0:
                    print('Iteration-{}, Time-{:0.4f}'.format(epoch,time))
                    time = 0
        
        # Post processing:
        mean_theta = torch.mean(self.theta, axis=1)
        mean_latent = torch.mean(self.zstore, axis=1)
        
        sig_theta = torch.cov(self.theta, correction=1) 
        return mean_theta, mean_latent, sig_theta 
    
    
"""
# Bayesian Interference:
"""
def BayInt(D, xdt):
    # for the dictionary:
    muD = np.mean(D,0)
    sdvD1 = np.std(D,0)
    sdvD = np.diag(sdvD1)
    Ds = np.dot((D - np.ones([len(D),1])*muD), LA.inv(sdvD))
    
    # for the observed data:
    muxdt = np.mean(xdt)
    xdts = np.vstack(xdt) - np.ones([len(D),1])*muxdt
    xdts = np.reshape(xdts, -1)
    
    return Ds, xdts, sdvD


"""
Initial latent vector finder:
"""
def latent(ns, nl, D, yy):
    """
    Initializes the latent variable identification vector

    Parameters
    ----------
    ns : scalar, number of states
    nl : scalar, number of basis function.
    D : matrix or 2D tensor,
        Library of candidate basis functions.
    yy : vector or 1D tensor,
        Target label vector.

    Returns
    -------
    zint : 1D tensor,
        The initial latent variable vector.

    """
    ### Forward finder:
    zint = torch.zeros(nl, dtype=torch.float64)
    theta = torch.matmul(torch.linalg.pinv(D), yy)
    index = torch.where(zint != 0)[0]
    Dr = D[:, index]
    thetar = theta[index]
    err = F.mse_loss(yy, torch.matmul(Dr, thetar))[None]
    
    for i in range(nl):
        Dr = D[:, i]
        thetar = theta[i]
        err = torch.cat(( err, F.mse_loss(yy, torch.mul(Dr, thetar))[None] ))
        if err[i+1] <= err[i]:
            zint[i] = 1
        else:
            zint[i] = 0
    
    ### Backward finder:
    index = torch.where(zint != 0)[0]
    Dr = D[:, index]
    thetar = theta[index]
    err = F.mse_loss(yy, torch.matmul(Dr, thetar))[None]
    ind = 0
    for i in range(nl-1, -1, -1):
        index = ind
        Dr = D[:, index]
        thetar = theta[index]
        err = torch.cat(( err, F.mse_loss(yy, torch.mul(Dr, thetar))[None] ))
        if err[ind+1] <= err[ind]:
            zint[index] = 1
        else:
            zint[index] = 0
        ind = ind + 1
    
    # states are kept active all the time
    # zint[1:ns] = 1
    return zint
