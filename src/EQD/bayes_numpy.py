"""
Created on Thu Sep  9 23:21:36 2021

@author: Tapas Tripura
"""

import numpy as np
from scipy.special import loggamma as LG
from scipy.stats import invgamma
from numpy import linalg as LA
from numpy.random import gamma as IG
from numpy.random import beta
from numpy.random import binomial as bern
from numpy.random import multivariate_normal as mvrv
import utils

def MSE(x,y):
    return np.mean((x-y)**2)

"""
Sparse-Least square regression
"""
def sparsifyDynamics(library,target,lam,iteration=10):
    if target.ndim > 1:
        target = target.flatten()[None, :]
    Xi = np.matmul(np.linalg.pinv(library), target.T) # initial guess: Least-squares
    for k in range(iteration):
        smallinds = np.where(abs(Xi) < lam)   # find small coefficients
        Xi[smallinds] = 0
        for ind in range(Xi.shape[1]):
            biginds = np.where(abs(Xi[:,ind]) > lam)
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds[0], ind] = np.matmul(np.linalg.pinv(library[:, biginds[0]]), target[ind, :].T) 
    return Xi

"""
The Dictionary creation part:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def library(xt, force=None, polyn=3, harmonic=False, modulus=False):
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
    
    ns, nt = [*xt.shape]
    # poly order 0
    D = np.ones([nt,1])
    # poly order 1
    if polyn >= 1:
        for i in range(ns):
            D = np.append(D, xt[i,:][:,None], axis=1)
    ns = ns // 2 # Only take displacement states for nonlinearity 
    # ploy order 2     
    if polyn >= 2: 
        for i in range(ns):
            D = np.append(D, (xt[i,:]**2)[:,None], axis=1)
    # ploy order 3
    if polyn >= 3:    
        for i in range(ns):
            D = np.append(D, (xt[i,:]**3)[:,None], axis=1) 
    # ploy order 4
    if polyn >= 4:
        for i in range(ns):
            D = np.append(D, (xt[i,:]**4)[:,None], axis=1)
    # ploy order 5
    if polyn >= 5:
        for i in range(ns):
            D = np.append(D, (xt[i,:]**5)[:,None], axis=1) 
    # ploy order 6
    if polyn >= 6:
        for i in range(ns):
            D = np.append(D, (xt[i,:]**6)[:,None], axis=1) 
    if modulus:
        # for the signum or sign operator
        for i in range(ns):
            D = np.append(D, np.sign(xt[i,:])[:,None], axis=1)
        # for the modulus operator
        for i in range(ns):
            D = np.append(D, np.abs(xt[i,:])[:,None], axis=1)
    if harmonic:
        # for sin(x)
        for i in range(ns):
            D = np.append(D, np.sin(xt[i,:])[:,None], axis=1)
        # for cos(x)
        for i in range(ns):
            D = np.append(D, np.cos(xt[i,:])[:,None], axis=1)
    # The force u
    for i in range(force.shape[0]):
        D = np.append(D, force[i,:][:, None], axis=1) 
    # Total number of library 
    nb = D.shape[1]  
    
    return D, nb

"""
The Dictionary creation part:
"""
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
    D = np.ones([nx*nt, 1])
    # Time derivative of u
    D = np.append(D, xt[1,:].flatten()[:,None], axis=1) 
    # Poly order of u
    for i in range(polyn):
        D = np.append(D, (xt[0,:]**(i+1)).flatten()[:,None], axis=1)
    # Compute the derivatives:
    dudx = np.zeros((order, nx, nt))
    for i in range(order):
        for j in range(nt):
            dudx[i, :, j] = utils.FiniteDiff(xt[0,:,j], dx, d=(i+1))
    # Space derivative of u
    for i in range(order):
        for j in range(polyn):
            D = np.append(D, (dudx[i, ...]**(j+1)).flatten()[:,None], axis=1) 
    # u times space derivate of u:
    for i in range(order):
        for j in range(polyn):
            D = np.append(D, (xt[0,:] * dudx[i, ...]**(j+1)).flatten()[:,None], axis=1) 
    
    if modulus:
        # for the signum or sign operator
        D = np.append(D, np.sign(xt[0,:]).flatten()[:,None], axis=1) 
        # for the modulus operator
        D = np.append(D, np.abs(xt[0,:]).flatten()[:,None], axis=1)
        # for the tensor operator
        D = np.append(D, np.multiply(xt[0,:], abs(xt[0,:])).flatten()[:,None], axis=1) 
    if harmonic:
        # for sin(x)
        D = np.append(D, np.sin(xt[0,:]).flatten()[:,None], axis=1)  
        # for cos(x)
        D = np.append(D, np.cos(xt[0,:]).flatten()[:,None], axis=1) 
    # The force u
    if force.any() != None:
        D = np.append(D, force[:,None], axis=1)  
    # Total number of library 
    nb = D.shape[1]  

    return D, nb

"""
# Gibbs sampling:
"""
class Gibbs(object):
    def __init__(self, ns, nl, nt, ap=0.1, bp=1, av=0.5, bv=0.5, asig=1e-2, bsig=1e-2,
                 p0_init=0.5, vs_init=10, iterations=1000, burn_in=None):
        
        # Hyper-parameters
        self.N = nt
        self.ns = ns
        self.nl = nl
        self.ap = ap 
        self.bp = bp 
        self.av = av
        self.bv = bv 
        self.asig = asig
        self.bsig = bsig  
        self.MCMC = iterations
        self.burn_in = burn_in if burn_in != None else int(iterations/2) 
        
        # Parameter Initialisation:
        self.p0 = np.zeros(self.MCMC)
        self.vs = np.zeros(self.MCMC)
        self.sig = np.zeros(self.MCMC)
        self.zstore = np.zeros((self.nl, self.MCMC-self.burn_in))
        self.theta = np.zeros((self.nl, self.MCMC-self.burn_in))
        self.p0[0] = p0_init
        self.vs[0] = vs_init
        
    def res_var(self, L, y):
        # Residual variance:
        beta = np.dot(LA.pinv(L), y)
        error = y - np.matmul(L, beta)
        return np.var(error)
    
    def Bsig_and_Bmu(self, z, L, vs, y, prior="independent", eps=1e-6):
        # Theta: Multivariate Normal distribution
        index = np.where(z != 0)[0]
        Dr = L[:,index] 
        if prior == "independent":
            Aor = np.eye(len(index)) # independent prior
        else:
            Aor = np.dot(len(Dr), LA.inv(np.matmul(Dr.T, Dr))) # g-prior
            
        SIG = LA.inv(np.matmul(Dr.T,Dr) + np.dot(pow(vs,-1), LA.inv(Aor))) + eps
        MU = np.matmul(np.matmul(SIG,Dr.T),y)
        return MU, SIG, Aor, index

    def pyzv(self, D, ztemp, vs, y, prior="independent"):
        # P(Y|zi=(0|1),z-i,vs)
        rind = np.where(ztemp != 0)[0]
        Sz = sum(ztemp)
        Dr = D[:, rind] 
        if prior == "independent":
            Aor = np.eye(len(rind)) # independent prior
        else:
            Aor = np.dot(self.N, LA.inv(np.matmul(Dr.T, Dr))) # g-prior
            
        BSIG = np.matmul(Dr.T, Dr) + np.dot(pow(vs, -1),LA.inv(Aor)) 
    
        (sign, logdet0) = LA.slogdet(LA.inv(Aor))
        (sign, logdet1) = LA.slogdet(LA.inv(BSIG))
        
        PZ = LG(self.asig + 0.5*self.N) -0.5*self.N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
                + self.asig*np.log(self.bsig) - LG(self.asig) + 0.5*logdet0 + 0.5*logdet1
        denom1 = np.eye(self.N) - np.matmul(np.matmul(Dr, LA.inv(BSIG)), Dr.T)
        denom = (0.5*np.matmul(np.matmul(y.T, denom1), y))  
        PZ = PZ - (self.asig+0.5*self.N)*(np.log(self.bsig + denom))
        return PZ

    def pyzv0(self, y):
        # P(Y|zi=0,z-i,vs)
        PZ0 = LG(self.asig + 0.5*self.N) - 0.5*self.N*np.log(2*np.pi) \
            + self.asig*np.log(self.bsig) - LG(self.asig) \
                + np.log(1) - (self.asig+0.5*self.N)*np.log(self.bsig + 0.5*np.matmul(y.T, y))
        return PZ0
    
    def slab(self, theta, Aor, Sz, sigma):
        # sample 'vs' from inverse Gamma:
        avvs = self.av + 0.5*Sz
        bvvs = self.bv + (np.matmul(np.matmul(theta.T, LA.inv(Aor)), theta))/(2*sigma)
        # return 1/IG(avvs, 1/bvvs) # inverse gamma RVs
        return invgamma.rvs(a=avvs, scale=bvvs, size=1)
    
    def binary(self, Sz):
        # sample 'p0' from Beta distribution:
        app0 = self.ap+Sz
        bpp0 = self.bp+self.nl-Sz # Here, P=nl (no. of functions in library)
        return beta(app0, bpp0)
    
    def noise_sig(self, MU, BSIG, y):
        # sample 'sig^2' from inverse Gamma:
        asiggamma = self.asig+0.5*self.N
        temp = np.matmul(np.matmul(MU.T, LA.inv(BSIG)), MU)
        bsiggamma = self.bsig+0.5*np.abs(np.dot(y.T, y) - temp) 
        # return 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
        return invgamma.rvs(a=asiggamma, scale=bsiggamma, size=1)
    
    def weights(self, MU, BSIG, sigma, index, eps=1e-6):
        # Sample theta from Normal distribution:
        beta = np.zeros(self.nl)
        # print(np.dot(sigma, BSIG))
        thetar = mvrv(MU, np.dot(sigma, BSIG) + (eps * np.eye(MU.shape[0])))
        beta[index] = thetar
        return beta, thetar
    
    def latent_variable(self, D, y, zz, vs, p0):
        # sample z from the Bernoulli distribution:
        for jj in range(self.nl):
            ztemp_0 = zz
            ztemp_0[jj] = 0
            if np.mean(ztemp_0) == 0:
                PZ0 = self.pyzv0(y)
            else:
                PZ0 = self.pyzv(D, ztemp_0, vs, y)
            
            ztemp_1 = zz
            ztemp_1[jj] = 1      
            PZ1 = self.pyzv(D, ztemp_1, vs, y)
            
            zeta = PZ0 - PZ1
            zeta = p0/( p0 + np.exp(zeta)*(1-p0))
            zz[jj] = bern(1, p = np.clip(zeta, a_min=0, a_max=1), size = None)
        return zz
    
    def forward(self, D, yy, theta0=None, zmean0=None, verbose=False, verbose_interval=50):
        self.sig[0] = self.res_var(D, yy)
        if zmean0 is None:
            zint  = latent(self.ns, self.nl, D, yy)
        else:
            zint = zmean0
        zval = zint.copy()
        
        Bmu, BSIG, Aor, index = self.Bsig_and_Bmu(zval, D, self.vs[0], yy)
        _, thetar = self.weights(Bmu, BSIG, self.sig[0], index)
        
        for epoch in range(1, self.MCMC):
            if verbose:
                if epoch % verbose_interval == 0:
                    print('Iteration-{}'.format(epoch))
            
            # sample z from the Bernoulli distribution:
            zr = zval.copy()
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
        
        # Post processing:
        mean_theta = np.mean(self.theta, axis=1)
        mean_latent = np.mean(self.zstore, axis=1)
        
        sig_theta = np.cov(self.theta, bias=False) 
        return mean_theta, mean_latent, self.theta, sig_theta 
    

"""
# Bayesian Interference:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    
    return Ds, xdts, muD, sdvD


"""
# Initial latent vector finder:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def latent(ns, nl, D, xdts):
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
    # Forward finder:
    zint = np.zeros(nl)
    theta = np.matmul(LA.pinv(D), xdts)
    index = np.where(zint != 0)[0]
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    for i in range(0, nl):
        Dr = D[:, i]
        thetar = theta[i]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[i+1] <= err[i]:
            zint[i] = 1
        else:
            zint[i] = 0
    
    # Backward finder:
    index = np.where(zint != 0)[0]
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    ind = 0
    for i in range(nl-1, -1, -1):
        index = ind
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[ind+1] <= err[ind]:
            zint[index] = 1
        else:
            zint[index] = 0
        ind = ind + 1
    
    # states are kept active all the time
    zint[1:ns] = 1
    return zint
