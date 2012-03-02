# classes.py
"""Provides classes for pyBA

"""
# Author:            Berian James (UCB)
# Written:            1-Oct-2011
# Modified:          27-Oct-2011  added det, chol and trace to __init__ in Bivarg
#                    02-Mar-2012  created Bgmap class

import numpy as np
from numpy.linalg import solve, det, cholesky, eig

class Bivarg:
    """ Implements bivariate gaussian structure and routines to modify it.
    """

    __author__ = "Berian James"
    __version__ = "0.1"
    __email__ = "berian@berkeley.edu"

    def __init__(self,mu=np.array([0.,0.]),sigma=np.array([ [1.,0.],[0.,1.] ]),theta=0):
        self.mu = np.squeeze(np.array(mu))
        
        self.E,self.V = eig(np.array(sigma))
        self.E = np.diag(np.real(self.E))
        U = np.array([ [np.cos(theta),-np.sin(theta)],
                       [np.sin(theta), np.cos(theta)] ])
        self.V = np.dot(U,self.V)
        self.sigma = np.dot( self.V, np.dot(self.E,self.V.T) )

        self.det = det(self.sigma)
        self.chol = cholesky(self.sigma)
        self.trace = np.trace(self.sigma)
        self.theta = np.math.degrees(np.math.atan2(self.V[0,1],self.V[0,0]))
        return
        
    def transform(self,
                  dmu=np.array([0,0]),
                  L=np.array([1,1]),
                  theta=0.,
                  d0=np.array([0,0])):
        """ Maps a bivariate gaussian distribution (self) to another
        bivariate gaussian by translation (dmu), scaling of
        the principal axes (L) and rotation (theta) about a given point (d0).
        """

        # Prep inputs
        dmu = np.squeeze(dmu)
        theta = np.squeeze(theta)
        L = np.squeeze(L)
        d0 = np.squeeze(d0)

        # Calculate transformed centre
        U = np.squeeze(np.array([ [np.cos(theta),-np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)] ]))
        mu = np.dot(U,(self.mu * L + dmu) - d0) + d0

        # Calculate transformed covariance
        L = np.diag(L)
        V = np.dot(U,self.V)
        E = np.dot(L,self.E)
        sigma = np.dot( V, np.dot(E,V.T) ) 
    
        # Create new bivarg object with transformed values
        N = Bivarg(mu,sigma,0)
        return N

    def sample(self,n=1):
        """ Draw n samples from bivariate distribution M
        """
    
        vals = np.zeros( (n,2) )
        stds = np.random.randn(2,n)
        for i in range(0,n):
            vals[i] = self.mu + np.dot( self.chol, stds[:,i] ).T
            
        return vals

class Bgmap:
    """ Background mapping structure.
    """

    __author__ = "Berian James"
    __version__ = "0.1"
    __email__ = "berian@berkeley.edu"

    def __init__(self, 
                 dx=np.array([0.,0.]),
                 theta=0.,
                 d0=np.array([0.,0.]),
                 L = np.array([1.,1.]),
                 sigma = np.zeros(7) + np.inf ):

        # Define central likelihood value
        self.mu = np.array([ dx[0], dx[1], theta, d0[0], d0[1], L[0], L[1] ])

        # Define covariance matrix
        if sigma.ndim == 1:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

        # Define parameter handles for convenience
        self.dx = self.mu[0:2]
        self.theta = self.mu[2]
        self.d0 = self.mu[3:5]
        self.L = self.mu[5:7]

        return

    def llik(self,P=np.array( [0., 0., 0., 0., 0., 1., 1.] ) ):
        """ Compute log-likelihood of parameter set P within 
        likelihood distribution bgmap object.
        """

        delta = self.mu - P
        sigma = self.sigma.copy()

        # Interpret infs in covariance matrix as contributing
        #  zero to the chi^2 (set delta[i] = 0).
        I = np.nonzero(np.isinf(np.diag(sigma)))[0]
        delta[I] = 0
        sigma[I,:] = 0
        sigma[:,I] = 0
        sigma[I,I] = 1

        return -0.5 * delta.dot( solve( sigma, delta ) )
