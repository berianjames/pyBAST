"""Provides classes for pyBA. """

import numpy as np
from numpy.linalg import solve, det, cholesky, eig

class Bgmap:
    """ Background mapping structure.
    """

    __author__ = "Berian James"
    __version__ = "0.2"
    __email__ = "berian@berkeley.edu"

    def __init__(self, 
                 dx=np.array([0.,0.]),
                 theta=0.,
                 d0=np.array([0.,0.]),
                 L = np.array([1.,1.]),
                 mu = np.empty( (7,) ) + np.nan,
                 sigma = np.zeros(7) + np.inf ):

        # Define central likelihood value
        if np.isnan(mu).any():
            self.mu = np.array([ dx[0], dx[1], theta, d0[0], d0[1], L[0], L[1] ])
        else:
            self.mu = mu

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

    def sample(self,n=1):
        """ Returns n samples from a Bgmap distribution.
        """
        # N.B.! Won't work with covariance matrices containings infs!
        
        #vals = np.zeros( (n,len(self.mu)) )
        stds = np.random.randn( 2,len(self.mu) )
        chol = cholesky(self.sigma)
        return [self.mu + np.dot( chol, stds[:,i] ).T for i in range(n)]
            
class Bivarg:
    """ Implements bivariate gaussian structure and routines to modify it.
    """

    __author__ = "Berian James"
    __version__ = "0.2"
    __email__ = "berian@berkeley.edu"

    def __init__(self,mu=np.array([0.,0.]),sigma=np.array([ [1.,0.],[0.,1.] ]),theta=0):
        self.mu = np.squeeze(np.array(mu))

        if sigma.size == 2:
            sigma = np.array( [ [sigma[0],0], [0,sigma[1]] ] )
        elif sigma.size == 3:
            sigma = np.array( [ [sigma[0], sigma[2]], [sigma[2], sigma[1]] ] )

        # Non-optimal code
        #self.sigma = sigma
        #self.det = det(self.sigma)
        #self.trace = np.trace(self.sigma)

        # Get determinant and trace for quick eigenvalue computation
        #self.det = sigma[0,0]*sigma[1,1] - sigma[0,1]*sigma[1,0]
        #self.trace = sigma[0,0] + sigma[1,1]

        # Non-optimal code for eigenvalue computation?
        self.E,self.V = eig(np.array(sigma))
        #epart = np.sqrt( self.trace*self.trace / 4. - self.det )
        #self.E = np.array( [self.trace/2 + epart, self.trace/2 - epart] )
        #if sigma[1,0] != 0:
        #    self.V = np.array([ [self.E[0]-sigma[1,1], self.E[1]-sigma[1,1]], [sigma[1,0], sigma[1,0]] ])
        #elif sigma[0,1] != 0:
        #    self.V = np.array([ [sigma[0,1], sigma[0,1]], [self.E[0]-sigma[0,0], self.E[1]-sigma[0,0]] ])
        #else:
        #    self.V = np.array([ [1.,0.], [0.,1.] ])
        
        self.E = np.diag(np.real(self.E))
        if theta!=0:
            U = np.array([ [np.cos(theta),-np.sin(theta)],
                           [np.sin(theta), np.cos(theta)] ])
            self.V = np.dot(U,self.V)

        self.sigma = np.dot( self.V, np.dot(self.E,self.V.T) )

        sigma = self.sigma
        self.det = sigma[0,0]*sigma[1,1] - sigma[0,1]*sigma[1,0]
        self.trace = sigma[0,0] + sigma[1,1]

        #self.chol = cholesky(self.sigma)
        self.chol = np.array([ [np.sqrt(sigma[0,0]),0.],
                               [sigma[0,1]/np.sqrt(sigma[0,0]), 
                                np.sqrt( sigma[1,1]-sigma[1,0]*sigma[0,1]/sigma[0,0] ) ] ])

        self.theta = np.math.degrees(np.math.atan2(self.V[0,1],self.V[0,0]))
        return

    def __sub__(self,other):
        return Bivarg( mu=self.mu-other.mu, sigma=self.sigma+other.sigma )

    def __add__(self,other):
        return Bivarg( mu=self.mu+other.mu, sigma=self.sigma+other.sigma )

#    def __repr__(self):
#        """ Print beautiful representation of bivarg object.
#        """
#        return
        
    def transform(self,other=Bgmap().mu):
        """ Maps a bivariate gaussian distribution (self) to another
        bivariate gaussian by translation (dmu), scaling of
        the principal axes (L) and rotation (theta) about a given point (d0).
        """

        # Prep inputs
        dmu = other[0:2]
        theta = other[2]
        d0 = other[3:5]
        L = other[5:7]
    

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
        return LittleBivarg(mu,sigma)

    def sample(self,n=1):
        """ Draw n samples from bivariate distribution M
        """
    
        vals = np.zeros( (n,2) )
        stds = np.random.randn(2,n)
        for i in range(0,n):
            vals[i] = self.mu + np.dot( self.chol, stds[:,i] ).T
            
        return vals

class LittleBivarg(Bivarg):
    """ This is a minimal bivarg class that is used for representing
    transformed Bivargs that are going to be thrown away later, but might
    be used for likelihood computations.
    """
    def __init__(self,mu=np.array([0.,0.]),sigma=np.array([ [1.,0.],[0.,1.] ])):
        self.mu = mu
        self.sigma = sigma
        self.det = sigma[0,0]*sigma[1,1] - sigma[1,0]*sigma[0,1]
        
        return

    def promote(self):
        """ Promotes a LitteBivarg to a full Bivarg. This might happen
        when a transformed Bivarg needs to be used for transformation in
        its own right.
        Good job LittleBivarg!
        """
        return Bivarg(mu=self.mu,sigma=self.sigma)

class Amap:
    """ Implements astrometric mapping class as a gaussian process.
    """

    __author__ = "Berian James"
    __version__ = "0.2"
    __email__ = "berian@berkeley.edu"

    def __init__(self,P,A,B,scale=100.0,amp=20.0):
        """ Create instance of astrometric map from a background mapping
        (Bgmap object P) and objects in each frame (Bivarg arrays A and B).
        """
        from pyBA.distortion import astrometry_mean, astrometry_cov, d2

        self.P = P
        self.A = A
        self.B = B

        # Default GP hyperparameters
        self.scale = scale
        self.amp = amp
        self.hyperparams = {'scale': self.scale, 'amp': self.amp}

        # Define covariance matrix for data points
        xyarr = np.array([o.mu for o in self.A])
        self.d2 = d2(xyarr,xyarr)
        self.Cx = astrometry_cov(self.d2, self.scale, self.amp,
                                 nugget = np.array([o.sigma[0,0] for o in A]))
        self.Cy = astrometry_cov(self.d2, self.scale, self.amp,
                                 nugget = np.array([o.sigma[1,1] for o in A]))

        # Don't compute cholesky decomposition of C until needed
        self.chol = False
        self.cholx = None
        self.choly = None

        return 

    def draw_background(self, res=30):
        """ Method to draw maximum likelihood background mapping 
        on grid of given resolution."""

        from pyBA.plotting import draw_MAP_background
        draw_MAP_background(self.A, self.B, self.P, res = res )
        return

    def draw_realisation(self, res=30):
        """ Draw realisation from Dmap gaussian process and
        plot it on grid of given resolution."""

        from pyBA.plotting import draw_realisation

        # If GP is not conditioned (as checked by self.chol not yet computed),
        #  draw realisation without using input data
        if self.chol == False:
            draw_realisation(self.A, self.B, self.P, self.scale, 
                             self.amp, self.chol, self.cholx, self.choly,
                             res=res)

        # Otherwise, perform regression on observed data
        else:
            draw_realisation(self.A, self.B, self.P, self.scale, 
                             self.amp, self.chol, self.cholx, self.choly,
                             res=res)
            
        return

    def draw_residuals(self, res=30, scaled='no'):
        """ Draw residuals of object mappings between frames
        from background mapping."""

        from pyBA.plotting import draw_MAP_residuals
        draw_MAP_residuals(self.A, self.B, self.P, scaled=scaled)
        return

    def build_covariance(self,scale,amp):

        from pyBA.distortion import astrometry_cov

        self.scale = scale
        self.amp = amp

        self.Cx = astrometry_cov(self.d2, self.scale, self.amp,
                                 nugget=self.nuggetx)
        self.Cy = astrometry_cov(self.d2, self.scale, self.amp,
                                 nugget=self.nuggety)

        # Compute cholesky decomposition of C with optimised parameters
        from scipy.linalg import cho_factor
        self.chol = True
        self.cholx = cho_factor(self.Cx)
        self.choly = cho_factor(self.Cy)

    def condition(self):
        """ Conditions hyper-parameters of gaussian process.
        """

        from pyBA.distortion import d2
        from pyBA.distortion import optimise_HP

        # Initial hyperparameter vector
        HP0 = np.array([self.scale, self.amp])

        # Store handles to objects lists and mean processes
        A = self.A
        B = self.B
        P = self.P

        # Optimise hyperparameters
        ML_output = optimise_HP(A, B, P, HP0)
        ML_HP = ML_output[0]
        ML_lnprob = ML_output[1]

        # Reconstruct Dmap with conditioned hyperparameters
        # New GP hyperparameters
        self.scale = ML_HP[0]
        self.amp = ML_HP[1]
        self.hyperparams = {'scale': self.scale, 'amp': self.amp}

        # Define covariance matrix for data points
        xyarr = np.array([o.mu for o in A])
        self.d2 = d2(xyarr,xyarr)
        self.nuggetx = np.array([o.sigma[0,0] for o in A]) + \
             np.array([o.sigma[0,0] for o in B])
        self.nuggety = np.array([o.sigma[1,1] for o in A]) + \
             np.array([o.sigma[1,1] for o in B])

        self.build_covariance(self.scale,self.amp)
        
        return ML_HP, ML_lnprob
        
