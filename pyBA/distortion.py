import numpy as np
from numpy import linspace, array, meshgrid, sqrt
from pyBA.classes import Bgmap
from numpy.linalg import norm
#from pymc.gp import Mean, Covariance, Realization, matern, point_eval, observe

def astrometry_cov(scale=1.,amp=.4):
    """ Define covariance function for gaussian process."""
    from pymc.gp import Covariance, matern

    C = Covariance(eval_fun = matern.euclidean, diff_degree=1.4,
                   scale = scale, amp = amp, rank_limit=00)

    return C

def astrometry_mean(T=Bgmap()):
    """ Defines the mean functions for the gaussian process
    astrometric solution."""
    from pymc.gp import Mean

    # Prep inputs
    dmu = T.mu[0:2]
    theta = T.mu[2]
    d0 = T.mu[3:5]
    L = T.mu[5:7]
    
    U = np.squeeze(np.array([ [np.cos(theta),-np.sin(theta)],
                              [np.sin(theta), np.cos(theta)] ]))

    def mx(x):
        """ Takes n x 2 array of coordinates and provides x-
        component of their transformation."""
        
        # Make use of broadcasting
        p = L*x + dmu - d0

        # But rotation must be done component-wise
        rx = U[0,0]*p[:,0] + U[0,1]*p[:,1]

        return rx + d0[0]

    def my(x):
        """ Takes n x 2 array of coordinates and provides y-
        component of their transformation."""
        
        # Make use of broadcasting
        p = L*x + dmu - d0

        # But rotation must be done component-wise
        ry = U[1,0]*p[:,0] + U[1,1]*p[:,1]

        return ry + d0[1]

    return Mean(mx), Mean(my)
