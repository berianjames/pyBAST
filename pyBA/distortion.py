import numpy as np
import scipy as sp
from numpy import array
from pyBA.classes import Bgmap, Bivarg
from numpy.linalg import eigh, slogdet
from numpy.linalg.linalg import LinAlgError

def d2(x,y):
    """For two n x 2 vectors (can be the same), compute the squared
    Euclidean distance between every pair of elements."""
    from scipy.spatial.distance import cdist
    
    return cdist(x, y, 'sqeuclidean')

def astrometry_cov(d2,scale=100.,amp=np.eye(2),var=None):
    """Evaluate covariance for gaussian process,
    given squared distance matrix and covariance parameters."""
    from numpy import exp, diag
    from scipy.linalg import block_diag
    
    S = exp( -d2 / scale ) # Correlation matrix
    C = np.kron(S, amp)    # Covariance matrix

    if var != None:
        # If scalar measurement uncertainty ('nugget') is used
        if np.size(var) == 1: 
            C += diag(np.tile(var,C.shape[0]))

        # If measurement uncertainty is a vector
        elif np.size(var) == C.shape[0]: # Vector
            C += diag(var)

        # If measurement uncertainty is a vector of 2x2 matrices
        elif np.shape(var) == (C.shape[0]/2, 2, 2):
            C += block_diag( *[v for v in var] )

    return C

def astrometry_mean(xy, T=Bgmap()):
    """Mean functions for the gaussian process
    astrometric solution. Takes affine transformation
    and applies it to n x 2 vector of points."""

    # Prep inputs
    dmu = T.mu[0:2]
    theta = T.mu[2]
    d0 = T.mu[3:5]
    L = T.mu[5:7]
    
    U = np.squeeze(np.array([ [np.cos(theta),-np.sin(theta)],
                              [np.sin(theta), np.cos(theta)] ]))
        
    p = (L*xy + dmu - d0).dot(U) + d0

    return xy - p

def compute_displacements(objectsA = np.array([ Bivarg() ]),
                          objectsB = np.array([ Bivarg() ])):
    """From arrays of tie objects, return the locations of the centres
    of the first set of objects, and the displacements from this location
    to the tie object in the second list. Used in plotting to show the
    displacement between image frames.
    """

    nobj = len(objectsA)
    xobs = np.array([o.mu[0] for o in objectsA])
    yobs = np.array([o.mu[1] for o in objectsA])
    vxobs = np.array([objectsB[i].mu[0] - objectsA[i].mu[0] for i in range(nobj) ])
    vyobs = np.array([objectsB[i].mu[1] - objectsA[i].mu[1] for i in range(nobj) ])
    sxobs = np.array([objectsB[i].sigma[0,0] + objectsA[i].sigma[0,0] for i in range(nobj) ])
    syobs = np.array([objectsB[i].sigma[1,1] + objectsA[i].sigma[1,1] for i in range(nobj) ])
    return xobs, yobs, vxobs, vyobs, sxobs, syobs

def compute_residual(objectsA, objectsB, P):
    """Compute residual between tie object displacements and 
    mean function of Gaussian process."""

    # Extract centres of objects in each frame
    obsA = array([o.mu for o in objectsA])
    obsB = array([o.mu for o in objectsB])

    # Compute residual between empirical displacements and mean function
    dxy = (obsB - obsA) - astrometry_mean(obsB, P)

    return dxy[:,0], dxy[:,1]

def realise(xyarr, P, scale, amp):
    """Evaluate GP realisation on given grid."""
     
    # Compute covariance matrix on grid
    d2_grid = d2(xyarr,xyarr)
    C = astrometry_cov(d2_grid, scale, amp)

    # Perform cholesky decomposition
    from numpy.linalg import cholesky
    A = cholesky(C)

    # Compute realisation on grid
    from numpy.random import randn
    v = astrometry_mean(xyarr, P)
    v += A.dot(randn(xyarr.size)).reshape(v.shape)

    return v[:,0], v[:,1]

def regression(objectsA, objectsB, xyarr, P, scale, amp, chol):
    """ Perform regression on the gaussian processes for the 
    the distortion map. This uses the input data to push known
    values onto a new grid, using the covariance properties of
    the GP."""

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs, _, _ = compute_displacements(objectsA, objectsB)
    xyobs = np.array([xobs.flatten(),yobs.flatten()]).T

    # Compute covariance matrix between meshes
    d2_grid = d2(xyarr,xyobs)
    C = astrometry_cov(d2_grid, scale, amp)

    # Compute mean function
    v = astrometry_mean(xyarr, P)

    # Get residuals to mean function
    from scipy.linalg import cho_solve
    dx, dy = compute_residual(objectsA, objectsB, P)
    dxy = np.array([dx, dy]).T.flatten()

    v += C.dot(cho_solve(chol, dxy)).reshape(v.shape)
    vx = v[:,0]
    vy = v[:,1]

    # Compute uncertainties in regression
    d2_grid = d2(xyarr, xyarr)
    K = astrometry_cov(d2_grid, scale, amp)
    S = K - C.dot(cho_solve(chol, C.T))

    s = np.diag(S).reshape(v.shape)
    sx = s[:,0]
    sy = s[:,1]
    
    return vx, vy, sx, sy

def optimise_HP(A, B, P, HP0):
    """ Condition hyperparameters of gaussian process associated 
    with astrometric mapping, based on observed data.
    """

    from scipy.optimize import fmin, fmin_bfgs
    from scipy.linalg import cho_factor, cho_solve

    # Get coordinates of objects in first frame
    xobs, yobs, _, _, _, _ = compute_displacements(A, B)
    xyobs = np.array([xobs.flatten(), yobs.flatten()]).T

    # Get residuals to mean function
    dx, dy = compute_residual(A, B, P)

    # CAUTION: This is tricky! Pack elements into a
    #  vector [(x1,y1), (x2,y2) ... (xn,yn)], 
    #  !NOT! [(x1,x2,...,xn), (y1,y2,...,yn)]
    dxy = np.array([dx, dy]).T.flatten()
    
    # Pre-compute distance matrix and grab nugget components
    d2_obs = d2(xyobs, xyobs)
    V = np.array([a.sigma for a in A]) + np.array([b.sigma for b in B])

    # Define loglikelihood function for gaussian process given data
    def lnprob_cov(C):

        # Get first term of loglikelihood expression (y * (1/C) * y.T)
        # Do computation using Cholesky decomposition
        try:
            U, luflag = cho_factor(C)
            x2 = cho_solve((U, luflag), dxy)
            L1 = dxy.dot(x2)
        except LinAlgError:
            #print "EV tweak"
            E, EV = eigh(C)
            Ep = 1/E
            Ep[abs(Ep)>1e5] = 0
            Ci = EV.T.dot(np.diag(Ep)).dot(EV)
            L1 = Ci.dot(dxy)
                    
        # Get second term of loglikelihood expression (log det C)
        sign, L2 = slogdet(C)

        # Why am I always confused by this?
        thing_to_be_minimised = (L1 + L2)

        return thing_to_be_minimised

    def make_pos(scale, amp, crossamp=0):
        """Make scale and amplitude parameters positive, and 
        amplitude matrix positive semi-definite. """

        # Ensure scale parameter is positive
        scale_pos = np.abs( scale )

        # Ensure amplitude term is positive
        amp_pos = np.abs( amp )

        # Ensure amplitude matrix will be positive semi-definite
        #  by clipping the magnitude of the off diagonal term to be
        #  just less than that of the  diagonal if required, but
        #  keep its overall sign.
        if crossamp*crossamp >= amp_pos*amp_pos:
            crossamp = 0#(0.99 * amp_pos) * (crossamp / np.abs(crossamp))

        amp_M = np.array([ [amp_pos, crossamp], [crossamp, amp_pos] ])

        return scale_pos, amp_M

    # Define loglikelihood function for hyperparameter vector
    def lnprob_HP(HP):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        hyperparameter set HP for the Gaussian process.
        """
            
        # Make input parameters physically plausible
        scale_pos, ampM_pos = make_pos(*HP)

        # Build trial covariance matrix
        c_try = astrometry_cov(d2_obs, scale_pos, ampM_pos, var=V)
        
        # Evaluate loglikelihood
        llik = lnprob_cov(c_try)

        #print scale_pos
        #print ampM_pos
        #print llik
        return llik

    # Perform optimisation
    #ML_HP = fmin(lnprob_HP,HP0, xtol=1.0e-2, ftol=1.0e-6, disp=False)
    ML_HP = fmin_bfgs(lnprob_HP,HP0, disp=False)

    scale_pos, ampM_pos = make_pos(*ML_HP)
    return scale_pos, ampM_pos, lnprob_HP(ML_HP)
    
