import numpy as np
from numpy import linspace, array, meshgrid, sqrt
from pyBA.classes import Bgmap, Bivarg
from numpy.linalg import norm

def d2(x,y):
    """For two n x 2 vectors (can be the same), compute the squared
    Euclidean distance between every pair of elements."""
    from scipy.spatial.distance import cdist
    
    return cdist(x, y, 'sqeuclidean')

def astrometry_cov(d2,scale=100.,amp=1.,nugget=None):
    """Evaluate covariance for gaussian process,
    given squared distance matrix and covariance parameters."""
    from numpy import exp, diag
    
    C = amp * exp( -d2 / scale )

    if nugget != None:
        # Assume nugget is a vector to be applied along the diagonal
        C += diag(nugget)

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
    from numpy.random import standard_normal
    v = astrometry_mean(xyarr, P)
    v += A.dot(standard_normal(xyarr.shape))

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

    vx = C.dot(cho_solve(chol, dx))
    vy = C.dot(cho_solve(chol, dy))

    # Add mean function
    vx += v[:,0]
    vy += v[:,1]

    # Compute uncertainties in regression
    d2_grid = d2(xyarr, xyarr)
    K = astrometry_cov(d2_grid, scale, amp)
    S = K - C.dot(cho_solve(chol, C.T))

    print np.diag(S).reshape( (30, 30) )
    sx = np.diag(S)
    sy = np.diag(S)
    
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
    
    # Pre-compute distance matrix
    d2_obs = d2(xyobs, xyobs)

    # Define loglikelihood function for gaussian process given data
    def lnprob_cov(C,direction):
        
        # Cholesky computation with numpy
        U, luflag = cho_factor(C)
        
        # Get correct vector of residuals
        if direction is 'x':
            y = dx
        elif direction is 'y':
            y = dy

        # Get first term of loglikelihood expression (y * (1/C) * y.T)
        x2 = cho_solve((U, luflag), y)
        L1 = y.dot(x2)

        # Get second term of loglikelihood expression (2*pi log det C)
        L2 = 2 * np.pi *  np.sum( 2*np.log(np.diag(U)) )

        # Why am I always confused by this?
        thing_to_be_minimised = L1 + L2

        return thing_to_be_minimised

    # Define loglikelihood function for hyperparameter vector
    def lnprob_HP(HP):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        hyperparameter set HP for the Gaussian process.
        """
            
        # Ensure parameters are positive
        HPpos = np.abs( HP )

        #print HPpos

        cx_try = astrometry_cov(d2_obs, *HPpos)
        cy_try = astrometry_cov(d2_obs, *HPpos)
        
        llik = lnprob_cov(cx_try,direction='x') + \
            lnprob_cov(cy_try,direction='y')

        #print HPpos, llik
        return llik

    # Perform optimisation
    ML_HP = fmin(lnprob_HP,HP0, xtol=1.0e-2, ftol=1.0e-6, disp=False)

    return ML_HP, lnprob_HP(ML_HP)
    
