import numpy as np
from numpy import linspace, array, meshgrid, sqrt
from pyBA.classes import Bgmap, Bivarg
from numpy.linalg import norm
#from pymc.gp import Mean, Covariance, Realization, matern, point_eval, observe

def astrometry_cov(scale=100.,amp=1.):
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

        # Return x-displacement from original value
        return x[:,0] - rx + d0[0]

    def my(x):
        """ Takes n x 2 array of coordinates and provides y-
        component of their transformation."""
        
        # Make use of broadcasting
        p = L*x + dmu - d0

        # But rotation must be done component-wise
        ry = U[1,0]*p[:,0] + U[1,1]*p[:,1]

        # Return y-displacement from original value
        return x[:,1] - ry + d0[1]

    return Mean(mx), Mean(my)

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
    return xobs, yobs, vxobs, vyobs

def compute_residual(objectsA, objectsB, mx, my):
    """Compute residual between tie object displacements and 
    mean function of Gaussian process."""

    # Extract centres of objects in each frame
    obsA = array([o.mu for o in objectsA])
    obsB = array([o.mu for o in objectsB])

    # Compute residual between empirical displacements and mean function
    dx = (obsB[:,0] - obsA[:,0]) - mx(obsB)
    dy = (obsB[:,1] - obsA[:,1]) - my(obsB)

    return dx, dy

def regression(objectsA, objectsB, M, C, direction='x'):
    """ Perform regression on the gaussian processes for the 
    the distortion map. This updates the
    gaussian process, which previously contains only information
    from the background mapping (the mean function), to include
    local distortion information from the tie objects.

    Input: objectsA, objectsB - two objects lists
           M - Gaussian process mean
           C - Gaussian process covariance function
    """

    #from pyBA.plotting import compute_displacements
    from pymc.gp import observe

    # Compute displacements between frames for tie objects
    xobs, yobs, vxobs, vyobs = compute_displacements(objectsA, objectsB)

    obs = np.array([xobs.flatten(), yobs.flatten()]).T

    # Currently, the x- and y-component GP regression is performed
    # seperately, so observe should be run for each. The direction
    # keyword controls which direction is being used.
    if direction is 'x':
        data = vxobs.flatten()
    elif direction is 'y':
        data = vyobs.flatten()
       
    # Perform observation
    observe(M=M,C=C,
            obs_mesh = obs,
            obs_vals = data)

    return M,C

def MAP_hyperparameters(M, C, objectsA, objectsB, direction='x'):
    """Extremise the GP hyperparmaters based on the observed
    displacements between image frames."""
    
    #from pyBA.plotting import compute_displacements

    # Compute displacements between frames for tie objects
    xobs, yobs, vxobs, vyobs = compute_displacements(objectsA, objectsB)

    obs = np.array([xobs.flatten(), yobs.flatten()]).T

    # Define residual displacement from mean function
    dx, dy = compute_residual(objectsA, objectsB, M)
    if direction is 'x':
        data = dx
    elif direction is 'y':
        data = dy

    # Define objective function
    
