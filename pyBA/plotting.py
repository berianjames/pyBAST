from pylab import plot, show, quiver, figure, gca
from matplotlib.patches import Ellipse
import numpy as np
from pyBA.classes import Bivarg, Bgmap

def draw_objects(objects=np.array( [Bivarg()] ), replot='no'):
    ells = [Ellipse(xy=O.mu, width=O.sigma[0,0],
                    height= O.sigma[1,1], angle = O.theta)
            for O in objects]

    # Decide if plot is to be on top of whatever is already plotted
    if replot is 'no':
        fig = figure(figsize=(10,10))
        ax = fig.add_subplot(111, aspect='equal')
    else:
        ax = gca()

    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(.1)

    ax.autoscale(enable=None, axis='both', tight=True)
    if replot is 'no':
        show()

    return ells

def make_grid(objects = np.array([ Bivarg() ]), res=30):
    """Makes evenly-space two-dimensional grid of
    object locations of given resolution. Uses input object
    list to define ranges.

    Input: objects - list or nparray of Bivargs
           res - scalar, density of grid
    """
    
    xmin = min([o.mu[0] for o in objects])
    xmax = max([o.mu[0] for o in objects])
    ymin = min([o.mu[1] for o in objects])
    ymax = max([o.mu[1] for o in objects])

    xs = np.linspace(xmin,xmax,res)
    ys = np.linspace(ymin,ymax,res)
    x,y = np.meshgrid(xs,ys)

    return x,y

def draw_MAP_background(objectsA = np.array([ Bivarg() ]),
                        objectsB = np.array([ Bivarg() ]),
                        P = Bgmap(),
                        res = 30):
    """ Plot the background parametric mapping between frames (the mean function
    for the gaussian process) on a grid of given resolution. Overplot observed
    displacements from lists of tie objects.
    """

    from pyBA.distortion import compute_displacements, astrometry_mean
    from numpy import array, sqrt

    # Grid for regression
    x,y = make_grid(objectsA,res=res)

    # Perform evaluation of background function on grid
    xy = np.array([x.flatten(),y.flatten()]).T
    vxy = astrometry_mean(xy, P)
    vx, vy = vxy[:,0], vxy[:,1]

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs, sxobs, syobs = compute_displacements(objectsA, objectsB)

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)
    ax.autoscale(enable=None, axis='both', tight=True)

    # Also plot error ellipses on interpolated points
    #ellipses = array([ Bivarg( mu = array([xarr[i,0] + vx[i], xarr[i,1] + vy[i]]),
    #                           sigma = array([ sx[i], sy[i] ]) )
    #                   for i in range(len(xarr)) ])
    #draw_objects(ellipses, replot='yes')

    show()

    return

def draw_MAP_residuals(objectsA, objectsB, P, scaled='no'):
    from pyBA.distortion import compute_displacements, compute_residual
    from numpy import array
    
    # Compute displacements between frames for tie objects
    xobs, yobs, vxobs, vyobs, sxobs, syobs = compute_displacements(objectsA, objectsB)

    # Compute residual
    dx, dy = compute_residual(objectsA, objectsB, P)

    # Draw residuals
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    if scaled is 'yes':
        # Allow relative scaling of arrows
        quiver(xobs,yobs,dx,dy)
    else:
        # Show residuals in absolute size (often very tiny), with uncertainties

        # Also plot error ellipses
        ellipses = array([ Bivarg( mu = array([xobs[i] + dx[i], yobs[i] + dy[i]]),
                                   sigma = objectsA[i].sigma + objectsB[i].sigma )
                           for i in range(len(objectsA)) ])
        draw_objects(ellipses, replot='yes')

        # Residuals
        quiver(xobs,yobs,dx,dy,color='r', angles='xy', scale_units='xy', scale=1)
    ax.autoscale(enable=None, axis='both', tight=True)
    show()

def draw_realisation(objectsA = np.array([ Bivarg() ]),
                     objectsB = np.array([ Bivarg() ]),
                     P = Bgmap(), scale=100., amp= 1.,
                     chol = None, res = 30):

    from pyBA.distortion import compute_displacements, d2, compute_residual
    from pyBA.distortion import astrometry_cov, astrometry_mean

    # Grid for regression
    x,y = make_grid(objectsA,res=res)
    xyarr = np.array([x.flatten(),y.flatten()]).T

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs, _, _ = compute_displacements(objectsA, objectsB)
    xyobs = np.array([xobs.flatten(),yobs.flatten()]).T

    # Compute mean function
    v = astrometry_mean(xyarr, P)

    # If no cholesky matrix A provided, assume that we are
    #  drawing realisation on grid without using observed data
    if chol == None:

        # Compute covariance matrix on grid
        d2_grid = d2(xyarr,xyarr)
        C = astrometry_cov(d2_grid, scale, amp)

        # Perform cholesky decomposition
        from numpy.linalg import cholesky
        A = cholesky(C)

        # Compute realisation on grid
        from numpy.random import standard_normal
        v += A.dot(standard_normal(xyarr.shape))

    # Otherwise, use cholesky data to perform regression
    else:

        # Compute covariance matrix between meshes
        d2_grid = d2(xyarr,xyobs)
        C = astrometry_cov(d2_grid, scale, amp)

        # Get residuals to mean function
        from scipy.linalg import cho_solve
        dx, dy = compute_residual(objectsA, objectsB, P)

        vx = C.dot(cho_solve(chol, dx))
        vy = C.dot(cho_solve(chol, dy))

        v += np.array([vx, vy]).T

    # Break into components for plotting
    vx, vy = v[:,0], v[:,1]

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)
    ax.autoscale(enable=None, axis='both', tight=True)
    show()
    
    return
