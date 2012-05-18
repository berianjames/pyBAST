from pylab import plot, show, quiver, figure, gca
from matplotlib.patches import Ellipse
import numpy as np
from pyBA.classes import Bivarg, Bgmap
from pymc.gp import Mean, Covariance, matern

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
                        mx=Mean(lambda x: x-x),
                        my=Mean(lambda y: y-y),
                        Cx=Covariance(eval_fun=matern.euclidean, diff_degree=1.4, amp = .4, scale = .1),
                        Cy=Covariance(eval_fun=matern.euclidean, diff_degree=1.4, amp = .4, scale = .1),
                        res = 30):
    """ Plot the background parametric mapping between frames (the mean function
    for the gaussian process) on a grid of given resolution. Overplot observed
    displacements from lists of tie objects.
    """
    from pymc.gp import point_eval
    from pyBA.distortion import compute_displacements
    from numpy import array, sqrt

    # Grid for regression
    x,y = make_grid(objectsA,res=res)

    # Perform evaluation of background function on grid
    xarr = np.array([x.flatten(),y.flatten()]).T
    vx, sx = point_eval(mx, Cx, xarr)
    vy, sy = point_eval(my, Cy, xarr)

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs = compute_displacements(objectsA, objectsB)

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)
    ax.autoscale(enable=None, axis='both', tight=True)

    # Also plot error ellipses on interpolated points
    ellipses = array([ Bivarg( mu = array([xarr[i,0] + vx[i], xarr[i,1] + vy[i]]),
                               sigma = array([sqrt(sx[i]), sqrt(sy[i])]) )
                       for i in range(len(xarr)) ])
    draw_objects(ellipses, replot='yes')

    show()

    return

def draw_MAP_residuals(objectsA, objectsB, mx, my, scaled='no'):
    from pyBA.distortion import compute_displacements, compute_residual
    from pymc.gp import point_eval
    from numpy import array
    
    # Compute displacements between frames for tie objects
    xobs, yobs, vxobs, vyobs = compute_displacements(objectsA, objectsB)

    # Compute residual
    dx, dy = compute_residual(objectsA, objectsB, mx, my)

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
                     mx=Mean(lambda x: x-x),
                     my=Mean(lambda y: y-y),
                     Cx=Covariance(eval_fun=matern.euclidean, diff_degree=1.4, amp = .4, scale = .1),
                     Cy=Covariance(eval_fun=matern.euclidean, diff_degree=1.4, amp = .4, scale = .1), 
                     res = 30):
    from pymc.gp import Realization, point_eval
    from pyBA.distortion import compute_displacements

    # Grid for regression
    x,y = make_grid(objectsA,res=res)

    # Draw realisation of gaussian process x- and y-components
    Rx = Realization(mx, Cx)
    Ry = Realization(my, Cy)

    # Evaluate gaussian processes on grid
    xarr = np.array([x.flatten(),y.flatten()]).T
    vx = Rx(xarr)
    vy = Ry(xarr)

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs = compute_displacements(objectsA, objectsB)

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)
    ax.autoscale(enable=None, axis='both', tight=True)
    show()
    
    return
