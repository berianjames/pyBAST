from pylab import plot, show, quiver, figure
from matplotlib.patches import Ellipse
import numpy as np
from pyBA.classes import Bivarg, Bgmap
from pymc.gp import Mean, Covariance, matern

def draw_objects(objects=np.array( [Bivarg()] )):
    ells = [Ellipse(xy=O.mu, width=O.sigma[0,0],
                    height= O.sigma[1,1], angle = O.theta)
            for O in objects]

    fig = figure(figsize=(10,10))
    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        #e.set_alpha(.1)

    xmin = min([e.center[0]-e.width for e in ells])
    xmax = max([e.center[0]+e.width for e in ells])
    ymin = min([e.center[1]-e.height for e in ells])
    ymax = max([e.center[1]+e.height for e in ells])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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

def compute_displacements(objects = np.array([ Bivarg() ]),
                          objectsB = np.array([ Bivarg() ])):
    """From arrays of tie objects, return the locations of the centres
    of the first set of objects, and the displacements from this location
    to the tie object in the second list. Used in plotting to show the
    displacement between image frames.
    """

    nobj = len(objects)
    xobs = np.array([o.mu[0] for o in objects])
    yobs = np.array([o.mu[1] for o in objectsB])
    vxobs = np.array([objects[i].mu[0] - objectsB[i].mu[0] for i in range(nobj) ])
    vyobs = np.array([objects[i].mu[1] - objectsB[i].mu[1] for i in range(nobj) ])
    return xobs, yobs, vxobs, vyobs

def draw_MAP_background(objects = np.array([ Bivarg() ]),
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
    
    # Grid for regression
    x,y = make_grid(objects,res=res)

    # Perform evaluation of background function on grid
    xarr = np.array([x.flatten(),y.flatten()]).T
    xout = point_eval(mx, Cx, xarr)
    yout = point_eval(my, Cy, xarr)
    vx = xout[0]
    vy = yout[0]

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs = compute_displacements(objects, objectsB)

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)
    ax.autoscale(enable=None, axis='both', tight=True)
    show()

    return

def draw_realisation(objects = np.array([ Bivarg() ]),
                     objectsB = np.array([ Bivarg() ]),
                     mx=Mean(lambda x: x-x),
                     my=Mean(lambda y: y-y),
                     Cx=Covariance(eval_fun=matern.euclidean, diff_degree=1.4, amp = .4, scale = .1),
                     Cy=Covariance(eval_fun=matern.euclidean, diff_degree=1.4, amp = .4, scale = .1), 
                     res = 30):
    from pymc.gp import Realization, point_eval

    # Grid for regression
    x,y = make_grid(objects,res=res)

    # Draw realisation of gaussian process x- and y-components
    Rx = Realization(mx, Cx)
    Ry = Realization(my, Cy)

    # Evaluate gaussian processes on grid
    xarr = np.array([x.flatten(),y.flatten()]).T
    vx = Rx(xarr)
    vy = Ry(xarr)

    # Compute empirical displacements
    xobs, yobs, vxobs, vyobs = compute_displacements(objects, objectsB)

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)
    ax.autoscale(enable=None, axis='both', tight=True)
    show()
    
    return
