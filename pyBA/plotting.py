from pylab import plot, show, quiver, figure, gca, imshow, colorbar, draw
from matplotlib.patches import Ellipse
import numpy as np
from pyBA.classes import Bivarg, Bgmap
from matplotlib import pyplot, mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import matplotlib.cm as cmx


def draw_objects(objects=np.array( [Bivarg()] ), replot=False, alpha=0.2,figsize=(10,10),scale=1.0,
                show=False,colors="b",label=False):
    """
    scale : allow you to blow up or down the ellipses (defualt: 1.0)
    alpha: transparency for each ellipse (scalar or vector of length objects)
    """
    ells = [Ellipse(xy=O.mu, 
                    width=scale*2.0*np.sqrt(O.E[0,0]),
                    height=scale*2.0*np.sqrt(O.E[1,1]), 
                    angle=O.theta)
            for O in objects]
    
    jet = cm = plt.get_cmap('jet') 
    
            
    # Decide if plot is to be on top of whatever is already plotted
    if not replot:
        fig = figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect='equal')
    else:
        ax = gca()

    if isinstance(alpha,float) or isinstance(alpha,int):
        alphas = np.ones(len(ells))*alpha
    else:
        if len(alpha) != len(ells):
            alphas = np.ones(len(ells))*alpha[0]
        else:
            alphas = alpha

    if not isinstance(colors,list):
        colors = [colors for x in range(len(ells))]
                            
    if len(colors) != len(ells):
        colors = [colors[0] for x in range(len(ells))]
        
    cNorm  = cols.Normalize(vmin=colors[0], vmax=colors[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for i,e in enumerate(ells):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(alphas[i])
        e.set_facecolor(scalarMap.to_rgba(colors[i]))
    
    if label:
        ax.set_xlabel("RA offset arcsec")
        ax.set_ylabel("DEC offset arcsec")
        
    #ax.autoscale(enable=None, axis='both', tight=True)

    draw()
    
    if show:
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

def draw_realisation(objectsA, objectsB, P, scale, amp, chol, res = 30):

    # Grid for regression
    x,y = make_grid(objectsA,res=res)
    xyarr = np.array([x.flatten(),y.flatten()]).T

    # If no cholesky matrix A provided, assume that we are
    #  drawing realisation on grid without using observed data
    if chol == None:
        
        from pyBA.distortion import realise
        vx, vy = realise(xyarr, P, scale, amp)
        sx, sy = None, None

    # Otherwise, use cholesky data to perform regression
    else:

        from pyBA.distortion import regression
        vx, vy, sx, sy = regression(objectsA, objectsB, xyarr, P, 
                                    scale, amp, chol)

    # Get xy coordinates of base of vectors
    from pyBA.distortion import compute_displacements
    xobs, yobs, vxobs, vyobs, _, _ = compute_displacements(objectsA, objectsB)

    # Matplotlib plotting
    fig = figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    quiver(x,y,vx,vy,scale_units='width',scale=res*res)
    
    # If uncertainties are provided, plot them as a background image and colour the
    #  data vectors in white
    if sx != None:
        quiver(xobs,yobs,vxobs,vyobs,color='w',scale_units='width',scale=res*res)

        sarr = np.array(sx + sy).reshape( x.shape )
        imshow(np.sqrt(sarr), origin='upper', extent=(x.min(), x.max(), y.min(), y.max()),
               interpolation=None)
        colorbar()

    else:
        # Otherwise, no background and data vectors in red
        quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=res*res)

            
    ax.autoscale(enable=None, axis='both', tight=True)
    show()

    return 
