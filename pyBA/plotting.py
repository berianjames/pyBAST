from pylab import plot, show, quiver, figure
from matplotlib.patches import Ellipse
import numpy as np
from pyBA.classes import Bivarg

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
