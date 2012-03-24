# bivarg.py
"""Basic routines for manipulating bivariate gaussians

"""
# Author:            Berian James (UCB)
# Written:            1-Oct-2011
# Modified:          27-Oct-2011  added det, chol and trace to __init__

import numpy as np
from numpy.linalg import solve, det, cholesky, eig

def draw(pdfs):
    """ Plot bivariates with matplotlib
    """
    from pylab import figure, show, rand
    from matplotlib.patches import Ellipse

    ells = [Ellipse(xy=pdfs[i].mu,width=pdfs[i].E[0,0],height=pdfs[i].E[1,1],angle=pdfs[i].theta)
            for i in xrange(len(pdfs))]

    fig = figure()
    ax = fig.add_subplot(111)
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)
        e.set_facecolor(rand(3))

    # Should modify limits to depend on inputs
    ax.set_xlim(0.5,1.5)
    ax.set_ylim(0.,1.0)
    show()

    return ells
