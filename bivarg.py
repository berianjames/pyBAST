# bivarg.py
"""Basic routines for manipulating bivariate gaussians

"""
# Author:            Berian James (UCB)
# Written:            1-Oct-2011
# Modified:          27-Oct-2011  added det, chol and trace to __init__

import numpy as np
from numpy.linalg import solve, det, cholesky, eig

class Bivarg:
    """ Implements bivariate gaussian structure and routines to modify it.
    """

    __author__ = "Berian James"
    __version__ = "0.1"
    __email__ = "berian@berkeley.edu"

    def __init__(self,mu=np.array([0.,0.]),sigma=np.array([ [1.,0.],[0.,1.] ]),theta=0):
        self.mu = np.squeeze(np.array(mu))
        
        self.E,self.V = eig(np.array(sigma))
        self.E = np.diag(np.real(self.E))
        U = np.array([ [np.cos(theta),-np.sin(theta)],
                       [np.sin(theta), np.cos(theta)] ])
        self.V = np.dot(U,self.V)
        self.sigma = np.dot( self.V, np.dot(self.E,self.V.T) )

        self.det = det(self.sigma)
        self.chol = cholesky(self.sigma)
        self.trace = np.trace(self.sigma)
        self.theta = np.math.degrees(np.math.atan2(self.V[0,1],self.V[0,0]))
        return
        
def distance(M,N):
    """ Computes Bhattacharyya distance between two distributions
    """
    S = 0.5 * (N.sigma + M.sigma)
    da = (1./8.) * np.dot( (N.mu-M.mu), solve(S, (N.mu-M.mu).T) )
    #db = (1./2.) * np.log( det(S) / np.sqrt( det(N.E)*det(M.E) ) )
    #return da + db
    return da

def transform(M,dmu=np.array([0,0]),L=np.array([1,1]),theta=0.,d0=np.array([0,0])):
    """ Maps a bivariate gaussian distribution (M) to another
    bivariate gaussian by translation (dmu), scaling of
    the principal axes (L) and rotation (theta) about a given point (d0).
    """
    # Prep inputs
    dmu = np.squeeze(dmu)
    theta = np.squeeze(theta)
    L = np.squeeze(L)
    d0 = np.squeeze(d0)

    # Calculate transformed centre
    U = np.squeeze(np.array([ [np.cos(theta),-np.sin(theta)],
                              [np.sin(theta), np.cos(theta)] ]))
    mu = np.dot(U,(M.mu * L + dmu) - d0) + d0

    # Calculate transformed covariance
    L = np.diag(L)
    V = np.dot(U,M.V)
    E = np.dot(L,M.E)
    sigma = np.dot( V, np.dot(E,V.T) ) 
    
    # Create new bivarg object with transformed values
    N = Bivarg(mu,sigma,0)
    return N

def sample(M=Bivarg(),n=1):
    """ Draw n samples from bivariate distribution M
    """
    
    vals = np.zeros( (n,2) )
    stds = np.random.randn(2,n)
    for i in range(0,n):
        vals[i] = M.mu + np.dot( M.chol, stds[:,i] ).T

    return vals

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
