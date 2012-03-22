import numpy as np
from scipy.linalg import solve, det, inv
from pyBA.classes import Bgmap

def distance(M,N):
    """ Computes Bhattacharyya distance between two distributions
    """
    S = 0.5 * (N.sigma + M.sigma)
    #da = (1./8.) * np.dot( (N.mu-M.mu), solve(S, (N.mu-M.mu).T) )
    da = (1./8.) * np.dot( (N.mu-M.mu), inv(S) ).dot( (N.mu-M.mu).T )
    db = (1./2.) * np.log( det(S) / np.sqrt( N.det*M.det ) )
    return da + db

def suggest_prior(M,N):
    """ Suggests a start point for the background mapping fitting between
    two sets of objects. Formally, this is bad, as the data are being used
    twice. However, the likelihood surface is generally (always?) unimodal
    and smooth so that using the suggested starting point will not 
    change the outcome and will speed computation significantly.

    Input: M, N - lists (or nparrays) of Bivargs
    Output: Bgmap object with infinite variance (i.e. uniform prior)
    """
    nties = len(M)
    Mmu = [o.mu for o in M]
    Nmu = [o.mu for o in N]

    # Estimate translation by differencing means of object lists
    muM = reduce( lambda x,y: x+y, Mmu ) / nties
    muN = reduce( lambda x,y: x+y, Nmu ) / nties
    dx = np.array( muM - muN )

    # Estimate scalings by ratioing ranges of object lists
    rangeM = np.max( Mmu, axis=0 ) - np.min( Mmu, axis=0 )
    rangeN = np.max( Nmu, axis=0 ) - np.min( Nmu, axis=0 )
    L = rangeM / rangeN

    # Estimate rotation? Not sure how!
    theta = 0.0

    # Estimate centre of rotation? 
    d0 = np.array([0., 0.])

    return Bgmap( dx=dx,theta=theta,d0=d0,L=L )

def MAP(M,N,mu0=Bgmap().mu,prior=Bgmap(),norm_approx=True):
    """Find the peak of the likelihood distribution for the 
    mapping between two image frames. Input is two lists
    of bivargs, of equal length, representing pairs of objects
    in the two frames, and perhaps a suggested starting point 
    for the fitter and a prior distribution on the background
    mapping.
    """
    from scipy.optimize import fmin_bfgs

    def lnprob(P,M=M,N=N,prior=prior):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        mapping parameter set P for mapping between two sets of objects
        M and N.
        """
        llik = 0.5 * np.sum( distance(M[i],N[i].transform(P))
                          for i in xrange(len(M)) )

        return llik + prior.llik(P)

    ML = fmin_bfgs( lnprob,prior.mu,args=(M,N,prior),callback=None,gtol=0.1 )

    if norm_approx is False:
        return Bgmap(mu=ML)
    else:
        # Compute covariance matrix
        sigma = np.empty( (7,7) )
        delta = 1e-4
        units = np.eye(7)
        for i in range(7):
            for j in range(i,7):
                P1 = ML + delta*units[i,:] + delta*units[j,:]
                P2 = ML - delta*units[i,:] + delta*units[j,:]
                P3 = ML + delta*units[i,:] - delta*units[j,:]
                P4 = ML - delta*units[i,:] - delta*units[j,:]

                sigma[i,j] = lnprob(P1,M,N) - lnprob(P2,M,N) - lnprob(P3,M,N) + lnprob(P4,M,N)
                sigma[i,j] = sigma[i,j] / (4*delta*delta)
                sigma[j,i] = sigma[i,j]

        return Bgmap( mu=ML, sigma=sigma )

def ML(M,N,mu0=Bgmap().mu):
    """ Find peak of the likelihood distribution, ignoring information
    about priors on the mapping parameters. Input is two lists
    of bivargs of equal length, representing pairs of objects in the 
    two frames, and perhaps a starting point mu0 for the fitter.
    """
    from scipy.optimize import fmin_bfgs

    def lnprob(P,M=M,N=N):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        mapping parameter set P for mapping between two sets of objects
        M and N.
        """

        return 0.5 * sum( distance(M[i],N[i].transform(P))
                                    for i in xrange(len(M)) )

    def debug_callback(xk):
        print xk#, lnprob(xk)
        return

    #return fmin_powell( lnprob,mu0,args=(M,N),callback=debug_callback,ftol=0.1)
    return fmin_bfgs( lnprob,mu0,args=(M,N),callback=debug_callback,gtol=0.1)
    #return fmin( lnprob, mu0, args=(M,N), callback=debug_callback, disp=True )
    
def NA(M,N,pri=Bgmap()):
    """Approximates background mapping likelihood distribution as a 
    multivariate normal distribution and reports back the mean and
    covariance matrix for the distribution.
    """
    
# def MCMC(M,N):
# Performs MCMC computation of likelihood distribution for the 
#  background mapping between two frames.
