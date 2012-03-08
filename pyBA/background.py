import numpy as np
from scipy.linalg import solve, det, inv
from pyBA.classes import Bgmap

def distance(M,N):
    """ Computes Bhattacharyya distance between two distributions
    """
    S = 0.5 * (N.sigma + M.sigma)
    da = (1./8.) * np.dot( (N.mu-M.mu), solve(S, (N.mu-M.mu).T) )
    # da = (1./8.) * np.dot( (N.mu-M.mu), inv(S) ).dot( (N.mu-M.mu).T )
    db = (1./2.) * np.log( det(S) / np.sqrt( N.det*M.det ) )
    return da + db

def MAP(M,N,prior=Bgmap(),norm_approx=True):
    """Find the peak of the likelihood distribution for the 
    mapping between two image frames. Input is two lists
    of bivargs, of equal length, representing pairs of objects
    in the two frames, and a prior distribution on the background
    mapping.
    """
    from scipy.optimize import fmin,fmin_bfgs

    def lnprob(P,M=M,N=N,prior=prior):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        mapping parameter set P for mapping between two sets of objects
        M and N.
        """
        llik = 0.5 * sum( distance(M[i],N[i].transform(P))
                          for i in xrange(len(M)) )

        return llik + prior.llik(P)

    ML = fmin_bfgs( lnprob,prior.mu,args=(M,N,prior),callback=None,gtol=0.1)

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

def ML(M,N):
    """ Find peak of the likelihood distribution, ignoring information
    about priors on the mapping parameters. Input is two lists
    of bivargs of equal length, representing pairs of objects in the 
    two frames.
    """
    from scipy.optimize import fmin,fmin_bfgs

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

    return fmin_bfgs( lnprob,Bgmap().mu,args=(M,N),callback=None,gtol=0.1)
    
def NA(M,N,pri=Bgmap()):
    """Approximates background mapping likelihood distribution as a 
    multivariate normal distribution and reports back the mean and
    covariance matrix for the distribution.
    """
    
# def MCMC(M,N):
# Performs MCMC computation of likelihood distribution for the 
#  background mapping between two frames.
