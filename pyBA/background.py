import numpy as np
from numpy.linalg import solve, det, inv
from pyBA.classes import Bgmap
from functools import reduce


def distance(M,N):
    """ Computes Bhattacharyya distance between two distributions
    """
    S = 0.5 * (N.sigma + M.sigma)
    #da = (1./8.) * np.dot( (N.mu-M.mu), solve(S, (N.mu-M.mu).T) )
    da = (1./8.) * np.dot( (N.mu-M.mu), inv(S) ).dot( (N.mu-M.mu).T )
    db = (1./2.) * np.log( det(S) / np.sqrt( N.det*M.det ) )
    return da + db

def suggest_mapping(M,N):
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

    Can also approximate background mapping likelihood distribution as a 
    multivariate normal distribution and reports back the mean and
    covariance matrix for the distribution.
    """
    from scipy.optimize import fmin_bfgs, fmin

    def lnprob(P,M=M,N=N,prior=prior):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        mapping parameter set P for mapping between two sets of objects
        M and N.
        """
        llik = 0.5 * np.sum( distance(M[i],N[i].transform(P))
                          for i in range(len(M)) )

        return llik + prior.llik(P)

    ML = fmin( lnprob,mu0,args=(M,N,prior),callback=None,
               xtol=1.0e-2, ftol=1.0e-6, disp=False, 
               maxiter=150 )

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

                sigma[i,j] = - (lnprob(P1,M,N) - lnprob(P2,M,N) - lnprob(P3,M,N) + lnprob(P4,M,N))
                sigma[i,j] = sigma[i,j] / (4*delta*delta)
                sigma[j,i] = sigma[i,j]


        sigma = inv(sigma)

        # Ensure variances are positive
        for i in range(7):
            if sigma[i,i] < 0:
                # If variance on diagonal is negative, flip its values
                #  and those of the associated covariances.
                sigma[i,:] = -1*sigma[i,:]
                sigma[:,i] = -1*sigma[:,i]
                sigma[i,i] = -1*sigma[i,i]

        # Ensure matrix can be Cholesky decomposed (i.e. that it is positive semidefinite)
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.linalg.LinAlgError:
            # Zero negative eigenvalues. This is the method of Higham (2002).
            E, V = np.linalg.eigh(sigma)
            E[E<0] = 1e-12
            sigma = V.dot(np.diag(E).dot(V.T))
        
        return Bgmap( mu=ML, sigma=sigma )

def MCMC(M,N,mu0=Bgmap().mu,prior=Bgmap(),nsamp=1000,nwalkers=20):
    """ Performs MCMC computation of likelihood distribution for the 
    background mapping between two frames.
    """
    import emcee

    def lnprob(P,M=M,N=N,prior=prior):
        """ Returns the log probability (\propto -0.5*chi^2) of the
        mapping parameter set P for mapping between two sets of objects
        M and N.
        """
        llik = -0.5 * np.sum( distance(M[i],N[i].transform(P))
                          for i in range(len(M)) )

        if np.all(np.isinf(np.diag(prior.sigma))):
            # De-facto uniform prior; don't bother computing prior llik.
            return llik
        else:
            return llik + prior.llik(P)

    ndim = 7
    p0 = [mu0+np.random.randn(ndim) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[M,N,prior])
    sampler.run_mcmc(p0, nsamp)

    return sampler

def cross_validate(M,N,k=2,mu0=Bgmap().mu,prior=Bgmap()):
    """ Performs k-fold cross-validation on the normal approximation
    to the likelihood surface for the mapping between the tie object
    lists M and N.
    """
    from sklearn.cross_validation import KFold

    # 1. Partition the data
    nties = len(M)
    n = nties / k

    # Randomly permute the indices, just in case
    ix_all = np.random.permutation(nties)
    
    # Get lists of indices
    kf = KFold(nties,k,indices=True)
    i = 0
    partition = np.empty(k,dtype='object')
    for train in kf: 
        partition[i] = ix_all[train[0]]
        i += 1


    # 2. Compute MAP Bgmap with normal approximation for each partition
    #    Each partition set is independent, so these can be done in parallel
    return map( lambda part: MAP(M[part],N[part],mu0=mu0,prior=prior), partition )

