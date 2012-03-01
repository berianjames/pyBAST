import bivarg
from pymc import deterministic, observed, Normal, MvNormalCov

# Instantiate bivariate
M = bivarg.Bivarg((1,0.5),((0.2,.05),(.05,.1)),0)
Mobs = MvNormalCov('Mobs',mu=M.mu,C=M.sigma,observed=True,value=M.mu)

# Sample from bivariate
p = bivarg.sample(M)

# Translate point to give new centre
p += (-.2,.2)

# Make new bivariate
N = bivarg.Bivarg(p,((0.1,.01),(.01,0.2)),0)
Nobs = MvNormalCov('Nobs',mu=N.mu,C=N.sigma,observed=True,value=N.mu)

# Now MCMC...
@deterministic
def DBx(Mobs=Mobs,Nobs=Nobs):
    #return bivarg.distance(M,bivarg.transform(N,(dx,dy)))
    return (Mobs[0]-Nobs[0])**2

@deterministic
def DBy(Mobs=Mobs,Nobs=Nobs):
    return (Mobs[1]-Nobs[1])**2

# Set priors on dx and dy
dx = Normal('dx',mu=DBx,tau=.2**(-2))
dy = Normal('dy',mu=DBy,tau=.2**(-2))

#obsDB = Normal('obsDB', mu=DB, tau=.1**(-2), observed=True, value=1)

#bivarg.draw((M,N))

