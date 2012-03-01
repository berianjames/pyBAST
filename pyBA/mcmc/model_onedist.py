import bivarg
from pymc import deterministic, Lognormal, Normal, Uniform
from numpy import pi

# Instantiate bivariate
M = bivarg.Bivarg((1,0.5),((0.02,.0),(.0,.01)),0)

# Sample from bivariate
p = bivarg.sample(M) #p = M.mu

# Translate point to give new centre
p = p + (-.2,.2)

# Make new bivariate
N = bivarg.Bivarg(p,((0.01,.0),(.0,0.02)),0)

# Set priors on dx and dy
dx = Normal('dx',mu=0,tau=.3**(-2))
#dx = Uniform('dx',loc=-0.5,scale=1.0)
dy = Normal('dy',mu=0,tau=.3**(-2))
#dy = Uniform('dy',loc=-0.5,scale=1.0)
theta = Normal('theta',mu=0,tau=.5**(-2))
Lx = Normal('Lx',mu=1,tau=.1**(-2))
Ly = Normal('Ly',mu=1,tau=.1**(-2))

# Now MCMC...
@deterministic
def DB(dx=dx,dy=dy,theta=theta,Lx=Lx,Ly=Ly):
    return bivarg.distance(M,bivarg.transform(N,(dx,dy),(Lx,Ly),theta))
    #return bivarg.distance(M,bivarg.transform(N,(dx,dy),theta))
    #return bivarg.distance(M,bivarg.transform(N,(dx,dy)))
    
obsDB = Normal('obsDB', mu=DB, tau=.01**(-2), observed=True, value=0)

