import bivarg
from pymc import deterministic, Lognormal, Normal, Uniform
from numpy import pi
from numpy.random import rand

ndists = 20 # Number of distributions

# Instantiate bivariate
M = [ bivarg.Bivarg((rand(),rand()),((0.02,.0),(.0,.01)),0) for i in range(ndists) ]

# Sample from bivariate
#p = bivarg.sample(M) #p = M.mu

# Translate point to give new centre
#p = p + (-.2,.2)

# Make new bivariates
p = (-.2,.2)
N = [ bivarg.transform(M[i],p) for i in range(ndists) ]
#N = bivarg.Bivarg(p,((0.01,.0),(.0,0.02)),0)

# Set priors on dx and dy
dx = Normal('dx',mu=0,tau=.3**(-2))
#dx = Uniform('dx',loc=-0.5,scale=1.0)
dy = Normal('dy',mu=0,tau=.3**(-2))
#dy = Uniform('dy',loc=-0.5,scale=1.0)
#theta = Normal('theta',mu=0,tau=.5**(-2))
#Lx = Normal('Lx',mu=1,tau=.1**(-2))
#Ly = Normal('Ly',mu=1,tau=.1**(-2))

# Now MCMC...
@deterministic
def DB(dx=dx,dy=dy):#,theta=theta,Lx=Lx,Ly=Ly):
    #dN = [bivarg.transform(distlistT[i],(dx,dy),(Lx,Ly),theta) for i in range(ndists)]
    dN = [bivarg.transform(N[i],(dx,dy)) for i in range(ndists)]

    D = [bivarg.distance(M[i],dN[i]) for i in range(ndists)]
    return sum(D) / ndists
    
obsDB = Normal('obsDB', mu=DB, tau=.01**(-2), observed=True, value=0)

