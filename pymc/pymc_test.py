import bivarg
import pymc
import mymodel
from itertools import chain

S = pymc.NormApprox(mymodel)
S.fit()
dx = S.mu[S.dx]
dy = S.mu[S.dy]
#theta = S.mu[S.theta]
#Lx = S.mu[S.Lx]
#Ly = S.mu[S.Ly]

M = mymodel.M
N = [bivarg.transform(mymodel.N[i],(dx,dy)) for i in range(20)]#,(Lx,Ly),theta)
ells = list(chain.from_iterable( (M,N,mymodel.N) ))


bivarg.draw( ells )




## MCMC run
#S = pymc.MCMC(mymodel, db='pickle')
#S.sample(iter=20500,burn=500, thin=20)
#pymc.Matplot.plot(S)

