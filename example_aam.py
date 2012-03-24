# pyBA test using some data from A. A. Miller, Esq.
import numpy as np
import pyBA
import timeit

# Load data
data = np.loadtxt('examples/astrom_match_stats')
nties = len(data)

# Parse catalogues into object list
t1 = timeit.time.time()
objectsA = np.array( [ pyBA.Bivarg(mu=data[i,0:2],sigma=data[i,2:5]) for i in range(nties) ] )
t2 = timeit.time.time()
#print objectsA[0].det, objectsA[0].trace
#print objectsA[0].E
#print objectsA[0].V
#print objectsA[0].sigma
#print objectsA[0].chol
print t2 - t1

objectsB = np.array( [ pyBA.Bivarg(mu=data[i,5:7],sigma=data[i,7:10]) for i in range(nties) ] )
#obj_diff = [ objectsA[i] - objectsB[i] for i in range(nties) ]

# Find maximum likelihood background transformation
ix = np.random.randint(0,nties,100)
#ix = np.arange(len(objectsA))
#print ix

from pyBA.background import distance
t1 = timeit.time.time()
#P = pyBA.background.MAP( objectsA[ix], objectsB[ix], prior=pyBA.Bgmap(), norm_approx=True );
#test1 = np.sum( distance(objectsA[i], objectsB[i]) for i in xrange(len(objectsA)) )
t2 = timeit.time.time()
print t2 - t1

t1 = timeit.time.time()
S = pyBA.background.suggest_prior(objectsA,objectsB)
#P2 = pyBA.background.ML( objectsA[ix], objectsB[ix], mu0=S.mu )
#P = pyBA.background.MAP( objectsA[ix], objectsB[ix], mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True )
#Psamp = pyBA.background.MCMC( objectsA[ix], objectsB[ix], mu0 = S.mu, nsamp=10)
#test1 = np.sum( distance(objectsA[i], objectsB[i]) for i in xrange(len(objectsA)) )
t2 = timeit.time.time()
print t2 - t1

#print P
#print P2

# Compute normal approximation to ML background distritbution
# ...

# Run MCMC on background transformation distribution
# ...
