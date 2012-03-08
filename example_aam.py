# pyBA test using some data from A. A. Miller, Esq.
import numpy as np
import pyBA
import timeit

# Load data
data = np.loadtxt('examples/astrom_match_stats')
nties = len(data)

# Parse catalogues into object list
objectsA = np.array( [ pyBA.Bivarg(mu=data[i,0:2],sigma=data[i,2:5]) for i in range(nties) ] )
objectsB = np.array( [ pyBA.Bivarg(mu=data[i,5:7],sigma=data[i,7:10]) for i in range(nties) ] )
#obj_diff = [ objectsA[i] - objectsB[i] for i in range(nties) ]

# Find maximum likelihood background transformation
ix = np.random.randint(0,nties,100)
#ix = np.arange(len(objectsA))
print ix

t1 = timeit.time.time()
#P = pyBA.background.ML( objectsA[ix], objectsB[ix] )
P = pyBA.background.MAP( objectsA[ix], objectsB[ix], prior=pyBA.Bgmap(), norm_approx=True );
t2 = timeit.time.time()
print t2 - t1

type(P)
print P.mu 
print P.sigma

# Compute normal approximation to ML background distritbution
# ...

# Run MCMC on background transformation distribution
# ...
