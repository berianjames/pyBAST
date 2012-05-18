# pyBA test using some data from Adam Miller
import numpy as np
import pyBA
import timeit

# Load data
data = np.loadtxt('examples/astrom_match_stats')
nties = len(data)

# Parse catalogues into object list
t1 = timeit.time.time()
objectsA = np.array( [ pyBA.Bivarg(mu=data[i,0:2],sigma=data[i,2:5]) for i in range(nties) ] )
objectsB = np.array( [ pyBA.Bivarg(mu=data[i,5:7],sigma=data[i,7:10]) for i in range(nties) ] )
t2 = timeit.time.time()
print t2 - t1

# Select random subset of tie objects
nsamp = 500
ix = np.random.permutation(nties)[:nsamp]
#print ix

# Find maximum likelihood background transformation
from pyBA.background import distance
t1 = timeit.time.time()
S = pyBA.background.suggest_mapping(objectsA,objectsB)
P = pyBA.background.MAP( objectsA[ix], objectsB[ix], mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True )
t2 = timeit.time.time()
print t2 - t1

print P.mu

# Create distortion map object
D = pyBA.Dmap(P,objectsA[ix], objectsB[ix])
#D = pyBA.Dmap(P,objectsA, objectsB)

nres = 30 # Density of interpolation grid points

## PRIOR TO OBSERVATION
# Show mean function (the background transformation)
#D.draw_background(res=nres)

# Draw realisation of distortion map prior to observation
#D.draw_realisation(res=nres)

# Plot residuals
#D.draw_residuals(res=nres)

## OBSERVATION
# Condition GP hyperparameters
D.condition()

# Show mean function (the background transformation)
D.draw_background(res=nres)

# Draw realisation of distortion map after observation
D.draw_realisation(res=nres)

