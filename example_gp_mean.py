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
print t2 - t1

objectsB = np.array( [ pyBA.Bivarg(mu=data[i,5:7],sigma=data[i,7:10]) for i in range(nties) ] )
#obj_diff = [ objectsA[i] - objectsB[i] for i in range(nties) ]

# Select random subset of tie objects
nsamp = nties
ix = np.random.permutation(nties)[:nsamp]
#print ix

# Find maximum likelihood background transformation
from pyBA.background import distance
t1 = timeit.time.time()
S = pyBA.background.suggest_prior(objectsA,objectsB)
P = pyBA.background.MAP( objectsA[ix], objectsB[ix], mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True )
t2 = timeit.time.time()
print t2 - t1

print P.mu

# Gaussian process
from pyBA.distortion import astrometry_mean, astrometry_cov
mx,my = astrometry_mean(P)
C = astrometry_cov()

nres = 30

# Show mean function (the background transformation)
from pyBA.plotting import draw_MAP_background
draw_MAP_background(objectsA[ix],
                    objectsB[ix],
                    mx, my, C,
                    res=nres )

# Draw realisations of distortion map prior to observation
from pyBA.plotting import draw_realisation
draw_realisation(objectsA[ix],
                 objectsB[ix],
                 mx, my, C,
                 res = nres)

