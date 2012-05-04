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
nsamp = 500
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
Cx = astrometry_cov(scale = 100., amp = 1.)
Cy = astrometry_cov(scale = 100., amp = 1.)

nres = 30

# Show mean function (the background transformation)
#from pyBA.plotting import draw_MAP_background
#draw_MAP_background(objectsA[ix],
#                    objectsB[ix],
#                    mx, my, Cx, Cy,
#                    res=nres )

# Draw realisation of distortion map prior to observation
from pyBA.plotting import draw_realisation
#draw_realisation(objectsA[ix],
#                 objectsB[ix],
#                 mx, my, Cx, Cy,
#                 res = nres)

# Observe gaussian processes
from pyBA.distortion import regression
mxo, Cxo = regression(objectsA[ix],
                      objectsB[ix],
                      mx, Cx, direction='x')

myo, Cyo = regression(objectsA[ix],
                      objectsB[ix],
                      my, Cy, direction='y')

# Draw realisation of distortion map after observation
draw_realisation(objectsA[ix],
                 objectsB[ix],
                 mxo, myo, Cxo, Cyo,
                 res = nres)

