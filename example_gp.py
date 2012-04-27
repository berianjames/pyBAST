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
print ix

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

#print mx
#print my
#print C

# Grid for regression
xmin = min([o.mu[0] for o in objectsA])
xmax = max([o.mu[0] for o in objectsA])
ymin = min([o.mu[1] for o in objectsA])
ymax = max([o.mu[1] for o in objectsA])

nres = 30
xs = np.linspace(xmin,xmax,nres)
ys = np.linspace(ymin,ymax,nres)
x,y = np.meshgrid(xs,ys)
xarr = np.array([x.flatten(),y.flatten()]).T

# Mapping from mean function
#vx = mx(xarr)
#vy = my(xarr)

# Regression from mean function without observation
from pymc.gp import point_eval
xout = point_eval(mx, C, xarr)
yout = point_eval(my, C, xarr)
vx = xout[0]
vy = yout[0]

xobs = np.array([objectsA[i].mu[0] for i in ix])
yobs = np.array([objectsA[i].mu[1] for i in ix])
vxobs = np.array([objectsA[i].mu[0] - objectsB[i].mu[0] for i in ix ])
vyobs = np.array([objectsA[i].mu[1] - objectsB[i].mu[1] for i in ix ])

# Plot mean function vectors
from pylab import figure, quiver, show
fig = figure(figsize=(16,16))
ax = fig.add_subplot(111, aspect='equal')
quiver(x,y,vx,vy,scale_units='width',scale=nres*nres)
quiver(xobs,yobs,vxobs,vyobs,color='r',scale_units='width',scale=nres*nres)
show()
