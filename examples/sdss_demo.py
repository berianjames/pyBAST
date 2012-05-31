# Demo script to analyse data from Stripe 82
#
# This is minimal version of the procedure carried out in the
#  corresponding notebook (sdss_demo.ipynb).

import pyBA
import sdss
import numpy as np

# Load data
data = sdss.csv2rec("match_astrom_342.1914_-0.90194_run4198.dat")
nties = len(data)
print nties

# Parse data array into objects
# the A objects will be the fiducial positions from the deep coadd
# the B objects will be source positions in this epoch
# both are given in arcseconds relative to a fiducial center
objectsA = np.array( [ pyBA.Bivarg(mu=[x["master_dra"],x["master_ddec"]],sigma=np.array([x["master_raerr"],x["master_decerr"]])) for x in data ] )
objectsB = np.array( [ pyBA.Bivarg(mu=[x["dra"],x["ddec"]], sigma=np.array([x["raerr"],x["decerr"]])) for x in data ] )

# Suggest starting point for background mapping
S = pyBA.background.suggest_mapping(objectsA,objectsB)

# Get maximum a posteriori background mapping parameters
P = pyBA.background.MAP( objectsA, objectsB, mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True )
print P.mu

# Create astrometric mapping object
D = pyBA.Amap(P,objectsA, objectsB)

# Condition GP
D.condition()
print D.scale
print D.amp

# Regression of a single point
print 'Single point regression'
xy = np.array([0,0])
R = D.regression(xy)
print R

# Regression of a single distribution
print 'Single distribution regression'
BV = pyBA.Bivarg()
R = D.regression(BV)
print R

# Regression of an array of points
print 'Multiple point regression'
xy = 100*np.random.rand(10,2) - 50
R = D.regression(xy)
print R

# Regression of an array of distributions
print 'Multiple distribution regression'
BV = [pyBA.Bivarg(mu=xyi, sigma=np.random.rand(2)) for xyi in xy]
R = D.regression(BV)
print R
