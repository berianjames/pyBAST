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
from pyBA.background import distance
S = pyBA.background.suggest_mapping(objectsA,objectsB)
print S.mu

# Get maximum a posteriori background mapping parameters
P = pyBA.background.MAP( objectsA, objectsB, mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True )
print P.mu

# Create astrometric mapping object
D = pyBA.Amap(P,objectsA, objectsB)
#D.build_covariance(scale=10.0)

# Plotting prior to conditioning
#D.draw_realisation()

# Condition GP
D.condition()

print D.scale

print D.amp

#D.build_covariance(scale=500.0)

D.draw_realisation()
