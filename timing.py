import numpy as np
import pyBA
import timeit

t1 = timeit.time.time()
np.array( [ pyBA.Bivarg() for i in range(1000) ] )
t2 = timeit.time.time()

print (t2 - t1) / 1000
