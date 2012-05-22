***************************************
 pyBAST: Bayesian Astrometry in Python
***************************************

:Date: May 22, 2012
:Version: 0.3
:Authors: Berian James, Josh Bloom
:Web site: https://github.com/berianjames/pyBAST
:Copyright: This document has been placed in the public domain.
:License: This code is released under the MIT license.

===========================
pyBAST: Bayesian Astrometry
===========================

pyBAST is a Python implementation of the Bayesian Astrometry 
framework. It provides a module for handling probability
distributions that represent astronomical objects and for 
analysing the changes to these distributions between images.

Typical interactive use might look like::

    >>> import pyBA
    >>> data = np.loadtxt('examples/astrom_match_stats')
    >>> nties = len(data)

    # Load array data into objects
    >>> objectsA = np.array([pyBA.Bivarg(mu=data[i,0:2], sigma=data[i,2:5]) for i in range(nties)])
    >>>	objectsB = np.array([pyBA.Bivarg(mu=data[i,5:7], sigma=data[i,7:10]) for i in range(nties)])

    # Select random subset of objects (speeds up testing)
    >>> nsamp = 100
    >>> ix = np.random.permutation(nties)[:nsamp]

    # Compute background mapping
    >>> from pyBA.background import distance
    >>> S = pyBA.background.suggest_mapping(objectsA,objectsB)
    >>> P = pyBA.background.MAP(objectsA[ix], objectsB[ix], mu0=S.mu, prior=pyBA.Bgmap(), norm_approx=True)

    # Create astrometric mapping and condition the local distortions
    >>> D = pyBA.Amap(P,objectsA[ix], objectsB[ix])
    >>> D.condition()

    # Plot regression onto regular grid
    >>> D.draw_realisation(res=nres)

This functionality is provided in an example script (pyBAST_example.py) and also, with detailed comments, in an iPython notebook (``pyBAST_example.ipynb``; also in ``pyBAST_example.pdf``).

For non-interactive use, a command utility ``pyBAST`` is provided::

    > ./pyBAST -h 
    usage: pyBAST [-h] {fit,summary,apply} ...

    Perform probabilistic astrometry with pyBA.

    positional arguments:
       {fit,summary,apply}  pyBA command option: should be 'fit', 'summary' or
                            'apply'
         fit                determine astrometric mapping solution
         summary            summarize astrometric mapping solution
         apply              apply astrometric mapping solution at new locations

    optional arguments:
      -h, --help           show this help message and exit

This provides the functionality to fit an astrometric solution (which is written compactly to disk), summarize that solution and then apply it to arbitrary locations::

    > ./pyBAST fit -h
    usage: pyBAST fit [-h] [-s N] [-r n] file [output]

    Determine astrometric mapping solution

    positional arguments:
       file                 Path to input data file
       output               Optional[=file.pyBA] path to save output pyBA solution.

    optional arguments:
     -h, --help           show this help message and exit
     -s N, --subsample N  Use only N random objects from data set
     -r n, --reject n     Excludes objects with > n sigma residuals

    > ./pyBAST summary -h
    usage: pyBAST summary [-h] [-p] file

    Summarize astrometric mapping solution

    positional arguments:
      file        Path to astrometric mapping file

    optional arguments:
      -h, --help  show this help message and exit
      -p, --plot  Plot astrometric map on grid.

    > ./pyBAST apply -h  
    usage: pyBAST apply [-h] [-xy x y] [-g res] file [batchfile]

    positional arguments:
      file                Path to astrometric mapping file
      batchfile           Optional path to file with list of coordinates

    optional arguments:
      -h, --help          show this help message and exit
      -xy x y             Map coordinate pair (x,y)
      -g res, --grid res  Map grid of coordinates with density res

What can Bayesian Astrometry in pyBAST do?
==========================================

pyBAST provides:

* Classes for respresenting astronomical objects
  and astrometric mappings as probability distributions

* Maximum likelihood and multivariate normal likelihood
  approximation routines with these objects.

* A full *non-parametric* astrometic analysis of local distortions using gaussian processes

* MCMC likelihood computation (using emcee)

* A helpful set of examples

It aspires to (but does not yet) provide:
  
* An interface with wcslib and pyfits

* Handling of priors on object proper motions, parallax

* Robust support for parallel computation on cluster (though n.b. that native threading via BLAS will occur by default)

See the TODO and ROADMAP documents for short- and long-term
targets, respectively.

The rest of this README provides a short overview of the
package. Detailed instructions will be provided in the
documentation (by version 0.4).

Representation of astronomical objects
======================================

Bayesian astrometry represents astronomical objects as
bivariate gaussians. The **bivarg** module provides the
routines for creating these objects. Upon initialisation,
these objects are assigned the following properties:

1. Fundamental descriptors of the distribution:

* *mu*: A two-vector representing the central location
  of the objects

* *sigma*: A 2x2 covariance matrix representing the
  uncertainity in the location of the object.

* *theta*: A complementary representation of the covariance
  between the x- and y-coordinates.

2. Derived quantities used for manipulating objects

* *E*,*V*: The eigenvalues and eigenvectors of the covariance
  matrix, used for linear transformations of the distribution.

* *det*, *chol*, *trace*: The determinant, Cholesky root and
  the trace of the covariance matrix.

Manipulating bivarg objects
---------------------------

Computing 'distance' between objects
------------------------------------

Astrometry between image frames
===============================

Validating astrometric solutions
================================

Exporting astrometric solutions
===============================

Thanks
======

This work was funded by NSF grant #0941742. The following people contributed to the development of this package: Adam Miller, Henrik Brink, Joey Richards, Dan Starr.
