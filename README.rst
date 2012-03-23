***********************************
pyBA: Bayesian Astrometry in Python
***********************************

:Date: March 23, 2012
:Version: 0.1
:Authors: Berian James
:Web site: https://github.com/berianjames/pyBA
:Copyright: This document has been placed in the public domain.
:License: This code is released under the MIT license.

=========================
pYBA: Bayesian Astrometry
=========================

pyBA is a Python implementation of the Bayesian Astrometry 
framework. It provides a module for handling probability
distribution that represent astronomical objects and for 
analysing the changes to these distributions between images.

Typical use might look like::

    #!/usr/bin/env python
    
    from pyBA import bivarg
    # ... more basic example code here

It provides:

* Classes for respresenting astronomical objects
  and astrometric mappings as probability distributions

* Maximum likelihood and multivariate normal likelihood
  approximation routines.

* MCMC likelihood computation (using emcee)

It aspires to (but does not yet) provide:

* A full gaussian process astrometic analysis
  (perhaps using scikits.learn)

* An interface with wcslib and pyfits

* A helpful set of examples

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

This work was funded by NSF grant #...
The following people contributed: ...
