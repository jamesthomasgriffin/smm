smm.rvs - Random Variables
==========================

.. contents::

Simplex Random Variables
------------------------

These are classes that are used in simplicial mixture models.
The most general is the uniform simplex, however this can not be used for the EM algorithm without estimating values using Markov chain methods.
The EdgeRV implements an equivalent distribution to a 1-dimensional UniformSimplexRV, but can be used in the EM algorithm.
The NormalSimplexRV class implements a Gaussian approximation to the distribution used in UniformSimplexRV, which can be used in the EM algorithm.

.. _UniformSimplexRV:

UniformSimplexRV
++++++++++++++++

.. autoclass:: smm.rvs.UniformSimplexRV
    :members:

NormalSimplexRV
+++++++++++++++

.. autoclass:: smm.rvs.NormalSimplexRV
    :members:

EdgeRV
++++++

.. autoclass:: smm.rvs.EdgeRV
    :members:

Base Classes
------------

BaseRV
++++++

.. autoclass:: smm.rvs.BaseRV
    :members:

BaseSimplexRV
+++++++++++++

.. autoclass:: smm.rvs.BaseSimplexRV
    :members:

Classical Distributions
-----------------------

NormalRV
++++++++

.. autoclass:: smm.rvs.NormalRV
    :members:

PointRV
+++++++

.. autoclass:: smm.rvs.PointRV
    :members:

Other Distributions
-------------------

These are more less commonly used distributions, but are included to help users define their own RV classes.

QuadraticBezierRV
+++++++++++++++++

.. autoclass:: smm.rvs.QuadraticBezierRV
    :members:

CubicBezierRV
+++++++++++++

.. autoclass:: smm.rvs.CubicBezierRV
    :members:
