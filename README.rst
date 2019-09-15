Simplicial Mixture Models
=========================

A simplicial mixture model is a parametric probability distribution which is used to fit topological structure to data.

Prerequisites
-------------

* numpy > 1.16.2
* numba > 0.43.1

Minimal example
---------------

.. code-block:: python
  :caption: Generate data around a circle and fit a model to it

  from smm import SimplicialMM, GLEMM_Parameters, MCMC_Integrator
  from smm.helpfulfunctions import initialise_V
  import matplotlib.pyplot as plt

  % Generate some data, a noisy circle
  N = 100
  t = 2.0 * np.pi * np.random.rand(N)
  X = np.vstack([np.cos(t), np.sin(t)]).T + \
        0.1 * np.random.randn(N, 2)
  plt.scatter(X[:,0], X[:,1])

  % Setup the model with 5 feature vectors and 1-dimensional simplices
  m, n, k = 5, 2, 1
  L = SimplicialMM(m, n, k)

  % Initialise parameters for the model
  TH = GLEMM_Parameters(initialise_V(m, X),  % initial vertex positions
                        L.M,                 % number of simplices
                        covar_type='spherical',
                        covar=0.1)

  % Markov-Chain Monte-Carlo integrator
  M = MCMC_Integrator(L, TH, X)

  % Stochastic EM-algorithm to fit the model
  per_step = "CUq" * 10 + "m"
  M.perform(per_step * 100)

  % Plot points sampled from the model using the fitted parameters M.TH
  Y, _ = L.sample(M.TH, 1000)
  plt.scatter(Y[:,0], Y[:,1], c='r', s=1)
  plt.show()

Installation
------------

To install run::

  python setup.py sdist
  pip install dist/SMM-*.*.*.tar.gz

Documentation
+++++++++++++

I use Sphinx for the documentation.

On Windows in a cmd shell in the docs/ directory I run::

  make.bat html

to make the documentation.

Testing
+++++++

I use pytest for testing, run::

  pytest

from the project folder, to examine coverage I run::

  pytest --cov=smm --cov-report html smm/test/

which requires pytest-cov to be installed.
