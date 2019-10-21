Layout of the code
==================

There are 4 types of classes used in the module smm.

Model and Parameters
--------------------

The two classes defining a model are

* a *mixture model* class, typically derived from :ref:`LinearlyEmbeddedMM`, but you
  are likely to want to use :ref:`SimplicialMM` or :ref:`GraphMM`.  This class holds a list
  of random variables and other 'hyperparameters'.  The class implements a
  methods that given parameters will

    - compute mean, covariance etc.,
    - sample from the model,
    - carry out the exact EM algorithm, if the random variables support this.

* a *parameter* class, which holds the parameters defining the mixture
  weights and the particular embedding, the main class you will need is
  :ref:`GLEMM_Parameters`, which also has a covariance parameter.  The methods of
  this class are primarily used to define new parameters maximising a given
  function.

Individual Random Variables
---------------------------

There are also *random variable* classes which define the component random
variables, but if you are using :ref:`SimplicialMM` or :ref:`GraphMM` then these classes
will be initialised for you.  These classes are contained in the namespace
smm.rv, the main example is :ref:`UniformSimplexRV`.

Markov chain Monte Carlo Integrator
-----------------------------------

Finally there is a class which handles Markov chain Monte Carlo sampling,
:ref:`MCMC_Integrator`.  This defines a Markov chain for each data point which
'walks' through the latent variables according to a conditional distribution.
This class is required to carry out the stochastic EM algorithm for mixture
models where the exact EM algorithm is unavailable.

The details of the random walk steps need to be specified by hand via a
'regime', which is a word in the letters 'U', 'C', 'q' and 'm'.  See the
examples for details.
