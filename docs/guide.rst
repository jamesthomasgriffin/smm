What is a Simplicial Mixture Model?
===================================

Introduction
------------

* A simplicial mixture model is a probability distribution defined on a Euclidean
  space by a mixture of simplices.
* Each simplex is the convex hull of a set of feature vectors.
* The set of feature vectors will typically be small and the number of potential
  simplices large.
  So each feature vector influences the embedding of many simplices.

There are two viewpoints of Simplicial Mixture Models, arising from
two different interpretations of a simplex:

1. *(Compositional)* A simplex, a set of non-negative numbers summing to one,
   parametrises the possible compositions of a quantity into sub-quantities.
2. *(Geometric)* A simplex is the convex hull of its vertices, and may be used
   as a building block of a simplicial space.

Compositional Viewpoint
+++++++++++++++++++++++

From the compositional viewpoint the feature vectors are the most important
part of the model.
A simplicial mixture model consists of a set of archetype feature vectors,
or *endmembers*, along with a latent variable controlling the random distribution
of mixtures of these feature vectors.
This latent variable is a mixture of Dirichlet distributions, one for each
simplex.

Geometric Viewpoint
+++++++++++++++++++

From the geometric viewpoint the simplices define the topology of a space
and are importantly independent from the feature vectors.
The feature vectors are points within a space and the simplices fill in the
gaps between them.


Example - Composition of colours in a painting
++++++++++++++++++++++++++++++++++++++++++++++
A child draws a picture with :math:`m` crayons; the resulting colours
correspond to the child's choice of crayons.
Now suppose an artist wishes to paint to a canvas and she has a palette of
:math:`m` paints.
When painting the artist could apply one of these paints directly to the canvas.
However in practice she mixes a small number together to obtain the desired
colour and then applies this blend to the canvas.

In the child's drawing there is a discrete set of colours, this could be
represented by a Gaussian mixture model, the components are the choice of
crayon and the mean vectors the colour of that crayon.
In the artist's painting there is a continuum of colour which would be
represented by a **simplicial mixture model**, now there is a feature vector
for each pigment in the palette and each simplex describes a combination
blended together.

When we observe either finished piece, the drawing or painting, our eyes see
the colours.
A digital image would be represented by an RGB triple for each point.
However there is a hidden (latent) higher dimensional image, where each point is instead represented by an
:math:`m`-dimensional vector, representing the choice of crayon or the
quantity of pigment.
This hidden image represents the true constituents of the piece, while the
regular image records what it looks like.
Utilising a Gaussian mixture model corresponds to finding the set of crayons and
then inferring the hidden image from the model.
Similarly a simplicial mixture model could be used to find the artist's palette
and then infer the hidden ratio of pigments at each point.

Example - Geometry of a hand-written digit
++++++++++++++++++++++++++++++++++++++++++
The feature vectors pick out the important points of the digit; perhaps the end
of each stroke, along with any corners.
Those important points are joined by lines, following the strokes of the pen.
These lines are the simplices, they fill in the gaps between feature vectors.
The process of fitting a model to a digit is a form of vectorisation, a raster
image can be interpreted as a histogram of a random variable, then fitting a
model gives vector information representing this random variable.

---------------------------------------------------------------------

Theoretical Details
-------------------

For full details see the preprint

    James T. Griffin,
    *Probabilistic Fitting of Topological Structure to Data*,
    `arXiv:1909.10047 <https://arxiv.org/abs/1909.10047>`_


Reminder - Gaussian Mixture Models
++++++++++++++++++++++++++++++++++
To understand the details behind a simplicial mixture model (SMM) I will first
remind the reader about Gaussian mixture models (GMMs).
SMMs are a generalisation in the sense that if one chooses the simplest possible
set of parameters the resulting SMM is just a GMM, however the essential idea
behind SMMs is a little different.

A GMM defines a random variable, perhaps multivariate.
We will fix a dimension, n describing the ambient dimension, and an integer, m
describing the number of components of the model.
One then needs a choice of parameters

.. math::
  \Theta = ((p_1,\ldots,p_m),
  (\mu_1,\ldots,\mu_m),
  (\Sigma_1,\ldots,\Sigma_m))

where

* :math:`(p_1,\ldots,p_m)`, the *mixing probabilities* are non-negative real
  numbers which sum to 1,
* :math:`(\mu_1,\ldots,\mu_m)`, the *means* are n-dimensional vectors, and
* :math:`(\Sigma_1,\ldots,\Sigma_m)`, the *covariances* are n×n positive
  definite symmetric matrices.

To pick a sample from the GMM, first choose a component k from
:math:`\{1,\ldots,m\}` at random according to the mixing probabilities, then
choose a point p from the multi-variate normal distribution
:math:`N(\mu_k,\Sigma_k)` with mean :math:`\mu_k` and covariance matrix
:math:`\Sigma_k`.
This p is the random sample.

The problem of fitting a GMM to a data set X of n-dimensional vectors involves
finding a set of parameters which best fits that cloud.
Formally, we wish to find the parameters which maximise the log-likelihood
function

.. math::
  \log L(\Theta; X) = \frac1N\sum_{i=1}^N \log P(x_i \mid \Theta),

where N is the number of points in the data set X and :math:`P(x_i \mid \Theta)`
is the probability of obtaining the point :math:`x_i` given the set of
parameters :math:`\Theta`.

In practice this maximisation problem is too difficult to solve directly
and instead an iterative approach, the *EM algorithm* is used.
More on that later.

Simplices
+++++++++
A probability simplex with m vertices is the set of all probability
distributions on m elements,
i.e. the set of n non-negative real numbers which sum to 1.
For m=2, the probability simplex is a line segment, for m=3 a triangle, for m=4
a tetrahedron, and so on...

A point :math:`(p_1,\ldots,p_m)` on a probability simplex can be used to form
a weighted average of n vectors :math:`v_1,\ldots,v_m` giving

.. math:: p_1v_1 + \cdots + p_m v_m

Every point in the convex hull of :math:`v_1,\ldots,v_m` may be expressed in
this way.

.. _sample-from-simplex:

Uniform distribution on a simplex
+++++++++++++++++++++++++++++++++
One can choose a point on a simplex at random using the uniform distribution
as follows, pick m numbers :math:`x_1,\ldots,x_m` independently and uniformly
from the numbers between 0 and 1.
Then sort these numbers to obtain :math:`0<y_1<\cdots<y_m<1`.
Now let :math:`p_k = y_{k+1} - y_{k}` for :math:`k=0, 1,\ldots,m`, where
:math:`y_0=0` and :math:`y_{m+1}=1`.
By construction these are all positive numbers that add up to 1, the
distribution sampled in this way is uniform.

Alternatively one could choose m+1 independent samples :math:`z_0,\ldots,z_m`
from an exponential distribution, then compute :math:`z` their sum, now let
:math:`p_k = z_k / z`.
Again each number is positive, they sum to 1 and the distribution is uniform.


Face and degenerate simplices
+++++++++++++++++++++++++++++
A **face** of a simplex on m vectices may be specified by a non-empty subset S
of the vertices, the points :math:`(p_1,\ldots,p_m)` of the face must satisfy
:math:`p_i = 0` if i is not contained in the subset S.
For example if m=3 the whole simplex is a triangle and there are 7 faces:

* :math:`\{1\}, \{2\}, \{3\}` define the vertices of the triangle,
* :math:`\{1,2\}, \{2,3\}, \{1,3\}` define the edges of the triangle,
* :math:`\{1, 2, 3\}` is the whole of the triangle.

Each of these faces is itself a simplex, but perhaps with fewer vectices.

However importantly there are also degeneracies of a simplex.
Recall that a *multi-set* is like a set which is defined by its elements,
however each element may have multiplicity.
A **degeneracy** of the simplex :math:`\{1,\ldots,m\}` is a multi-set which
contains every element 1 to n, but which may contain multiple copies of each
vertex.

The points of a degeneracy are defined to be a tuple of non-negative real
numbers which sum to 1, with one number for each element of the multi-set.
For example :math:`\{1,2,2,3\}` defines a degenerate simplex with points
defined by four numbers, perhaps :math:`(p_1, p_{12}, p_{22}, p_3)`.
This should be viewed as a regular simplex with 4 vertices, but where one of
the vertices is repeated twice.
A point of the original simplex may be recovered by summing over all the
repeated points.
Given vectors :math:`v_1,v_2,v_3` this still defines a point in the
convex hull, :math:`p_1v_1 + p_{12}v_2 + p_{22}v_2 + p_3v_3`.

The importance of degeneracies is that they allow for natural non-uniform
distributions on the original simplex.
i.e. choose a point uniformly from the degenerate simplex and then map it
down to the original simplex.
These distributions are
`Dirichlet distributions <https://en.wikipedia.org/Dirichlet_distribution>`_
with integer parameters.


Simplicial Mixture Models
+++++++++++++++++++++++++
Fix an ambient dimension n and an integer m for the number of features of our
model.
Now choose a finite set A of simplices with vertices from
:math:`\{1,\ldots,m\}`, these simplices may be degenerate
Label the simplices :math:`S_1,\ldots,S_M`, so M is the total number of
simplices.
These are the fixed parameters of our model.

For the variable parameters we have

* :math:`(p_1,\ldots,p_M)`, the *mixing probabilities* are non-negative real
  numbers which sum to 1,
* :math:`(v_1,\ldots,v_m)`, the *feature vectors* are n-dimensional vectors, and
* :math:`(\Sigma_1,\ldots,\Sigma_M)`, the *covariances* are n×n positive
  definite symmetric matrices.

Note that there is a mixing probability and a covariance matrix for each simplex
and a feature vector for each vertex.

Sampling from the SMM proceeds as follows,

* choose a simplex :math:`S_j` at random according to the mixing probabilities,
* choose a point z at random from :math:`S_j` and use it to form a weighted sum
  of the feature vectors, denoted :math:`V(z)`.
* finally add a point :math:`\zeta` drawn from the normal distribution
  :math:`N(0, \Sigma_j)` to :math:`V(z)`.

The point :math:`V(z)` is the sample from the SMM, and we say that
:math:`V(z)+\zeta` is drawn from the corresponding Gaussian simplicial mixture
model (GSMM).

If :math:`A=\{(1), (2), \ldots, (m)\}` is the set of 0-simplices (or points),
then the SMM is a choice of column of :math:`V`, hence the SMM is a GMM with
means the columns of :math:`V`.


Linearly Embedded Mixture Models
++++++++++++++++++++++++++++++++
To understand the mathematics of fitting a simplicial mixture model to a dataset
it helps to abstract the model further.
Let :math:`U_1,\ldots,U_M` be a list of known random variables on
:math:`\mathbb{R}^m`.
Let :math:`n` be the dimension of points in the dataset.
The variable parameters consist of a choice

.. math::
  \theta = ((p_1,\ldots,p_M), V
  (\Sigma_1,\ldots,\Sigma_M)),

where

* :math:`(p_1,\ldots,p_M)` describes a categorical distribution,
* :math:`V` is a linear map from :math:`\mathbb{R}^{m}` to
  :math:`\mathbb{R}^{n}`,
* :math:`(\Sigma_1,\ldots,\Sigma_M)` are covariance matrices used to define a
  Gaussian kernel for each distribution.

Let the latent variable :math:`C` correspond to the choice of the M
distributions according to the probabilities.
Let the latent variable :math:`Z = U_C` be the variable sampled by choosing a
distribution :math:`U_c` according to :math:`C`, then sampling from :math:`U_c`.
The linear map :math:`V` is represented by an n×m matrix.
The latent variable :math:`Z` is a standard mixture model, the linearly
embedded mixture model is given by applying the linear map to get :math:`V(Z)`.
The linearly embedded mixture model (LEMM) is a random variable on
:math:`\mathbb{R}^n` defined by choosing a distribution :math:`U_C` at random
using the latent variable :math:`C`, then choosing :math:`z` from :math:`S_C`.
Finally the Gaussian LEMM (GLEMM) is the variable

.. math::
  V(Z) + N(0, \Sigma_{S_C}).

A simplicial mixture model is a particular example of a linearly embedded
mixture model when the list of random variables on :math:`\mathbb{R}^m`
arises from a list of simplices.
