r"""
Implements the a multivariate Gaussian approximation to a uniform simplex
distribution.

The expected values may be computed explicitly for this distribution.
"""

import numpy as np
from smm.rvs.normalrv import NormalRV
from smm.rvs.basesimplexrv import BaseSimplexRV
from scipy.linalg import eigh


class NormalSimplexRV(NormalRV, BaseSimplexRV):
    r"""
    A class to generate points randomly from a normal approximation to a
    simplex distribution in :math:`\mathbb{R}^m`.

    It is initialised by providing a set of vertices, S.  The random variable
    is then a multi-variate Gaussian with mean

    .. math:: \mu_S = \frac1{|S|}\sum_{v\in S} e_v

    and variance

    .. math::
        \Sigma_S = \frac1{|S|}\sum_{v\in S}(e_v - \mu_S)(e_v - \mu_S)^t
        = \frac1{|S|}\sum_{v\in S} e_ve_v^t - \mu_S\mu_S^t

    The eigenvalues of :math:`\Sigma_S` are 0 and :math:`\frac1{|S|}` with
    multiplicities :math:`m-k` and :math:`k` respectively, where
    :math:`k=|S|-1`.
    The support of the random variable is the k-dimensional plane containing
    :math:`e_v` for :math:`v\in S`.
    """

    def __init__(self, m, S, alpha=1.0):
        """
        Initialises the NormalSimplexRV class.

        This involves computing the defining matrix for a NormalRV and
        handling inheritance from both BaseSimplexRV and NormalRV.

        Parameters
        ----------
        m : int
            the number of hidden features.
        S : list of elements in range(m)
            the vertices of the simplex.
        alpha : float
            factor to multiply covariance by.
        """

        # Note, there is a diamond of classes here, with BaseRV at the other
        # vertex.  So BaseRV.__init__ is called twice.  This could be fixed
        # by using super(), however this gets complicated because the various
        # __init__'s take different arguments.  As it is, so little is done
        # in BaseRV.__init__ that running it twice has little penalty and no
        # adverse consequences.      I hope... :-)

        BaseSimplexRV.__init__(self, m, S)
        self.k = len(self.support) - 1
        self.covar *= alpha  # Modify covariance according to scaling factor
        self.alpha = alpha

        if self.k > 0:
            v, M = eigh(self.covar, eigvals=(m-self.k, m-1))
            M *= np.sqrt(v)
        else:
            M = np.zeros((m, 0))

        NormalRV.__init__(self, self.mean, M.T)

        # Check that we didn't make a mistake
        assert np.allclose(self.M.T.dot(self.M), self.covar)

    def __str__(self):
        return BaseSimplexRV.__str__(self)

    def __repr__(self):
        return BaseSimplexRV.__repr__(self)

    def __eq__(self, other):
        """
        Two simplices are equal if they have the same ambient dimension,
        the same list of (ordered) vertices and the same scaling factor.
        """
        return BaseSimplexRV.__eq__(self, other) and self.alpha == other.alpha

    def __hash__(self):
        """
        Two simplices hash to the same value if they have the same ambient
        dimension and the same set of vertices, notice that self.S is sorted
        in __init__.
        """
        return hash((self.m, self.S, self.alpha))


__all__ = ["NormalSimplexRV"]
