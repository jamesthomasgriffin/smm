"""
Implements the random variable associated to a single point.
"""

import numpy as np
from smm.rvs.baserv import BaseRV
from smm.helpfulfunctions import invsqrtS_logdetS_from_S


class PointRV(BaseRV):
    r"""
    A class to represent a single-valued distribution taking the value of a
    canonical basis vector in :math:`\mathbb{R}^m`.
    """

    def __init__(self, m, v):
        """
        Initialises the PointRV class

        Parameters
        ----------
        m : int
            the number of hidden features.
        v : int
            the vertex, a number in range(m).
        """
        BaseRV.__init__(self, m)
        if type(v) is int:
            if v not in range(m):
                raise ValueError(f"Vertex {v} not in range")
            self.Z = np.zeros((m,))
            self.Z[v] = 1.0
        else:
            if v.shape != (m,):
                raise ValueError("Vector of wrong dimension")
            self.Z = v

        self.mean = self.Z
        self.covar = np.zeros((m, m))
        self.k = 0

    def __str__(self):
        return str(self.Z)

    def __repr__(self):
        return "PointRV({}, {})".format(self.m, self.Z)

    def __eq__(self, other):
        """
        Two simplices are equal if they have the same ambient dimension and
        the same set of vertices, notice that self.S is sorted in __init__.
        """
        return self.m == other.m and (self.Z == other.Z).all()

    def __hash__(self):
        """
        Two simplices hash to the same value if they have the same ambient
        dimension and the same set of vertices, notice that self.S is sorted
        in __init__.
        """
        return hash((self.m, self.Z))

    def sample(self, d, rnd=None):
        """
        Sampling from such a one-point distribution is not very exciting,
        but here it is.

        Parameters
        ----------
        d : int
            the number of samples.
        rnd : None
            this is not used.
        Returns
        -------
        sample : (d, m) ndarray
            an array of samples.
        """
        return self.Z[None, :] * np.ones((d, 1))

    def log_prob_X(self, V, S, X):
        r"""
        Given an array X of values in :math:`\mathbb{R}^n`, and with
        conditional probabilities

        .. math:: P(X\mid Z) = \rho_S(X-VZ),

        where :math:`\rho_S` is the multivariate normal distribution with
        mean 0 and covariance :math:`S`, calculate the

        * logarithm of the probability of :math:`X`,

        Parameters
        ----------
        V : (m, n) ndarray
            linear map :math:`\mathbb{R}^m \rightarrow\mathbb{R}^n`
        S : {(n, n) ndarray, (n,) ndarray, float}
            covariance in either matrix form, diagonal form, or spherical form.
        X : (N, n) ndarray
            data

        Returns
        -------
        log_PX : (N,) ndarray
            logarithm of probability of :math:`X`.

        Note
        ----
        Since Z and ZZt are constant, we only return one value, not
        one for each X, the results are broadcastable to the
        expected shapes of (N,m) and (N,m,m) respectively.
        """
        N, n = X.shape
        logA = n * np.log(2.0*np.pi) / 2.0
        VZ = self.Z.dot(V)

        invsqrtS, logdetS = invsqrtS_logdetS_from_S(S, n)

        if len(np.shape(S)) == 2:
            Y = (X - VZ[None, :]).dot(invsqrtS)
        else:
            Y = (X - VZ[None, :]) * invsqrtS

        log_PX = -0.5 * np.sum(np.square(Y), axis=1) - (logA + logdetS/2)
        return log_PX

    def moments_marg_over_X(self, weights, X):
        r"""
        Given weights for the array X of values in :math:`\mathbb{R}^n`,
        calculate

        * the expected value of :math:`Z` given :math:`X`,
        * the expected value of :math:`ZZ^t` given :math:`X`.

        using values pre-calculated by log_prob_X.

        Parameters
        ----------
        weights : (N,) ndarray
            weights for the data points.
        X : (N, n) ndarray
            the data.

        Returns
        -------
        qZZj: (m,m) ndarray
            total of :math:`ZZt` from this distribution given
            :math:`X`.
        qZXj: (m,n) ndarray
            total of :math:`ZX^t` from this distribution given
            :math:`X`.
        """
        weighted_X = weights.dot(X)
        total_weight = weights.sum()
        return self.Z[:, None] * weighted_X[None, :], \
            (total_weight * self.Z)[:, None] * self.Z[None, :]

    def mean_given_X(self):
        """
        The expected value of Z over the conditional distribution on Z given X.

        Parameters
        ==========
        None

        Returns
        =======
        Z : (1, m) ndarray
            expected value of Z.
        """
        return self.Z[None, :]

    def diff_entropy_U(self):
        return 0

    def diff_entropy_Z(self):
        return 0


__all__ = ["PointRV"]
