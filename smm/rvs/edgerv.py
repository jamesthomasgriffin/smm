r"""
Implements a uniform distribution on an edge and a triangular distribution
on an edge.

The expected values of Z given X may be computed explicitly for these
distributions.

Some background mathematics:

* The moment generating function of a truncated normal distribution

  ..math ::
    e^{\mu t + \sigma^2 t^2 / 2} \left[
    \frac{\Phi(\alpha-\sigma t) - \Phi(\beta - \sigma t)}{
    \Phi(\beta) - \Phi(\alpha)}
    \right]

  where

  * :math:`\mu` is the mean of the untruncated normal distribution,
  * :math:`\sigma` is the standard deviation,
  * :math:`(\alpha,\beta)` are the bounds of the distribution.

"""

import numpy as np
from smm.rvs.uniformsimplexrv import UniformSimplexRV
from smm.rvs.basesimplexrv import BaseSimplexRV
from smm.helpfulfunctions import invsqrtS_logdetS_from_S
from scipy.stats import norm
Phi = norm.cdf


def phi(x):
    return np.exp(-0.5 * np.square(x)) / (np.sqrt(2*np.pi))


class EdgeRV(UniformSimplexRV):
    r"""
    A class to generate points uniformly from an edge in :math:`\mathbb{R}^m`.
    """

    def __init__(self, m, S, alpha=1.0):
        """
        Parameters
        ----------
        m : int
            the number of hidden features.
        S : pair of elements in range(m)
            the vertices of the simplex.
        """
        if not len(S) == 2:
            raise ValueError("Must specify a pair of vertices")

        UniformSimplexRV.__init__(self, m, S)

    def __str__(self):
        return BaseSimplexRV.__str__(self)

    def __repr__(self):
        return BaseSimplexRV.__repr__(self)

    def U_sample(self, d, rnd=np.random.RandomState()):
        return np.random.rand(d, 1)

    def U_moments_given_Y(self, a, b, c):
        r"""
        For the latent variable U and function f with

        .. math::
            f(U) = \exp(a + b U - \frac12 cU^2)

        calculate the following values:

        .. math::
            &\log E = \log\mathbb{E}f(U), \\
            &F = \frac{1}{E}\mathbb{E}_{U \mid Y} f(U)U \\
            &G = \frac{1}{E}\mathbb{E}_{U \mid Y} f(U)UU^t.

        Note that F is the expected value of U with the distribution
        proportional to f, while G is the second moment. This distribution
        is a truncated normal distribution.

        Parameters
        ----------
        a : (* shape1) ndarray
            array of constants,
            shape1 should be broadcastable with shape2 and shape3.
        b : (* shape2, 1) ndarray
            array of linear terms,
            shape2 should be broadcastable with shape1 and shape3.
        C : (* shape3, 1, 1) ndarray
            array of quadratic terms,
            shape3 should be broadcastable with shape1 and shape2.

        Returns
        -------
        logE : (shape,) ndarray
            see above
        F : (shape, 1) ndarray
            see above
        G : (shape, 1, 1) ndarray
            see above
        """
        σ2 = 1 / c.reshape(c.shape[:-2])  # strip out the two ones
        σ = np.sqrt(σ2)
        μ = b.reshape(b.shape[:-1]) * σ2
        α = -μ / σ
        β = 1 / σ + α
        Z = Phi(β) - Phi(α)
        Zthreshold = 1e-12
        mask = Z < Zthreshold
        Z[mask] = Zthreshold

        non_Z_part = a + μ**2 / (2*σ2) + 1 / 2 * np.log(2*np.pi) + np.log(σ)
        logE = non_Z_part + np.log(Z)

        EU = μ + σ * (phi(α) - phi(β)) / Z

        EUUt = σ2 * (1 + (α*phi(α) - β*phi(β)) / Z) + 2*μ*EU - μ**2

        # If Z is small then use endpoints (should find better approximation)
        EU[mask] = np.clip(μ[mask], 0, 1)
        EUUt[mask] = EU[mask]

        return logE, EU[:, None], EUUt[:, None, None]

    def pullback_quad_form(self, V, S, X):
        r"""
        For each row x of X, one has a quadratic form on u

        .. math::
            \log \rho_S(VMu + Vm - x)
            = a + b u - \frac{1}{2} cu^2

        where :math:`\rho_S` is the density function of the multivariate normal
        distribution with mean 0 and covariance matrix S.  The formulae for the
        constants are

        .. math::
            a &= -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log\text{det} S
              -\frac{1}{2}(Vm - x)^tS^{-1}(Vm - x) \\
            b &= -(VM)^t S^{-1} (Vm - x) \\
            c &= (VM)^tS^{-1}VM
        """
        N, n = X.shape
        logA = n / 2 * np.log(2.0*np.pi)

        Vm = self.Moffset.dot(V)
        VM = self.M.dot(V)

        invsqrtS, logdetS = invsqrtS_logdetS_from_S(S, n)

        if len(np.shape(S)) == 2:  # full matrix case
            Y = (X - Vm).dot(invsqrtS)
            J = VM.dot(invsqrtS)
        if len(np.shape(S)) <= 1:  # spherical or diagonal case
            Y = (X - Vm) * invsqrtS
            J = VM * invsqrtS
        a = -logA - logdetS / 2 - 0.5 * np.sum(np.square(Y), axis=-1)
        b = Y.dot(J.T)
        c = J.dot(J.T)
        return a, b, c.reshape((1, 1, 1))

    def log_prob_X(self, V, S, X):
        r"""
        Given an array X of values in :math:`\mathbb{R}^n`, and conditional
        probabilities

        .. math:: P(X\mid Z) = \rho_S(X-VZ),

        where :math:`\rho_S` is the multivariate normal distribution with
        mean 0 and covariance :math:`S`, calculate the

        * logarithm of the probability density of each row of :math:`X`.

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
        """
        quad_forms = self.pullback_quad_form(V, S, X)

        log_PX, E_UgX, E_UUtgX = self.U_moments_given_Y(*quad_forms)

        self.saved_E_UgX = E_UgX  # broadcastable to (N, k)
        self.saved_E_UUtgX = E_UUtgX  # broadcastable to (N, k, k)

        return log_PX

    def mean_given_X(self):
        r"""
        Given probabilities for X of values in :math:`\mathbb{R}^n`,
        calculate the mean of :math:`Z` given :math:`X` using values
        pre-calculated by log_prob_X.

        Parameters
        ----------
        None

        Returns
        -------
        E_ZgX: (N,m) ndarray
            total of :math:`Z` from this distribution given
            :math:`X`.
        """

        E_UgX = self.saved_E_UgX  # broadcastable to (N, k)

        E_ZgX = self.Moffset + E_UgX.dot(self.M)

        return E_ZgX

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
        qZZj: (N,m,m) ndarray
            total of :math:`ZZt` from this distribution given
            :math:`X`.
        qZXj: (N,m,n) ndarray
            total of :math:`ZX^t` from this distribution given
            :math:`X`.
        """

        UgX = self.saved_E_UgX
        UUgX = self.saved_E_UUtgX

        total_weight = np.sum(weights)
        U = UgX.T.dot(weights)
        UU = UUgX.T.dot(weights)

        # NB Z = MU + m
        M, m = self.M, self.Moffset

        UMm = U.dot(M)[:, None] * m

        ZZj = m * m[:, None] * total_weight + M.T.dot(UU.dot(M)) + UMm + UMm.T

        UX = (UgX * weights[:, None]).T.dot(X)
        ZXj = M.T.dot(UX) + m[:, None] * weights.dot(X)

        return ZZj, ZXj


__all__ = ["EdgeRV"]
