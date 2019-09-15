r"""
Implements the a multivariate Gaussian random variable. The expected values may
be computed explicitly for this distribution.

The mathematics implemented is the following.

For a random variable U distributed as a multi-variate Gaussian with mean 0
and identity covariance matrix on :math:`\mathbb{R}^k`, we compute the
expected values of the following

.. math::
    \exp(a + b^tU - \frac12 U^tCU),

.. math::
    U\exp(a + b^tU - \frac12 U^tCU)

and

.. math::
    UU^t\exp(a + b^tU - \frac12 U^tCU).

where :math:`a` is a constant, :math:`b` is a k-vector and :math:`C` is a
symmetric positive definite k×k matrix.
With :math:`f(u) = 1, U, UU^t` respectively these are given by the integrals

.. math::
    &\int f(u)\exp(a + b^t U - \frac12 U^tCU) d\mu \\
    &=\frac{1}{(2\pi)^{k/2}} \int_{\mathbb{R}^m}
    f(u)\exp(a + b^t U - \frac12 U^t(C + I)U) dU

Now we complete the square,

* let :math:`D = (C+I)^{-1}`,
* let :math:`e = Db` and
* let :math:`f = a + \tfrac12b^tDb`

Then on the level of quadratic forms we have

.. math::
    a + b^tU - \frac12 U^t(C + I)U = f - \frac12 (U-e)^tD(U-e)

The integral above becomes

.. math::
    \int |D|^{\tfrac12} f(U)\exp(f) \frac1{(2\pi)^{\tfrac{k}2}|D|^{\tfrac12}}
    \exp(- \frac12 (U-e)^tD(U-e)) dU

This is the expected value of :math:`|D|^{\tfrac12}f(U)\exp(f)` over a
multi-variate normal distribution with mean :math:`e=(C+I)^{-1}` and
covariance matrix :math:`D=(C+I)^{-1}`.
So

* with :math:`f(U) = 1`, we find
  :math:`|C+I|^{-\tfrac12} \exp(a + \tfrac12b^tDb)`
* with :math:`f(U) = U`, we find
  :math:`|C+I|^{-\tfrac12} \exp(a + \tfrac12b^tDb)(C+I)^{-1}b`
* with :math:`f(U) = UU^t`, we find

.. math::
    |C+I|^{-\tfrac12} \exp(a + \tfrac12b^tDb)
    ((C+I)^{-1} + (C+I)^{-1}bb^t(C+I)^{-1})
"""

import numpy as np
from smm.rvs.baserv import BaseRV
from smm.helpfulfunctions import invsqrtS_logdetS_from_S


class NormalRV(BaseRV):
    r"""
    A class to generate points randomly from a multivariate Gaussian
    distribution on :math:`\mathbb{R}^m`.

    It is initialised by providing a dimension k, a mean μ in
    :math:`\mathbb{R}^m` and a linear map
    :math:`M:\mathbb{R}^k\rightarrow\mathbb{R}^m`.
    The covariance matrix is then :math:`MM^t`.
    """

    def __init__(self, mean, M):
        """
        Initialises the NormalRV class

        Parameters
        ----------
        mean : (m,) ndarray
            the mean.
        M : (k, m) ndarray
            the linear map (acting on the right).
        """
        if mean.shape[0] != M.shape[1]:
            raise ValueError("Dimension mismatch")
        k, m = M.shape
        BaseRV.__init__(self, m)
        self.k = k

        self.mean = mean
        self.M = M
        self.Moffset = mean
        self.covar = M.T.dot(M)

    def __str__(self):
        return f"N({self.mean}, {self.covar})"

    def __repr__(self):
        return f"NormalRV({self.mean}, {self.M})"

    def __eq__(self, other):
        """
        Two normal distributions are equal if they have the same ambient
        dimension, the same mean and the same covariances.
        """
        return self.m == other.m and \
            np.allclose(self.mean, other.mean) and \
            np.allclose(self.covar, other.covar)

    def __hash__(self):
        """
        Two normal distributions hash to the same value if they have the same
        means and defining matrices.
        """
        return hash(str(self.mean.tolist(), self.M.tolist()))

    def U_sample(self, d, rnd):
        r"""
        Sample from the latent space, a multi-variate Gaussian with unit
        covariance matrix.

        Parameters
        ----------
        d : int
            the number of samples.
        rnd : np.random.RandomState
            random sampling.
        """
        return rnd.standard_normal(size=(d, self.k))

    def U_to_Z(self, U):
        return U.dot(self.M) + self.Moffset

    # def sample(self, d, rnd=np.random.RandomState()):
    #     r"""
    #     Samples from the random variable, returning values in
    #     :math:`\mathbb{R}^m`.
    #
    #     Parameters
    #     ----------
    #     d : int
    #         the number of samples.
    #     rnd : np.random.RandomState, optional
    #         random sampling.
    #
    #     Returns
    #     -------
    #     sample : (d, m) ndarray
    #         an array of samples.
    #     """
    #     return self.U_to_Z(self.U_sample(d, rnd))

    def U_moments_given_Y(self, a, b, C):
        r"""
        For the latent variable U and events Y with

        .. math::
            P(Y | U) = \exp(a - b\cdot U - \frac12 U^tCt)

        calculate the following values:

        .. math::
            &\log P(Y), \\
            $\mathbb{E}_{U \mid Y} U \\
            $\mathbb{E}_{U \mid Y} UU^t.

        Parameters
        ----------
        a : (* shape1) ndarray
            array of constants,
            shape1 should be broadcastable with shape2 and shape3.
        b : (* shape2, k) ndarray
            array of linear terms,
            shape2 should be broadcastable with shape1 and shape3.
        C : (* shape3, k, k) ndarray
            array of symmetric terms,
            shape3 should be broadcastable with shape1 and shape2.

        Returns
        -------
        log_PY : (shape,) ndarray
            probability of event Y
        EU_given_Y : (shape, k) ndarray
            expected value of :math:`U` given Y
        EUUt_given_Y : (shape, k, k) ndarray
            expected value of :math:`UU^t` given Y
        """
        D = np.linalg.inv(np.eye(self.k) + C)  # broadcastable to (sh, k, k)
        e = np.sum(D * b[..., None, :], axis=-1)  # broadcastable to (sh, k)
        f = a + np.sum(e * b, axis=-1) / 2  # broadcastable to (sh,)

        log_PY = np.log(np.linalg.det(D))/2 + f  # broadcastable to (sh,)
        # E_rho = np.exp(log_expected_rho)
        EU_given_Y = e  # broadcastable to (sh, k)
        EUUt_given_Y = (D + e[..., None, :] * e[..., None])  # broadcastable
        # to (sh, k, k)

        return log_PY, EU_given_Y, EUUt_given_Y

    def log_prob_X(self, V, S, X):
        r"""
        Given an array X of values in :math:`\mathbb{R}^n`, and conditional
        probabilities

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
        """
        N, n = X.shape
        logA = n * np.log(2.0*np.pi) / 2.0

        Vmu = self.mean.dot(V)
        VM = self.M.dot(V)

        invsqrtS, logdetS = invsqrtS_logdetS_from_S(S, n)

        if len(np.shape(S)) == 2:  # full matrix case
            Y = (X - Vmu).dot(invsqrtS)
            J = VM.dot(invsqrtS)
        if len(np.shape(S)) <= 1:  # spherical or diagonal case
            Y = (X - Vmu) * invsqrtS
            J = VM * invsqrtS
        a = -logA - logdetS / 2 - 0.5 * np.sum(np.square(Y), axis=-1)
        b = Y.dot(J.T)
        C = J.dot(J.T)

        log_PX, E_UgX, E_UUtgX = \
            self.U_moments_given_Y(a, b, C)

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

        E_ZgX = self.mean + E_UgX.dot(self.M)

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
        qZZj: (m,m) ndarray
            total of :math:`ZZt` from this distribution given
            :math:`X`.
        qZXj: (m,n) ndarray
            total of :math:`ZX^t` from this distribution given
            :math:`X`.
        """

        E_UgX = self.saved_E_UgX
        E_UUtgX = self.saved_E_UUtgX

        total_weight = np.sum(weights)
        U = E_UgX.T.dot(weights)
        UUt = E_UUtgX.T.dot(weights)

        # E_ZnC = self.mean + U.dot(self.M)

        E_UMmutnC = U.dot(self.M)[:, None] * self.mean

        qZZj = self.mean * self.mean[:, None] * total_weight + \
            self.M.T.dot(UUt.dot(self.M)) + \
            E_UMmutnC + E_UMmutnC.T

        UX = (E_UgX * weights[:, None]).T.dot(X)
        qZXj = self.M.T.dot(UX) + self.mean[:, None] * weights.dot(X)

        return qZZj, qZXj

    @classmethod
    def MCstep(cls, U, rnd, delta=0.1, type='gaussian'):
        r"""
        Implements a step in an AR(1) process in the latent space :math:`U`,

        .. math::
            U_{k+1} = e^{-\tfrac12\delta} U_k + \sqrt(1-e^{-\delta}) N,

        where :math:`N` is a unit Gaussian distribution.
        This is a discrete-time version of the Ornstein-Uhlenbeck process

        .. math::
            du = -\frac12udt + dW_t,

        with :math:`U_k = u(k\delta)`.  Hence it has stationary distribution
        the multi-variate Gaussian with unit variance.

        Parameters
        ----------
        U : (N, k) ndarray
            positions prior to step.
        rnd : np.random.RandomState
            random sampling.
        delta : float
            parameter determining step size.
        type : {'gaussian', 'default'}, optional
            the type of step to use.

        Returns
        -------
        newU : (N, k) ndarray
            positions after step.
        """
        if type not in ('default', 'gaussian'):
            raise ValueError(f"Unrecognised step type: {type}")
        newU = U * np.exp(-0.5 * delta)
        newU += rnd.standard_normal(size=U.shape) * np.sqrt(1 - np.exp(-delta))
        return newU


__all__ = ["NormalRV"]
