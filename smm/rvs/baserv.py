r"""
Implements a base class for random variables on :math:`\mathbb{R}^m`.
"""

import numpy as np
from scipy.special import logsumexp


class BaseRV:
    r"""
    A base class for random variables on :math:`\mathbb{R}^m` to inherit from.
    """

    def __init__(self, m):
        """
        Initialises the BaseRV class

        Parameters
        ----------
        m : int
            the number of hidden features.
        """
        self.m = m

    def __repr__(self):
        return "BaseRV({})".format(self.m)

    def U_sample(self, d, rnd=np.random.RandomState()):
        raise NotImplementedError()

    def U_to_Z(self, U):
        raise NotImplementedError()

    def sample(self, d, rnd=np.random.RandomState()):
        r"""
        Samples from the random variable, returning values in
        :math:`\mathbb{R}^m`.

        Parameters
        ----------
        d : int
            the number of samples.
        rnd : np.random.RandomState, optional
            random samples.

        Returns
        -------
        sample : (d, m) ndarray
            an array of samples.
        rnd : np.random.RandomState, optional
            random samples.
        """
        return self.U_to_Z(self.U_sample(d, rnd))

    def estimated_mean_and_covariance(self, n_samples=1000,
                                      rnd=np.random.RandomState()):
        """
        Estimates the mean and variance using samples from the distribution.
        Useful for testing.

        Parameters
        ----------
        n_samples : int, optional
            number of samples with which to compute the estimators.
        rnd : np.random.RandomState, optional
            random sampling.

        Returns
        -------
        mean : (m,) ndarray
            the estimated mean.
        covar: (m, m) ndarray
            the estimated covariance matrix using the unbiased estimator.
        """
        assert n_samples > 1, "Not enough samples for covariance estimator"
        Z = self.sample(n_samples, rnd)
        mean = np.mean(Z, axis=0)
        Z -= mean
        covar = Z.T.dot(Z) / (n_samples - 1)
        return mean, covar

    def estimated_expected_moments_given_X(self, V, S, X,
                                           n_samples=1000,
                                           rnd=np.random.RandomState()):
        r"""
        Given an array X of values in :math:`\mathbb{R}^n`, and conditional
        probabilities

        .. math:: P(X\mid Z) = \rho_S(X-VZ),

        where :math:`\rho_S` is the multivariate normal distribution with
        mean 0 and covariance :math:`S`, estimate the

        * logarithm of the probability of :math:`X`,
        * the expected value of :math:`Z` given :math:`X`,
        * the expected value of :math:`ZZ^t` given :math:`X`.

        This function uses a given number of samples to estimate these
        expected values by drawing samples from :math:`Z`.

        .. warning::
            This function is mainly for testing, it is not designed to be
            efficient or numerically stable.

        Parameters
        ----------
        V : (m, n) ndarray
            linear map :math:`\mathbb{R}^m \rightarrow\mathbb{R}^n`
        S : {(n, n) ndarray, (n,) ndarray, float}
            covariance in either matrix form, diagonal form, or spherical form.
        X : (N, n) ndarray
            data
        n_samples : int, optional
            the number of samples to use.
        rnd : np.random.RandomState, optional
            random sampling.

        Returns
        -------
        log_PX : (N,) ndarray
            logarithm of probability of :math:`X`.
        E_ZgX: (N,m) ndarray
            expected values of :math:`Z` given :math:`X`.
        E_ZZtgX: (N,m,m) ndarray
            expected values of :math:`ZZ^t` given :math:`X`.
        """
        Z = self.sample(n_samples, rnd)
        VZ = Z.dot(V)
        N, n = X.shape

        logA = n * np.log(2.0*np.pi)

        if len(np.shape(S)) == 2:  # full matrix case
            sqrtS = np.linalg.cholesky(S)
            invsqrtS = np.linalg.inv(sqrtS)
            Y = (X[:, None, :] - VZ[None, :, :]).dot(invsqrtS)
            logdetS = np.sum(np.log(np.diag(sqrtS)))
        if len(np.shape(S)) <= 1:  # spherical or diagonal case
            invsqrtS = 1. / np.sqrt(S)
            Y = (X[:, None, :] - VZ[None, :, :]) * invsqrtS
            if len(np.shape(S)) == 0:  # spherical case
                logdetS = n * np.log(S)
            else:  # diagonal case
                logdetS = np.sum(np.log(S))

        log_PXgZ = -0.5 * np.sum(np.square(Y), axis=-1) \
            - 0.5 * (logA + logdetS)
        PXgZ = np.exp(log_PXgZ)
        log_PX = logsumexp(log_PXgZ, axis=-1) - np.log(n_samples)
        PX = np.exp(log_PX)
        E_ZgX = np.mean(PXgZ[:, :, None] * Z[None, :, :], axis=1) / PX[:, None]
        E_ZZtgX = np.mean(
            PXgZ[:, :, None, None] * Z[None, :, None, :] * Z[None, :, :, None],
            axis=1) / PX[:, None, None]
        return log_PX, E_ZgX, E_ZZtgX

    def rate_fn_bound(self, D, V=None, gaussian_approx=True):
        r"""
        This function computes an upper bound of the rate distortion function
        :math:`R(D)` for the random variable, where :math:`D` defines a
        quadratic distortion function,

        ..math:
            x\mapsto x^t D^{-1} x.

        The variable :math:`D` can be a symmetric matrix, the diagonal of a
        diagonal matrix, or a scalar defining a spherical distortion function.
        Which case is determined by the shape of :math:`D`.

        The upper bound is calculated by replacing the random variable with a
        multi-variate normal distribution with the same mean and covariance
        matrix.

        Parameters
        ==========
        D : {float, (n,) array, (n, n) array}
            defines the distortion function.
        gaussian_approx : bool, optional
            determines whether to use a Gaussian distribution of the RV.
        V : (m, n) array, optional
            defines a linear map embedding the random variable.

        Returns
        =======
        R : float
            the bound of the rate for the given distortion, in nats.
        """
        if not gaussian_approx:
            raise NotImplementedError("No exact rate distortion function.")
            
        # First compute the covariance matrix
        if V is None:
            Sigma = self.covar
        else:
            m, n = V.shape
            if m != self.m:
                raise ValueError("Dimension mismatch")
            Sigma = V.T.dot(self.covar).dot(V)

        # Now find coordinate change so that S = I
        if len(np.shape(D)) == 2:  # full covariance
            sqrtS = np.linalg.cholesky(D)
            invsqrtS = np.linalg.inv(sqrtS)
        else:
            invsqrtS = 1 / np.sqrt(D) * np.eye(Sigma.shape[0])

        # Transform the covariance matrix
        Sigma = invsqrtS.T.dot(Sigma).dot(invsqrtS)

        # Compute the (real) eigenvalues
        variances = np.linalg.eigvals(Sigma).real

        return rate_dist_ind_Gaussians(1, variances)


def rate_dist_ind_Gaussians(D, variances):
    """
    Computes the rate distortion function for the standard sum of
    squares distortion function and distribution given by independent
    samples from a list of Gaussians with the given variances.
    """
    variances = np.sort(variances[variances > 1e-10])
    n = len(variances)
    cum_sum = 0
    for i in range(n):
        phi = cum_sum + (n-i)*variances[i]
        if D < phi:
            lmda = (D - cum_sum) / (n - i)
            break
        cum_sum += variances[i]
    else:
        lmda = np.inf

    Di = variances.copy()
    Di[Di > lmda] = lmda
    return 1/2 * np.log(variances / Di).sum()


__all__ = ["BaseRV"]
