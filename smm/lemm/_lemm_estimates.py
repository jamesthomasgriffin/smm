import numpy as np
from scipy.special import logsumexp


class LEMMEstimatesMixin:
    """
    A mixin class implementing various Monte Carlo estimates for when exact
    formulae are unavailable.
    """

    def log_X_given_Z_and_C(self, X, Z, indices, TH):
        """
        Logarithm of probability of X with for a given Z and the
        distribution it arose from (indicated by the indices list).

        Parameters
        ----------
        X : (N, D) ndarray
            the points at which we are calculating the conditional
            probabilities
        Z : (T, m) ndarray
            samples from Z
        indices : list of slices of length M
            defining a partition of {1, ..., T} into M subsets, giving the
            samples from C
        TH : dict
            LEMM parameters

        Returns
        -------
        log_XgZ : (N, T) ndarray

        See Also
        --------
        LinearlyEmbeddedMM.log_X_given_VZ_and_C

        """
        VZ = Z.dot(TH.V)
        return self.log_X_given_VZ(X, VZ, indices, TH)

    def log_X_given_VZ_and_C(self, X, VZ, indices, TH):
        r"""
        Logarithm of probability of X with for a given V(Z) and the
        distribution it arose from (indicated by the indices list).

        .. math::

            \log P(X | VZ, C) = - \frac{n}{2}\log(2\pi)
            - \frac12\log\det\Sigma_C
            -\frac12 (X-VZ)^t\Sigma_C^{-1}(X-VZ)

        Parameters
        ----------
        X : (N, n) ndarray
            the points at which we are calculating the conditional
            probabilities
        VZ : (T, n) ndarray
            samples from Z already transformed by TH.V
        indices : list of slices of length M
            defining a partition of {1, ..., T} into M subsets, giving the
            samples from C
        TH : dict
            LEMM parameters

        Returns
        -------
        log_XgVZ : (N, T) ndarray

        """
        logA = 0.5 * self.n * np.log(2*np.pi)
        if TH.tied:
            if TH.covar_type == 'spherical':
                Y = X * TH.cv_invchol
                W = VZ * TH.cv_invchol
            if TH.covar_type == 'diagonal':
                Y = X * TH.cv_invchol[None, :]
                W = VZ * TH.cv_invchol[None, :]
            if TH.covar_type == 'full':
                Y = X.dot(TH.cv_invchol)
                W = VZ.dot(TH.cv_invchol)
            return -0.5 * np.sum(np.square(
                Y[:, None, :] - W[None, :, :]
            ), axis=2) - (logA + 0.5 * TH.cv_logdet)
        else:
            Y = np.zeros_like(X)
            W = np.zeros_like(VZ)
            if TH.covar_type == 'spherical':
                for invchol, ix in zip(TH.cv_invchol, indices):
                    Y[ix] = X[ix] * invchol
                    W[ix] = VZ[ix] * invchol
            if TH.covar_type == 'diagonal':
                for invchol, ix in zip(TH.cv_invchol, indices):
                    Y[ix] = X[ix] * invchol[None, :]
                    W[ix] = VZ[ix] * invchol[None, :]
            if TH.covar_type == 'full':
                for invchol, ix in zip(TH.cv_invchol, indices):
                    Y[ix] = X[ix].dot(invchol)
                    W[ix] = VZ[ix].dot(invchol)

            logdet = np.zeros((VZ.shape[0],))
            for a, ix in zip(TH.cv_logdet, indices):
                logdet[ix] = a

            return -0.5 * np.sum(np.square(
                Y[:, None, :] - W[None, :, :]
            ), axis=2) - (logA + 0.5 * logdet)

    def log_X_and_VZ_and_C(self, X, VZ, indices, TH, log_PZ):
        r"""
        Logarithm of probability of X and V(Z) and the
        distribution it arose from (indicated by the indices list).
        Simply applies Bayes' Law

        .. math::

            P(X \cap VZ \cap C) = P(X \mid VZ, C) P(VZ \cap C)

        Parameters
        ----------
        X : (N, n) ndarray
        VZ : (T, n) ndarray
        indices : list of slices of length M
            defining a partition of {1, ..., T} into M subsets.
        TH : dict
            LEMM parameters.
        log_PZ : (T,) ndarray
            indicating the relative prior probability of each samples from Z.

        Returns
        -------

        log_XnVZ : (N, T) ndarray

        """
        return self.log_X_given_VZ_and_C(
            X, VZ, indices, TH) + log_PZ[np.newaxis, :]

    def log_X_and_Z_and_C(self, X, Z, indices, TH, log_PZ):
        """
        Logarithm of probability of X and Z and the
        distribution it arose from (indicated by the indices list).

        Parameters
        ----------
        X : (N, n) ndarray
        Z : (T, m) ndarray
        indices : list of slices of length M
            defining a partition of {1, ..., T} into M subsets.
        TH : dict
            LEMM parameters.
        log_PZ : (T,) ndarray
            indicating the relative prior probability of each value of Z.

        Returns
        -------
        log_XnZ : (N, T) ndarray

        See Also
        --------
        LinearlyEmbeddedMM.log_X_given_VZ_and_C

        """
        VZ = Z.dot(TH.V)
        return self.log_X_and_VZ_and_C(X, VZ, indices, TH, log_PZ)

    def log_Z_and_C_given_X(self, X, Z, indices, TH, log_PZ):
        """
        For each value in X, computes a distribution on Z, representing the
        probability that X is paired with a particular Z.
        The logarithms of the probabilities are returned.

        Parameters
        ----------
        X : (N, n) ndarray
        Z : (T, m) ndarray
        indices : list of slices of length M
            defining a partition of {1, ..., T} into M subsets.
        TH : dict
            LEMM parameters.
        log_PZ : (T,) ndarray
            indicating the relative prior probability of each value of Z.

        Returns
        -------
        log_ZgX : (N, T) ndarray
        """
        VZ = Z.dot(TH.V)
        return self.log_VZ_and_C_given_X(X, VZ, indices, TH, log_PZ)

    def log_VZ_and_C_given_X(self, X, VZ, indices, TH, log_PZ):
        """
        For each value in X, computes a distribution on V(Z), representing the
        probability that X is paired with a particular V(Z).
        The logarithms of the probabilities are returned.

        Parameters
        ----------
        X : (N, n) ndarray
        VZ : (T, n) ndarray
        indices : list of slices of length M
            defining a partition of {1, ..., T} into M subsets.
        TH : dict
            LEMM parameters.
        log_PZ : (T,) ndarray
            indicating the relative prior probability of each value of Z.

        Returns
        -------
        log_VZgX : (N, T) ndarray
        """
        log_PXnVZnC = self.log_X_and_VZ_and_C(X, VZ, indices, TH, log_PZ)
        log_PX = logsumexp(log_PXnVZnC, axis=1, keepdims=True)
        return log_PXnVZnC - log_PX

    def estimated_log_PX(self, TH, X, n_samples=1000):
        """
        Calculate an estimate of the log density function for each data point
        x of X, based on a sample from the LEMM.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            input data
        n_samples : int, optional
            number of samples from LEMM to use.

        Returns
        -------
        log_PX : (N,) ndarray
        """

        VZ, indices, log_PZ = self.sample(TH, n_samples, in_order=True,
                                          at_least=0, output_weights=True)

        log_PXnVZnC = self.log_X_and_VZ_and_C(X, VZ, indices, TH, log_PZ)
        log_PX = logsumexp(log_PXnVZnC, axis=1) - np.log(log_PXnVZnC.shape[1])

        return log_PX

    def estimated_log_likelihood(self, TH, X, **kw_args):
        """
        Calculate an estimate of the log likelihood given X, based on a sample
        from the LEMM.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            input data
        n_samples : int, optional
            number of samples from LEMM to use.

        Returns
        -------
        log_likelihood : float
            the estimated log likelihood.
        """

        return np.mean(self.estimated_log_PX(TH, X, **kw_args))

    def estimated_expected_ZgivenX(self, TH, X, n_samples=1000):
        """
        Given X estimates the expected value of hidden Z.

        Parameters
        ----------
        TH : dict
            parameters of LEMM,
        X : (N, n) ndarray
            input data.
        n_samples : int, optional
            number of samples from LEMM to use.

        Returns
        -------
        E_ZgX : (N, m) ndarray
        """

        Z, indices, log_PZ = self.Z_sample(TH.logp, n_samples,
                                           in_order=True, output_weights=True)
        VZ = Z.dot(TH.V)

        log_ZgX = self.log_VZ_and_C_given_X(X, VZ, indices, TH, log_PZ)
        E_ZgX = np.exp(log_ZgX).dot(Z)

        return E_ZgX

    def estimated_expected_CandZgivenX(self, TH, X, n_samples=1000):
        r"""
        Given X computes the expected value of hidden :math:`\mathbb{I}_CZ`.

        Parameters
        ----------
        TH : dict
            parameters of LEMM,
        X : (N, n) ndarray
            input data.
        n_samples : int, optional
            number of samples from LEMM to use.

        Returns
        -------
        E_CnZgX : (N, M, m) ndarray
        """

        Z, indices, log_PZ = self.Z_sample(TH.logp, n_samples,
                                           in_order=True, output_weights=True)
        VZ = Z.dot(TH.V)
        N = X.shape[0]

        log_ZgX = self.log_VZ_and_C_given_X(X, VZ, indices, TH, log_PZ)
        print(log_ZgX.shape)
        E_CnZgX = np.zeros((N, self.M, self.m))
        for i, ix2 in enumerate(indices):
            E_CnZgX[:, i, :] = np.exp(log_ZgX[:, ix2]).dot(Z[ix2])

        return E_CnZgX

    def estimated_expected_CgivenX(self, TH, X, **kw_args):
        """
        Given X calculated P(C | X) for the hidden categorical
        latent variable C.

        Parameters
        ----------
        TH : dict
            parameters of LEMM,
        X : (N, n) ndarray
            input data.
        n_samples : int, optional
            number of samples from LEMM to use.

        Returns
        -------
        E_CgX : (N, M) ndarray
        """

        E_CgX = np.sum(self.estimated_expected_CandZgivenX(
            TH, X, **kw_args), axis=2)

        return E_CgX

    def estimated_expected_q_values(self, X, TH, n_samples,
                                    batch_size=2000, at_least=1):
        """
        Estimates some expected values based on a sample
        from the LEMM with parameters TH and some set of data X.

        Parameters
        ----------
        X : (N, n) ndarray
            input data.
        TH : dict
            LEMM parameters.
        n_samples : int
            number of samples from LEMM to use.
        batch_size : int, optional
            the number of data values to process at once, if set to None
            will process all data at once.
        at_least : int, optional
            sample from each rv at least this number of times.

        Returns
        -------
        q : (M,) ndarray
        qZZ : (M, m, m) ndarray
        qZX : (M, m, n) ndarray
        qXX : (n, n) ndarray if tied, otherwise (M,n,n) ndarray
        """

        def _batch(self, X, Z, VZ, indices, TH, log_PZ):
            # shape (N, samples)
            log_ZgX = self.log_VZ_and_C_given_X(X, VZ, indices, TH, log_PZ)
            # shape (samples,)
            logp_per_Z = logsumexp(log_ZgX, axis=0)

            logq = np.zeros((self.M,))
            qZZ = np.zeros((self.M, self.m, self.m))
            qZX = np.zeros((self.M, self.m, self.n))
            if TH.tied:
                # qXX = X.T.dot(X)
                qXX = TH.calc_XX(X)
            else:
                qXX = np.zeros((self.M, self.n, self.n))
            p_per_Z = np.exp(logp_per_Z)
            p_ZgX = np.exp(log_ZgX)
            for i, ix in enumerate(indices):
                if ix.stop > ix.start:
                    logq[i] = logsumexp(logp_per_Z[ix])
                else:
                    logq[i] = np.log(0)

                qZZ[i] = np.sum(
                    p_per_Z[ix, None, None] * Z[ix, :, None] * Z[ix, None, :],
                    axis=0)

                # shape (n, ix)
                temp = np.sum(p_ZgX[None, :, ix] * X.T[:, :, None], axis=1)
                # shape (m, n)
                qZX[i] = np.sum(temp[None, :, :] * Z.T[:, None, ix], axis=2)

                if not TH.tied:
                    p_XgC = np.sum(p_ZgX[:, ix], axis=1)
                    qXX[i] = (p_XgC[:, None]*X).T.dot(X) / N

            return logq, qZZ, qZX, qXX

        # Generate samples from Z
        Z, indices, log_PZ = self.Z_sample(TH.logp, n_samples,
                                           in_order=True, at_least=at_least,
                                           output_weights=True)
        VZ = Z.dot(TH.V)
        N = X.shape[0]

        if batch_size is None:
            batch_size = N

        logq, qZZ, qZX, qXX = -1e20, 0, 0, 0
        for ix in [slice(i, i+batch_size) for i in range(0, N, batch_size)]:
            a, b, c, d = _batch(self, X[ix], Z, VZ, indices, TH, log_PZ)
            logq = np.logaddexp(logq, a)
            qZZ += b
            qZX += c
            qXX += d

        return np.exp(logq)/N, qZZ/N, qZX/N, qXX

    def stochastic_step(self, TH, X, n_samples=1000, batch_size=2000):
        """
        A single step of the stochastic EM algorithm, starting at the given
        parameters and using samples from Z to estimate the required expected
        values.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            input data.
        n_samples : int
            number of samples from LEMM to use.
        batch_size : int, optional
            the number of data values to process at once, if set to None
            will process all data at once.

        Returns
        -------
        newTH : dict
            modified LEMM parameters.
        """

        # E-step
        q, qZZ, qZX, qXX = \
            self.estimated_expected_q_values(X, TH, n_samples,
                                             batch_size=batch_size)

        # M-step
        newTH = TH.from_minimisation(q, qZZ, qZX, qXX)

        return newTH
