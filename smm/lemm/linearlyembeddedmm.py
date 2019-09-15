"""
The core module, implementing Linear Mixture Models and the EM algorithm.
"""

import numpy as np
from smm.lemm._lemm_estimates import LEMMEstimatesMixin
from smm.helpfulfunctions import shuffle_arrays, logsumexp


class LinearlyEmbeddedMM(LEMMEstimatesMixin):
    """
    A class for a linearly embedded mixture model, implementing sampling etc.

    The class holds the parameters of the model which are fixed, i.e. the
    the respective dimensions and the list of random variables to be mixed.
    The other parameters which are considered variable are held separately
    in a class that is passed to the functions that need them.

    Parameters
    ----------
    m : int
        the dimension of the latent model
    n : int
        the dimension of the observed model
    rvs : list of random variables
        the list of hidden components of the mixture
    rnd : np.random.RandomState, optional
        random number generator.
    """

    def __init__(self, m, n, rvs, rnd=np.random.RandomState()):
        self.m, self.n = m, n
        self.rvs = rvs
        self.M = len(rvs)
        self.rnd = rnd
        for rv in rvs:
            if rv.m != m:
                raise ValueError("Dimensions of random variables must agree.")

    def copy(self):
        """
        A soft copy of L.

        The list of random variables is copied, but the individual random
        variables are not.
        """
        newL = self.__class__(self.m, self.n, self.rvs.copy(), self.rnd)
        return newL

    def sample(self, TH, n_samples, in_order=False, output_weights=False,
               **kw_args):
        """
        Returns an embedded sample from the LEMM model with given
        parameters, TH.

        Note, the samples from each distribution are returned in sequence, so
        the samples are only independent after a random shuffle.

        Parameters
        ----------
        TH : dict
            parameters for the LEMM,
        n_samples : int
            number of requested samples.
        at_least : int, optional
            specifies that the sample must include at least
            this number of points from each distribution
        in_order : bool, optional
            whether to list the samples in order of the component random
            variables they were generated from.
        output_weights : bool, optional
            whether or not to list relative weights for each output.

        Returns
        -------
        sample : (n_samples, m) ndarray
             the returned samples.
        {indices, C_sample} : {list of slices, (n_samples,) ndarray}
            if in_order then a list of slices indicates which points are drawn
            from its respective distribution, else a list of those
            distributions.
        log_sw : (n_samples,) ndarray, only when output_weights==True
            the relative log-probabilities of each sample.
        """
        outputs = self.Z_sample(TH.logp, n_samples, in_order=in_order,
                                output_weights=output_weights, **kw_args)
        return (outputs[0].dot(TH.V), *outputs[1:])

    def noisy_sample(self, TH, n_samples, in_order=False, output_weights=False,
                     **kw_args):
        """
        Returns a sample from the LEMM model with given parameters, TH, plus
        a noise term from the respective kernel.

        Note, if in_order=True then the samples from each distribution are
        returned in sequence, so the samples are only independent after a
        random shuffle.

        Parameters
        ----------
        TH : dict
            parameters for the LEMM,
        n_samples : int
            number of requested samples.
        at_least : int, optional
            specifies that the sample must include at least
            this number of points from each distribution
        in_order : bool, optional
            whether to list the samples in order of the component random
            variables they were generated from.
        output_weights : bool, optional
            whether or not to list relative weights for each output.

        Returns
        -------
        sample : (n_samples, n) ndarray
             the returned samples.
        {indices, C_sample} : {list of slices, (n_samples,) ndarray}
            if in_order then a list of slices indicates which points are drawn
            from its respective distribution, else a list of those
            distributions.
        log_sw : (n_samples,) ndarray, only when output_weights==True
            the relative log-probabilities of each sample.
        """
        if not in_order:
            outputs = self.noisy_sample(TH, n_samples,
                                        output_weights=output_weights,
                                        in_order=True, **kw_args)

            C_sample = np.empty((n_samples,), dtype=np.int)
            for i, ix in enumerate(outputs[1]):
                C_sample[ix] = i
            new_outputs = (outputs[0], C_sample, *outputs[2:])

            shuffle_arrays(new_outputs, self.rnd)

            return new_outputs

        outputs = self.sample(TH, n_samples, in_order=True,
                              output_weights=output_weights, **kw_args)

        new_outputs = (outputs[0] + self.noise(TH, outputs[1]), *outputs[1:])
        return new_outputs

    def C_sample(self, logp, n_samples, at_least=0, in_order=False,
                 output_weights=False):
        """
        Returns a sample from the LEMM model with given parameters, TH.

        Note, if in_order=True then the samples from each distribution are
        returned in sequence, so the samples are only independent after a
        random shuffle.

        Parameters
        ----------
        logp : (M,) ndarray
            log-probabilities for the mixture of hidden distributions.
        n_samples : int
            number of requested samples.
        at_least : int, optional
            specifies that the sample must include at least
            this number of points from each distribution.
        in_order : bool, optional
            whether to list the samples in order of the component random
            variables they were generated from.
        output_weights: bool, optional
            whether or not to output a weight prescribed to each vertex.

        Returns
        -------
        {indices, C_sample} : {list of slices, (n_samples,) ndarray}
            if in_order then a list of slices indicates which points are drawn
            from its respective distribution, else a list of those
            distributions.
        log_sw : (n_samples,) ndarray, only for output_weights==True
            the relative log-probabilities of each sample.
        """
        assert len(logp) == self.M, "Wrong number of probabilities"

        assert n_samples >= self.M * at_least, \
            f"Not enough samples for simplices with at_least={at_least}"
        samples_per_rv = self.rnd.multinomial(n_samples - at_least*self.M,
                                              np.exp(logp)) + at_least

        changes = np.zeros((self.M + 1,), dtype=np.int)
        np.cumsum(samples_per_rv, out=changes[1:])
        indices = [slice(a, b) for a, b in zip(changes, changes[1:])]

        if output_weights:
            log_sample_weights = np.zeros((n_samples,))
            for ix, nlk in zip(indices, -np.log(samples_per_rv)):
                log_sample_weights[ix] = nlk

        if in_order:
            if output_weights:
                return indices, log_sample_weights
            else:
                return indices

        C_sample = np.empty((n_samples,), dtype=np.int)
        for i, ix in enumerate(indices):
            C_sample[ix] = i

        if output_weights:
            shuffle_arrays([C_sample, log_sample_weights])
            return C_sample, log_sample_weights
        else:
            shuffle_arrays([C_sample])
            return C_sample

    def Z_sample(self, logp, n_samples, in_order=False,
                 output_weights=False, **kw_args):
        """
        Returns a sample from the LEMM model with given parameters, TH.

        Note, the samples from each distribution are returned in sequence, so
        the samples are only independent after a random shuffle.

        Parameters
        ----------
        logp : (M,) ndarray
            log-probabilities for the mixture of hidden distributions.
        n_samples : int
            number of requested samples.
        at_least : int, optional
            specifies that the sample must include at least
            this number of points from each distribution.
        in_order : bool, optional
            whether to list the samples in order of the component random
            variables they were generated from.
        output_weights: bool, optional
            whether or not to output a weight prescribed to each vertex.

        Returns
        -------
        sample : (n_samples, m) ndarray
            the samples drawn from the mixture of hidden distributions.
        {indices, C_sample} : {list of slices, (n_samples,) ndarray}
            if in_order then a list of slices indicates which points are drawn
            from its respective distribution, else a list of those
            distributions.
        log_sw : (n_samples,) ndarray, only for output_weights==True
            the relative log-probabilities of each sample.
        """
        if not in_order:  # Shuffle ordered data sample
            output = self.Z_sample(logp, n_samples, in_order=True,
                                   output_weights=output_weights, **kw_args)

            C_sample = np.empty((n_samples,), dtype=np.int)
            for i, ix in enumerate(output[1]):
                C_sample[ix] = i

            new_output = (output[0], C_sample, *output[2:])

            shuffle_arrays(new_output, rnd=self.rnd)
            return new_output

        if output_weights:
            indices, log_sw = self.C_sample(logp, n_samples, in_order=True,
                                            output_weights=output_weights,
                                            **kw_args)
        else:
            indices = self.C_sample(logp, n_samples, in_order=True,
                                    output_weights=output_weights, **kw_args)

        Z_sample = np.zeros((n_samples, self.m))

        for rv, ix, lp in zip(self.rvs, indices, logp):
            k = ix.stop - ix.start
            if k > 0:
                Z_sample[ix] = rv.sample(k, self.rnd)

        if output_weights:
            return Z_sample, indices, log_sw
        else:
            return Z_sample, indices

    def noise(self, TH, indices):
        """
        A function generating noise using the individual kernels.
        Which kernel is used to produce which noise sample is determined by
        the list, indices.

        Parameters
        ----------
        TH : dict
            parameters of the LEMM.
        indices : list of slices of length M
            determines which points are drawn from which kernel

        Returns
        -------
        output : (N, n) ndarray
            the output noise samples where N is the number of samples required,
            which is determined from indices.
        """
        if len(indices) != self.M:
            raise ValueError("Indices do not match with RVs")

        def mv_normal(n_samples, type, chol):
            sample = self.rnd.standard_normal((n_samples, self.n))
            if type == 'spherical':
                return sample * chol
            if type == 'diagonal':
                return sample * chol[None, :]
            if type == 'full':
                return sample.dot(chol)

        n_samples = indices[-1].stop
        if TH.tied:
            output = mv_normal(n_samples, TH.covar_type, TH.cv_chol)
            return output
        output = np.zeros((n_samples, self.n))
        for ix, chol in zip(indices, TH.cv_chol):
            output[ix] = mv_normal(ix.stop-ix.start, TH.covar_type, chol)
        return output

    def mean_and_covar(self, TH, include_noise=False):
        """
        Calculate the mean and covariance of the LEMM, to take account of
        the kernels use 'include_noise=True'.

        Parameters
        ----------
        TH - dict

        Returns
        -------
        mean : (n,) array
            the mean of the LEMM
        covar : (n, n) array
            the covariance matrix of the LEMM
        """
        if include_noise:
            raise NotImplementedError()
        covar = np.zeros((self.n, self.n))
        mean = np.zeros((self.n,))
        V = TH.V
        for rv, p in zip(self.rvs, np.exp(TH.logp)):
            comp_mean = rv.mean.dot(V)
            comp_covar = V.T.dot(rv.covar).dot(V)
            mean += p * comp_mean
            covar += p * (comp_covar + np.outer(comp_mean, comp_mean))
        covar -= np.outer(mean, mean)
        return mean, covar

    def top_components(self, TH, number=None, threshold=0):
        """
        Compute an ordered list of the distributions with the highest
        mixing probabilities, each paired with their respective probabilities.

        Parameters
        ----------
        TH : dict
            parameters for the LEMM.
        number : int, optional
            if not None, returns only this number of distributions.
        threshold : float, optional
            only returns distributions with mixing probabilities above this
            threshold.

        Returns
        -------
        top_simplices : list of pairs
            a list, each entry is a pair (i, p) of with i denoting a component
            and p its mixing probability.
        """
        if number is None:
            number = self.M
        pairs = filter(lambda a: a[1] > threshold,
                       enumerate(np.exp(TH.logp)))
        return sorted(pairs, key=lambda a: -a[1])[:number]

    def entropy_of_distribution(self, logp, log_given=True):
        """
        The entropy of the given categorical distribution.

        Parameters
        ----------
        logp : (M,) ndarray
            logarithm of probabilities.

        Returns
        -------
        entropy : float
            the calculated entropy.
        """
        if not log_given:
            logp = np.log(logp)
        return -sum(lp * np.exp(lp) for lp in logp if lp > -100)

    def encoding_rate(self, TH, in_bits=False):
        """
        Computes an upper bound R on the rate required to encode samples
        (C, Z, X) from the model.
        This is used as a goodness-of-fit measure in conjunction with the
        amount of data required to encode the model itself.

        Note
        ----
        The bound only applies asymptotically, if accuracy up to exp(-B)
        is required then R + nB bounds the encoding rate for large B.
        So do not be surprised if the value is negative.

        Parameters
        ----------
        TH : dict
            the parameters of the model.
        in_bits : bool, optional
            return the answer in bits instead of nats.

        Returns
        -------
        R : float
            the bound on the rate.
        """
        if TH.tied:
            # Calculate rate distortion for each component random variable
            rate_per_U = [rv.rate_fn_bound(TH.covar, TH.V)
                          for rv in self.rvs]
            logdet_term = 0.5 * TH.cv_logdet
        else:
            rate_per_U = [rv.rate_fn_bound(cv, TH.V)
                          for rv, cv in zip(self.rvs, TH.covar)]
            logdet_term = 0.5 * np.exp(TH.logp).dot(TH.cv_logdet)

        # Calculate the average rate distortion over components
        rate_distortion = np.exp(TH.logp).dot(np.array(rate_per_U))

        bound = self.entropy_of_distribution(TH.logp) + \
            logdet_term + rate_distortion

        if in_bits:  # Convert to bits if requested
            bound /= np.log(2)
        return bound

    def number_to_keep(self, logp, loss=-np.log(0.99)):
        """
        Calculates how many components are needed to avoid losing more than
        a certain amount of entropy from the given distribution.

        Parameters
        ----------
        logp : (M,) ndarray
            logarithm of probabilities.
        loss : float, optional
            the amount of entropy we are prepared to lose, must be less than
            zero.

        Returns
        -------
        number : int
            the number required to avoid the loss.
        """
        if not loss > 0:
            raise ValueError("Loss must be greater than 0.")

        req_entr = self.entropy_of_distribution(logp) - loss
        number = int(np.exp(req_entr))  # At least this number required
        while self.entropy_of_distribution(logp[:number]) > req_entr:
            number += 1

        return number

    def log_prob_X_given_C(self, TH, X):
        """
        Calculates the probability of each row of X given each of the
        random variables.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            points at which to calculate probabilities.

        Returns
        -------
        log_PXgC : (M, N)
            calculated probability of each row of X for each random variable.
        """
        N = X.shape[0]
        log_PXgC = np.zeros((self.M, N))
        for i, rv in enumerate(self.rvs):
            if TH.tied:
                S = TH.covar
            else:
                S = TH.covar[i]
            log_PXgC[i] = rv.log_prob_X(TH.V, S, X)
        return log_PXgC

    def log_prob_C_given_X(self, TH, X):
        """
        Calculates the probability distribution of C given each row of X and
        the prior probabilities from TH.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            points at which to calculate probabilities.

        Returns
        -------
        log_PCgX : (M, N)
            calculated probability of each row of X for each random variable.
        """
        log_PXgC = self.log_prob_X_given_C(TH, X)

        log_PXgC += TH.logp[:, None]
        log_PXnC = log_PXgC
        log_PX = logsumexp(log_PXnC, axis=0)
        log_PXnC -= log_PX
        log_PCgX = log_PXnC

        return log_PCgX

    def expected_q_values(self, X, TH):
        """
        Calculates the expected values required of the 'E step' in the EM
        algorithm.

        Parameters
        ----------
        X : (N, n) ndarray
            input data.
        TH : dict
            LEMM parameters.

        Returns
        -------
        q : (M,) ndarray
        qZZ : (M, m, m) ndarray
        qZX : (M, m, n) ndarray
        qXX : (n, n) ndarray if tied, otherwise (M,n,n) ndarray
        """

        log_PCgX = self.log_prob_C_given_X(TH, X)

        logq = logsumexp(log_PCgX, axis=1)

        qZX = np.zeros((self.M, self.m, self.n))
        qZZ = np.zeros((self.M, self.m, self.m))

        PCgX = np.exp(log_PCgX)
        for i, rv in enumerate(self.rvs):
            qZZ[i], qZX[i] = rv.moments_marg_over_X(PCgX[i], X)

        if TH.tied:
            qXX = X.T.dot(X)
        else:
            qXX = X.T.dot((PCgX[:, :, None] * X)).swapaxes(0, 1)

        return np.exp(logq), qZZ, qZX, qXX

    def step(self, TH, X, alpha=None):
        """
        A single step of the EM algorithm, starting at the given
        parameters and using methods from the RVs to compute expected values.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            input data.
        alpha : float, optional
            if present represents a symmetric Dirichlet prior on the
            probabilities.

        Returns
        -------
        newTH : dict
            modified LEMM parameters.
        """

        # E-step
        q, qZZ, qZX, qXX = self.expected_q_values(X, TH)

        # M-step
        newTH = TH.from_minimisation(q, qZZ, qZX, qXX, alpha)

        return newTH

    def log_PX(self, TH, X):
        """
        Calculates the probability density of the LEMM with kernels at each
        point given.  This involves marginalising out the categorical
        distribution C determining the choice of mixture.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            points at which to calculate probabilities.

        Returns
        -------
        log_PX : (N,)
            calculated probability densities.
        """
        log_PXgC = self.log_prob_X_given_C(TH, X)

        # Marginalise over C
        log_PX = logsumexp(log_PXgC + TH.logp[:, None], axis=0)
        return log_PX

    def log_likelihood(self, TH, X, **kw_args):
        """
        Calculates the log-likelihood of the LEMM with given parameters for
        the given set of data X.
        This can only be provided when the method log_prob_X_given_C is
        implemented by the random variables.

        Parameters
        ----------
        TH : dict
            parameters of the LEMM.
        X : (N, n) ndarray
            points at which to base the log_likelihood.

        Returns
        -------
        log_likelihood : float
            The mean of the log-likelihood across the given set.
        """
        log_PX = self.log_PX(TH, X, **kw_args)
        return np.mean(log_PX)

    def expected_C_and_Z_given_X(self, TH, X):
        """
        Given X computes the expected value of hidden C and Z.

        Parameters
        ----------
        TH : dict
            parameters of LEMM,
        X : (N, n) ndarray
            input data.

        Returns
        -------
        E_ZnCgX : (N, M, m) ndarray
        """
        log_PCgX = self.log_prob_C_given_X(TH, X)  # shape (M, N)

        N = X.shape[0]
        E_ZgXnC = np.zeros((N, self.M, self.m))
        for i, rv in enumerate(self.rvs):
            E_ZgXnC[:, i, :] = rv.mean_given_X()

        PCgX = np.exp(log_PCgX).T  # shape (N, M)
        E_ZnCgX = PCgX[:, :, None] * E_ZgXnC

        return E_ZnCgX

    def expected_Z_given_X(self, TH, X):
        """
        Given X computes the expected value of hidden Z.

        Parameters
        ----------
        TH : dict
            parameters of LEMM,
        X : (N, n) ndarray
            input data.

        Returns
        -------
        E_ZgX : (N, m) ndarray
        """
        E_ZnCgX = self.expected_C_and_Z_given_X(TH, X)
        E_ZgX = np.sum(E_ZnCgX, axis=1)

        return E_ZgX

    def expected_C_given_X(self, TH, X):
        """
        Given X computes the expected value of hidden C.

        Parameters
        ----------
        TH : dict
            parameters of LEMM.
        X : (N, n) ndarray
            input data.

        Returns
        -------
        E_CgX : (N, M) ndarray
        """
        E_ZnCgX = self.expected_C_and_Z_given_X(TH, X)
        E_ZgX = np.sum(E_ZnCgX, axis=2)

        return E_ZgX

    def truncate(self, TH, threshold=None, number=None):
        """
        Returns a new LEMM with only those components with highest weight.

        If threshold is not specified, but number is, a maximum threshold will
        be calculated such that at least this number of components are
        retained.

        Parameters
        ----------
        TH : dict
            parameters of the LEMM.
        threshold : float, optional
            keep components above this weight threshold.
        number : int, optional
            keep this number of components.

        Returns
        -------
        trunc_L : LinearlyEmbeddedMM
            the new LEMM with fewer components.
        trunc_TH : dict
            parameters to match the new LEMM.
        """
        if threshold is None and number is None:
            raise ValueError("Must specify either a number or threshold.")

        if threshold is None:
            if number == self.M:
                log_threshold = -1000.0
            else:
                log_threshold = sorted(TH.logp, reverse=True)[number]
        else:
            log_threshold = np.log(threshold)

        to_keep = [i for i in range(self.M) if TH.logp[i] >= log_threshold]
        if len(to_keep) == 0:
            raise ValueError("With these inputs no components left.")

        # Create the new model based on to_keep
        new_rvs = [rv for i, rv in enumerate(self.rvs) if i in to_keep]
        trunc_L = self.copy()
        trunc_L.rvs = new_rvs
        trunc_L.M = len(new_rvs)

        # New covariances to match the new model
        if TH.tied:
            if TH.covar_type == 'spherical':
                trunc_cv = TH.covar
            else:
                trunc_cv = TH.covar.copy()
        else:
            trunc_cv = TH.covar[to_keep].copy()

        # New parameters
        new_logp = TH.logp[to_keep].copy()
        new_logp -= logsumexp(new_logp, keepdims=True)
        trunc_TH = TH.__class__(TH.V.copy(), len(to_keep),
                                new_logp,
                                TH.covar_type, trunc_cv)

        return trunc_L, trunc_TH

    # def set_V_regularisation(self, V_mean, V_invcov):
    #     r"""
    #     Adds Gaussian regularisation for the linear map V.
    #
    #     Parameters
    #     ----------
    #     V_mean : (m, n) ndarray
    #         mean of the regularisation.
    #     V_invcov : {(m, n, m, n) ndarray, (n, n) ndarray, float}
    #         inverse covariance of the regularisation, either the full matrix,
    #         a fixed covariance matrix on :math:`\mathbb{R}^n`, or a float.
    #     """
    #     self.V_mean = V_mean
    #     if len(np.shape(V_invcov)) == 2:
    #         self.V_invcov = V_invcov[None, :, None, :] * \
    #             np.eye(self.m)[:, None, :, None]
    #     else:
    #         self.V_invcov = V_invcov
    #     self.using_V_regularisation = True


__all__ = ["LinearlyEmbeddedMM"]
