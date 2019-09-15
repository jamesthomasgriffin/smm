import numpy as np
from smm.lemm.numba_optimisations import (count_C,
                                          indexed_A_xAy,
                                          indexed_A_gemv,
                                          indexed_Ab_xApb)


class EstimatedValuesAccumulator:
    """
    A class to keep track of expected values.  When a value is recorded a
    weight is also taken, allowing a weighted average to be formed.
    """

    def __init__(self):
        self.value = {}
        self.weight = {}

    def record(self, name, value, weight=1.0):
        """
        Records a value and an associated weight for a named quantity.

        Parameters
        ----------
        name : string
            the name of the quantity we wish to keep track of.
        value : ndarray
            the value of the quantity.
        weight : float, optional
            the number of samples summed to reach the given value.
        """
        if name not in self.value:
            self.value[name] = 0
            self.weight[name] = 0
        self.value[name] += value
        self.weight[name] += weight

    def reset(self, name):
        """
        Sets both the value and its weight to 0 for the given quantity.

        Parameters
        ----------
        name : string
            the name of the quantity we wish to reset.
        """
        if name in self.value:
            self.value[name] = 0
            self.weight[name] = 0

    def record_interpolate(self, name, value, alpha, weight=None):
        if name not in self.value:
            raise KeyError(f"Cannot interpolate {name} if it isn't there")
        self.value[name] = self.value[name] * (1-alpha) + value * alpha
        if weight is not None:
            self.weight[name] = self.weight[name] * (1-alpha) + weight * alpha

    def reweight(self, name, factor=None, weight=None):
        """
        This function reweights the given quantity, either by a factor, or
        to a given weight value.  This does not change the weighted average,
        but it does change the importance of the previously recorded values.

        Parameters
        ----------
        name : string
            the name of the quantity we wish to reset.
        factor : float, optional
            the factor we wish to reweight by.
        weight : float, optional
            the new weight.
        """
        if factor is None and weight is None:
            raise ValueError("One of factor or weight has to be given")
        if name not in self.value:
            raise KeyError(f"You cannot reweight {name} if it isn't there")
        if weight is not None:
            factor = weight / self.weight[name]
        self.value[name] *= factor
        self.weight[name] *= factor

    def expected_value(self, name):
        """
        Returns the weighted average of the requested quantity.

        Parameters
        ----------
        name : string
            the name of the quantity we wish to retrieve.
        """
        if name not in self.value:
            raise KeyError(f"No record of {name}")
        return self.value[name] / self.weight[name]


class MCMC_Integrator:
    r"""
    A class to hold a Markov chain for each data point.

    It handles a Markov process for each data point in a list, holding in
    memory the current state along with the associated quadratic forms.

    It requires that the latent variables in the list of random variables
    are equal in distribution, but they may have different embeddings in the
    feature space :math:`\mathbb{R}^m`.
    """

    def __init__(self, L, TH, X, alpha=1.0,
                 rnd=np.random.RandomState()):
        """
        Parameters
        ----------
        L : LinearlyEmbeddedMM
            the linear mixture model on which this process is based.
        TH : dict
            parameters for the LEMM.
        X : (N, n) ndarray
            data set the LEMM is intended to model.
        alpha : float, optional
            if present represents a symmetric Dirichlet prior on the
            probabilities.
        rnd : np.random.RandomState, optional
            random number generator.

        ..Warning ::

            Note that the contributions to expected values by a Markov chain
            are not independent, so the when updating the probabilities using
            a Dirichlet prior, the evidence is overstated.
        """
        self.L = L
        self.TH = TH
        self.X = X
        self.rnd = rnd

        # The typical random variable
        self.lead_rv = L.rvs[0]
        rv_cls = self.lead_rv.__class__
        self.k = self.lead_rv.k
        for rv in L.rvs:
            if rv.k != self.k:
                raise ValueError("All rvs must have the same latent dimension")
            if rv.__class__ != rv_cls:
                raise ValueError("All rvs must be of the same type")
        self.N, n = X.shape
        if n != L.n:
            raise ValueError("Dimension mismatch")

        # Initialise the state of the walkers
        self.U = self.lead_rv.U_sample(X.shape[0], self.rnd)
        self.C = self.L.C_sample(TH.logp, self.N, in_order=False)

        # These define the embeddings of the latent variables and do not
        # change
        self.ms = np.vstack([rv.Moffset for rv in self.L.rvs])
        self.Ms = np.vstack([rv.M[None, :, :] for rv in self.L.rvs])

        # Compute quantities representing the quadratic form on the latent
        # variables
        self.update_quad_forms()

        self._Z = np.empty((self.N, self.L.m))
        self._Z_needs_update = True
        self.Z()

        # A class that keeps track of estimates of expected values
        self.exp_values = EstimatedValuesAccumulator()

        # A term in the minimisation which never changes
        self.qXX = X.T.dot(X) / self.N

        self.alpha = alpha

    def Z(self):
        r"""
        Returns the coordinates of the Markov chain states in
        feature space :math:`\mathbb{R}^m`.
        """
        if self._Z_needs_update:
            self.U_to_Z(self.U, self.C, self._Z)
            self._Z_needs_update = False
        return self._Z

    def X_walkers(self):
        r"""
        Returns the coordinates of the Markov chain states in the same space
        as the data :math:`\mathbb{R}^n`.
        """
        if self._Z_needs_update:
            return self.U_to_X(self.U, self.C)
        else:
            return self._Z.dot(self.TH.V)

    def U_to_Z(self, U, C, Z=None):
        """
        Transform pairs (C, U) to Z, inplace if Z is not None.
        """
        if Z is None:
            Z = np.zeros((*U.shape[:-1], self.L.m))
        return indexed_Ab_xApb(C, U, self.Ms, self.ms, Z)

    def U_to_X(self, U, C, X=None):
        """
        Transform pairs (C, U) to X, inplace if X is not None.
        """
        if X is None:
            X = np.zeros((*U.shape[:-1], self.L.n))
        return indexed_Ab_xApb(C, U,
                               self.Ms.dot(self.TH.V),
                               self.ms.dot(self.TH.V), X)

    def C_candidates(self):
        """
        Computes candidates and relative probabilities in the C Markov chain.

        Returns
        -------
        log_ratio - (N,) array
            the log of the ratio of probabilities for use in
            Metropolis-Hastings, or None if all probabilities are equal.
        output - (N,) array of ints
            the candidate C values
        """
        log_ratio = None
        return (log_ratio,
                self.L.C_sample(self.TH.logp, self.N, in_order=False))

    def C_step(self):
        """
        A function which moves the chains between components of the LEMM.
        For each chain a candidate point in the LEMM is generated by first
        choosing a value of C from the C Markov chain, then a random value
        from U_C.
        The chain position is updated using the standard criterion
        for the Metropolis-Hastings algorithm.
        """
        # First generate a random sample for each data vector
        C_ratio, C_samples = self.C_candidates()

        U_samples = self.lead_rv.U_sample(self.N, self.rnd)

        # The proposed step
        X_samples = self.U_to_X(U_samples, C_samples)

        # The current position of the walkers
        X_walkers = self.X_walkers()

        # Now compute the log ratio of the probabilities
        diff = X_walkers - X_samples  # shape (N, n)
        summed = X_samples + X_walkers - 2 * self.X  # ditto
        if self.TH.tied:
            if self.TH.covar_type == 'full':
                diff = diff.dot(self.TH.cv_invchol)
                summed = summed.dot(self.TH.cv_invchol)
            else:
                diff = diff * self.TH.cv_invchol
                summed = summed * self.TH.cv_invchol
        else:
            if self.TH.covar_type == 'full':
                trans_diff = np.zeros_like(diff)
                indexed_A_gemv(self.C, 1.0, self.TH.cv_invchol, diff, 1.0,
                               trans_diff)
                diff = trans_diff
                trans_summed = np.zeros_like(summed)
                indexed_A_gemv(self.C, 1.0, self.TH.cv_invchol, summed, 1.0,
                               trans_summed)
                summed = trans_summed
            elif self.TH.covar_type == 'diagonal':
                diff = diff * self.TH.cv_invchol[self.C]
                summed = summed * self.TH.cv_invchol[self.C]
            else:
                diff = diff * self.TH.cv_invchol[self.C, None]
                summed = summed * self.TH.cv_invchol[self.C, None]

        log_ratio = -1/2 * np.einsum("Nn,Nn->N", diff, summed)
        if C_ratio is not None:
            log_ratio += C_ratio

        # Accept or reject the proposed steps
        accept = self.accept(log_ratio, self.rnd, "C_step_acceptance")

        # Now actually move the Markov chain
        self.update_positions(U_samples, accept)
        self.update_C(C_samples, accept)

    def accept(self, log_ratio, rnd, stats_name):
        r"""
        Takes an array of log ratios of probabilities of a given step, accepts
        each value :math:`l` if it is negative and with probability
        :math:`\exp(-l)` if it is positive.

        Parameters
        ----------
        log_ratio : (\*shape) ndarray
            array of log ratios.
        rnd : np.random.RandomState
            random number generator
        stats_name : string
            name to keep track of acceptance statistics.

        Returns
        -------
        accept : (\*shape) ndarray, dtype=bool
            whether to accept each step or not.
        """
        # This acceptance was faster than the commented version below
        e = rnd.standard_exponential(size=log_ratio.shape)
        accept = e > log_ratio
        # u = rnd.uniform(0, 1, delta.shape)
        # accept = u < np.exp(-log_ratio)

        self.exp_values.record(stats_name, np.sum(accept),
                               np.prod(log_ratio.shape))

        return accept

    def update_positions(self, newU, accept):
        self.U[accept] = newU[accept]
        self._Z_needs_update = True

    def update_C(self, newC, accept):
        C_changed = newC[accept]
        self.C[accept] = C_changed

        b_changed = -(self.MVmVs[C_changed])
        indexed_A_gemv(C_changed, 1.0, self.MVJs,
                       self.XJ[accept], 1.0, b_changed)
        self.b[accept] = b_changed

        self._Z_needs_update = True

    def log_ratio_Uspace(self, U1, U2):
        diff = U1-U2  # shape (N, k)
        quad_term = indexed_A_xAy(self.C, diff, self.MVMVs, U1+U2,
                                  np.zeros((self.N,)))

        return quad_term + np.einsum("Nk,Nk->N", diff, self.b)

    def U_step(self, delta=0.1, step_type='default'):
        """
        A function which moves the Markov chain states within the latent space
        of a single U-component of the LEMM.
        """
        # Get candidate new positions
        newU = self.lead_rv.MCstep(self.U, self.rnd, delta=delta,
                                   type=step_type)

        # Compute the logarithm of the ratio of the relative probabilities
        log_ratio = self.log_ratio_Uspace(self.U, newU)

        # Choose which to accept
        accept = self.accept(log_ratio, self.rnd, "U_step_acceptance")

        # Update the positions accordingly
        self.update_positions(newU, accept)

    def update_quad_forms(self):
        r"""
        The quadratic form on :math:`\mathbb{R}^n` defined for each data point
        :math:`X_i` by

        .. math::
            -\frac12(X_i - X)^t\Sigma^{-1}(X_i-X)

        can be pulled back to the latent space through the expression
        :math:`X = VZ = V(MU+m)`.
        The above quadratic form then has the form

        .. math::
            & U^t(-\frac12(VM)^t\Sigma^{-1}VM)U \\
            & + U(VM)\Sigma^{-1}X^t - U(VM)\Sigma^{-1}(Vm) \\
            & - \frac12X^t\Sigma^{-1}X + X^t\Sigma^{-1}(Vm)
            - \frac12(Vm)^t\Sigma^{-1}(Vm)

        Where the first line is the quadratic term, second the linear term
        and finally the third line the constant term.

        Computing this requires information about the embedding for
        each latent space and needs to be updated for every step the C values
        are updated, or whenever the parameter TH.V is changed.
        """
        # First calculate those terms that do not depend on X
        if self.TH.tied:
            if self.TH.covar_type == 'full':
                VJ = self.TH.V.dot(self.TH.cv_invchol)
            else:
                VJ = self.TH.V * self.TH.cv_invchol

            self.mVJs = self.ms.dot(VJ)  # shape (M, n)
            self.MVJs = self.Ms.dot(VJ)  # shape (M, k, n)
        else:
            if self.TH.covar_type == 'full':
                VJ = np.einsum("mk,Mkn->Mmn", self.TH.V, self.TH.cv_invchol)
            elif self.TH.covar_type == 'diagonal':
                VJ = self.TH.V * self.TH.cv_invchol[:, None, :]
            else:
                VJ = self.TH.V * self.TH.cv_invchol[:, None, None]

            self.mVJs = np.einsum("Mm,Mmn->Mn", self.ms, VJ)  # shape (M, n)
            self.MVJs = np.einsum("Mkm,Mmn->Mkn", self.Ms, VJ)  # shape (M,k,n)

        # -1/2(Vm)^tΣVm, shape (M,)
        # self.mVmVs = -0.5 * np.square(self.mVJs).sum(axis=1)

        # (VM)^tΣVm, shape (M, k)
        self.MVmVs = np.einsum("Mkn,Mn->Mk", self.MVJs, self.mVJs)

        # -1/2(VM)^tΣVM, shape (M, k, k)
        self.MVMVs = -0.5 * np.einsum("Mkn,Mln->Mkl", self.MVJs, self.MVJs)

        # Now calculate those terms that do depend on X, but not on C, the
        # choice of random variable.
        if self.TH.tied:
            if self.TH.covar_type == 'full':
                self.XJ = self.X.dot(self.TH.cv_invchol)
            else:
                self.XJ = self.X * self.TH.cv_invchol
        else:
            if self.TH.covar_type == 'full':
                self.XJ = np.zeros_like(self.X)
                indexed_A_gemv(self.C, 1.0, self.TH.cv_invchol, self.X, 1.0,
                               self.XJ)
            elif self.TH.covar_type == 'diagonal':
                self.XJ = self.X * self.TH.cv_invchol[self.C]
            else:
                self.XJ = self.X * self.TH.cv_invchol[self.C, None]

        # -1/2X^t Σ X, shape (N,)
        # self.c = -0.5 * np.square(XJ).sum(axis=1)

        # Finally compute the terms which do depend on C
        # b = (VM)^tΣ_C(x - Vm_C), shape (N, k)
        self.b = -self.MVmVs[self.C]
        indexed_A_gemv(self.C, 1.0, self.MVJs, self.XJ, 1.0, self.b)
        # self.c = (mVJs[self.C] * XJ).sum(axis=1) - mVmVs[self.C]

    def change_parameters(self, TH):
        self.TH = TH
        self._Z_needs_update = True
        self.update_quad_forms()

    def contribute_q(self):
        if not self.TH.tied:
            self.contribute_q_when_untied()
            return

        ZX_contrib = self.Z().T.dot(self.X)
        self.exp_values.record("ZX", ZX_contrib, self.N)

        ZZ_contrib = self.Z().T.dot(self.Z())
        self.exp_values.record("ZZ", ZZ_contrib, self.N)

        C_contrib = count_C(self.C, self.L.M)
        # C_contrib = np.zeros((self.L.M,))
        # np.add.at(C_contrib, self.C, 1)
        self.exp_values.record("C", C_contrib, self.N)

    def contribute_q_when_untied(self):
        # This is a prime target for optimisation with numba
        M, N, m, n = self.L.M, self.N, self.L.m, self.L.n
        ZX_contrib = np.zeros((M, m, n))
        ZZ_contrib = np.zeros((M, m, m))
        XX_contrib = np.zeros((M, n, n))
        for i in range(self.N):
            ZX_contrib[self.C[i]] += self._Z[i][:, None] * self.X[i]
            ZZ_contrib[self.C[i]] += self._Z[i][:, None] * self._Z[i]
            XX_contrib[self.C[i]] += self.X[i][:, None] * self.X[i]
        self.exp_values.record("ZX", ZX_contrib, N)
        self.exp_values.record("ZZ", ZZ_contrib, N)
        self.exp_values.record("XX", XX_contrib, N)

        C_contrib = count_C(self.C, M)
        self.exp_values.record("C", C_contrib, N)

    def contribute_Z_given_X(self):
        ZgX_contrib = self.Z()
        self.exp_values.record("Z_given_X", ZgX_contrib, 1)

    def estimated_exp_Z_given_X(self):
        Z_given_X = self.exp_values.expected_value("Z_given_X")
        return Z_given_X

    def contribute_C_given_X(self):
        CgX_contrib = np.zeros((self.N, self.L.M))
        CgX_contrib[range(self.N), self.C] = 1
        self.exp_values.record("C_given_X", CgX_contrib, 1)

    def estimated_exp_C_given_X(self):
        C_given_X = self.exp_values.expected_value("C_given_X")
        return C_given_X

    def estimated_expected_q_values(self):
        q = self.exp_values.value["C"]
        qZZ = self.exp_values.value["ZZ"]
        qZX = self.exp_values.value["ZX"]
        if self.TH.tied:
            qXX = self.qXX * self.exp_values.weight["C"]
        else:
            qXX = self.exp_values.value["XX"]

        return q, qZZ, qZX, qXX

    def M_step(self, prop_to_keep=0.0):
        """
        Performs the maximisation step of the EM algorithm, by extracting the
        required expected values previously contributed by the Markov chain.
        """
        # Finish the expectation step
        q, qZZ, qZX, qXX = self.estimated_expected_q_values()

        # Now minimize
        newTH = self.TH.from_minimisation(q, qZZ, qZX, qXX, self.alpha)

        # Update the parameters
        self.change_parameters(newTH)

        # Reset the estimators for the expected values
        for name in ["C", "ZZ", "ZX"]:
            self.exp_values.reweight(name, factor=prop_to_keep)

    def perform(self, regime, prop_to_keep=0.0, step_type='default'):
        """
        Performs a number of actions specified by the string, regime.
        An action is performed for each character:

        * 'U' - the latent variables U are updated,
        * 'C' - the latent variables C are updated,
        * 'q' - a contribution to the q-values is made,
        * 'm' - the parameters are updated by minimising using with the current
          state of the estimation of the q-values.

        Parameters
        ----------
        regime - string
            characters specifying actions
        prop_to_keep - float, optional
            after minimisation the estimator of the q-values is scaled by
            this amount, allowing for persistence between minimisation steps
        step_type - string, optional
            the type of step within the variables U to take

        Returns
        -------
        TH - class of parameters
            the parameters after fitting
        """

        for task in regime:
            if task == 'U':
                self.U_step(step_type=step_type)
            elif task == 'C':
                self.C_step()
            elif task == 'q':
                self.contribute_q()
            elif task == 'm':
                self.M_step(prop_to_keep=prop_to_keep)
            elif task == 'z':
                self.contribute_Z_given_X()
            else:
                raise ValueError(f"Step {task} not recognised")

        return self.TH
