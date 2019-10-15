import numpy as np
from smm.helpfulfunctions import logsumexp


class LEMM_Parameters:
    """
    This class holds the parameters of a linearly embedded mixture model.

    Parameters
    ----------
        V - (m, n) ndarray
            matrix defining embedding
        M - int
            the number of components
        logp - (M,) array, optional
            probability distribution defining mixture
    """

    def __init__(self, V, M, logp=None,
                 gaussian=None, covar_type=None, covar=None):
        self.m, self.n = V.shape
        self.M = M
        self.V = V

        if logp is None:
            self.logp = np.zeros(M) - np.log(M)
        else:
            if np.shape(logp) != (M,):
                raise ValueError("""The provided array of log probabilities
                                 is the wrong shape""")
            self.logp = logp - logsumexp(logp)

        self.gaussian = False

    def apply_Dirichlet_prior(self, evidence, alpha):
        """
        Update self.logp using a symmetric Dirichlet distribution with
        parameter alpha as a prior, given some evidence.

        N.B. Passing a properly shaped alpha allows for non-symmetric
        Dirichlet distributions.

        Parameters
        ----------
            evidence - (M,) array
                the evidence
            alpha - float
                the prior

        Returns
        -------
            self
        """
        logp = np.log(evidence + alpha)
        logp -= logsumexp(logp)
        self.logp = logp
        return self


class GLEMM_Parameters(LEMM_Parameters):
    """
    This class holds the parameters of a linearly embedded mixture model
    with a single Gaussian kernel and computes relevant quantities associated
    to the covariance parameter.

    Parameters
    ----------
        V - (m, n) ndarray
            matrix defining embedding
        M - int
            the number of components
        logp - (M,) array, optional
            probability distribution defining mixture
        covar_type - {"spherical", "diagonal", "full"}
            the type of covariance matrix or matrices
        covar - ndarray
            the covariance matrix or matrices
    """

    def __init__(self, V, M, logp=None, covar_type=None, covar=None):
        LEMM_Parameters.__init__(self, V, M, logp)
        self.gaussian = True
        self.tied = True
        self.set_Gaussian_terms(covar_type, covar)

    def set_Gaussian_terms(self, covar_type, covar):
        """
        Parameters
        ----------
            covar_type - {"spherical", "diagonal", "full"}
                the type of covariance matrix
            covar - ndarray
                the covariance matrix/vector/float
        """

        if covar_type not in ('spherical', 'diagonal', 'full'):
            raise ValueError("Covariance type not recognised")

        self.covar_type = covar_type

        covar_shape = {
            'spherical': (),
            'diagonal': (self.n,),
            'full': (self.n, self.n)
        }[covar_type]

        if np.shape(covar) == covar_shape:
            self.covar = covar
        else:
            raise ValueError(f"Expected covar shape: {covar_shape}")

        self.process_covar()

    def process_covar(self):
        """
        Calculate various constants associated to the covariance(s) for future
        use.  These are

        * self.cv_chol - the upper triangular Cholesky matrix of self.covar
        * self.cv_invchol - the inverse of the above
        * self.cv_logdet - the logarithm of the determinant of covar
        """

        if self.covar_type == 'full':
            self.cv_chol = np.linalg.cholesky(self.covar)
            self.cv_invchol = np.linalg.inv(
                self.cv_chol + 1e-10*np.eye(self.n))
            self.cv_logdet = 2.0*np.log(np.linalg.det(self.cv_chol))
            # The below code would be more efficient but taking diagonals when
            # the covariances are tied isn't supported by diagonal
            # self.cv_logdet = 2.0*np.sum(
            #     np.log(np.diagonal(self.covar)), axis=-1)
        else:
            self.cv_chol = np.sqrt(self.covar + 1e-20)
            self.cv_invchol = 1. / (self.cv_chol)
            if self.covar_type == 'spherical':
                self.cv_logdet = self.n * np.log(self.covar + 1e-20)
            if self.covar_type == 'diagonal':
                self.cv_logdet = np.sum(np.log(self.covar + 1e-20), axis=-1)

    def untie(self):
        """
        Converts a set of parameters with tied covariance values to a set of
        parameters with untied parameters.
        """
        if not self.tied:
            return self

        if self.covar_type == 'spherical':
            new_covar = self.covar + np.zeros(self.M)
        elif self.covar_type == 'diagonal':
            new_covar = np.ones((self.M, 1)) * self.covar
        else:
            new_covar = np.ones((self.M, 1, 1)) * self.covar

        return GLEMM_Parameters_Untied(self.V, self.M, self.logp,
                                       self.covar_type, new_covar)

    def relax_type(self, new_type):
        """
        Relaxing the type of the covariance does not change the Gaussian
        kernels but allows more freedom in future.
        """
        if new_type == self.covar_type:  # Nothing to do
            return self
        if new_type not in ("diagonal", "full"):
            raise ValueError(f"Cannot relax to type: {new_type}")
        if new_type == "diagonal" and self.covar_type == "full":
            raise ValueError(f"Cannot relax to type: {new_type}")

        if new_type == "full" and self.covar_type == "spherical":
            if self.tied:
                new_covar = self.covar * np.eye(self.n)
            else:
                new_covar = self.covar[..., None, None] * np.eye(self.n)
            # return self.relax_type("diagonal").relax_type("full")  # Slower
        elif new_type == "diagonal":  # self.covar_type must be "spherical"
            if self.tied:
                new_covar = self.covar * np.ones(self.n)
            else:
                new_covar = self.covar[:, None] * np.ones((1, self.n))
        else:  # new_type must be "full" and self.covar_type must be "diagonal"
            new_covar = self.covar[..., None] * np.eye(self.n)

        self.set_Gaussian_terms(new_type, new_covar)
        return self

    def restrict_type(self, new_type):
        """
        Restricting the type involves changing the kernel, though relaxing and
        then restricting back does not change the kernel.
        """
        if new_type == self.covar_type:  # Nothing to do
            return self
        if new_type not in ("diagonal", "spherical"):
            raise ValueError(f"Cannot restrict to type: {new_type}")
        if new_type == "diagonal" and self.covar_type == "spherical":
            raise ValueError(f"Cannot restrict to type: {new_type}")

        if new_type == "spherical" and self.covar_type == "full":
            new_covar = np.trace(self.covar, axis1=-2, axis2=-1) / self.n
            # return self.restrict_type("diagonal").restrict_type("spherical")
        elif new_type == "diagonal":  # self.covar_type must be "full"
            new_covar = np.diagonal(self.covar, axis1=-2, axis2=-1)
        else:  # new_type must be "spherical" and old must be "diagonal"
            new_covar = np.mean(self.covar, axis=-1)

        self.set_Gaussian_terms(new_type, new_covar)
        return self

    def from_minimisation(self, q, qZZ, qZX, qXX, alpha=None):
        """
        Find the parameters which minimize the quantity Q defined by q, qZZ,
        qZX and qXX.
        """

        # This allows the expectation values per value of C to be passed
        if len(qZX.shape) == 3:
            qZX = qZX.sum(axis=0)
        if len(qZZ.shape) == 3:
            qZZ = qZZ.sum(axis=0)
        if len(qXX.shape) == 3:
            qXX = qXX.sum(axis=0)

        weight = np.sum(q)
        for Q in [qZX, qZZ, qXX]:
            Q /= weight

        # if self.using_V_regularisation:
        #     if np.shape(self.V_invcov) == ():
        #         A = qZZ + self.V_invcov * np.eye(self.m)
        #         B = qZX + self.V_mean * self.V_invcov
        #         newV = np.linalg.lstsq(A, B, rcond=None)[0]
        #     else:
        #         A = qZZ[:, None, :, None] * np.eye(self.m)[None, :, None, :]
        #         A += self.V_inv_cov
        #         B = qZX + np.sum(
        #             self.V_mean * self.V_invcov, axis=(2, 3))
        #         newV = self.solve_tensor(A, B)
        # else:
        #     newV = np.linalg.lstsq(qZZ, qZX, rcond=None)[0]

        new_V = np.linalg.lstsq(qZZ, qZX, rcond=None)[0]

        qVZZV = qZZ.dot(new_V).T.dot(new_V)
        qVZX = new_V.T.dot(qZX)
        new_covar = (qVZZV + qXX - qVZX - qVZX.T)

        if self.covar_type == 'spherical':
            new_covar = new_covar.trace() / self.n
        if self.covar_type == 'diagonal':
            new_covar = new_covar.diagonal()

        cls = self.__class__
        if alpha is None:
            other = cls(new_V, self.M, np.log(q + 1e-24),
                        self.covar_type, new_covar)
        else:
            other = cls(new_V, self.M, None, self.covar_type, new_covar)
            other.apply_Dirichlet_prior(q, alpha)
        return other


class GLEMM_Parameters_Untied(GLEMM_Parameters):
    """
    This class holds the parameters of a linearly embedded mixture model
    with a Gaussian kernel for each component and computes relevant quantities
    associated to the covariances.

    Parameters
    ----------
        V - (m, n) ndarray
            matrix defining embedding
        M - int
            the number of components
        logp - (M,) array, optional
            probability distribution defining mixture
        covar_type - {"spherical", "diagonal", "full"}
            the type of covariance matrix or matrices
        covar - (M, ?) ndarray
            the covariance matrices
    """

    def __init__(self, V, M, logp=None, covar_type=None, covar=None):
        LEMM_Parameters.__init__(self, V, M, logp)
        self.gaussian = True
        self.tied = False
        self.set_Gaussian_terms(covar_type, covar)

        # This is how many iterations are to be used when minimizing
        self.untied_minimisation_iterations = 5

    def set_Gaussian_terms(self, covar_type, covar):
        """
        Parameters
        ----------
            covar_type - {"spherical", "diagonal", "full"}
                the type of covariance matrix
            covar - ndarray
                the covariance matrix/vector/float
        """

        if covar_type not in ('spherical', 'diagonal', 'full'):
            raise ValueError("Covariance type not recognised")

        self.covar_type = covar_type

        individual_covar_shape = {
            'spherical': (),
            'diagonal': (self.n,),
            'full': (self.n, self.n)
        }[covar_type]
        covar_shape = (self.M, *individual_covar_shape)

        if np.shape(covar) == covar_shape:
            self.covar = covar
        else:
            raise ValueError(f"Expected covar shape: {covar_shape}")

        self.process_covar()

    def from_minimisation(self, q, qZZ, qZX, qXX, alpha=None):
        """
        Find the parameters which minimize the quantity Q defined by q, qZZ,
        qZX and qXX.
        """
        weight = np.sum(q)
        for Q in [qZX, qZZ, qXX]:
            Q /= weight
        norm_q = q / weight

        new_V = self.V
        for _ in range(self.untied_minimisation_iterations):
            new_covar = self._untied_cv_from_V(new_V, norm_q, qZZ, qZX, qXX)
            new_V = self._V_from_untied_cv(new_covar, qZZ, qZX)

        cls = self.__class__
        if alpha is None:
            other = cls(new_V, self.M, np.log(q), self.covar_type, new_covar)
        else:
            other = cls(new_V, self.M, None, self.covar_type, new_covar)
            other.apply_Dirichlet_prior(q, alpha)
        return other

    def _untied_cv_from_V(self, V, q, qZZ, qZX, qXX):
        n = qZX.shape[2]
        qVZZV = qZZ.dot(V).swapaxes(1, 2).dot(V)
        qXZV = qZX.swapaxes(1, 2).dot(V)
        cv = (qVZZV + qXX - qXZV - qXZV.swapaxes(1, 2))

        # If the probability is too small we get numerical issues when
        # computing cv / q
        prob_cases = q < 1e-20
        cv[prob_cases] = np.eye(n)[None, :, :]
        modified_q = q.copy()
        modified_q[prob_cases] = 1.
        cv /= modified_q[:, None, None]

        # NB in these cases our computation has been very inefficient
        # TODO recode these cases
        if self.covar_type == 'spherical':
            cv = cv.trace(axis1=1, axis2=2) / n
        if self.covar_type == 'diagonal':
            cv = cv.diagonal(axis1=1, axis2=2)
        return cv

    def _V_from_untied_cv(self, cv, qZZ, qZX):
        _, m, n = qZX.shape
        # TODO, implement shortened calculation for spherical and
        # diagonal cases
        if self.covar_type == 'spherical':
            inv = 1/cv[:, None, None] * np.eye(n)[None, :, :]
        if self.covar_type == 'diagonal':
            inv = 1/cv[:, :, None] * np.eye(n)[None, :, :]
        if self.covar_type == 'full':
            inv = np.linalg.inv(cv)

        # A has shape (m,n,m,n)
        A = qZZ.T.dot(inv.swapaxes(0, 1)).swapaxes(1, 2)
        # B has shape (m,n)
        B = np.einsum("ijk,ilk->jl", qZX, inv)
        # B = np.sum(qZX[:, :, None, :] * inv[:, None, :, :], axis=(0, 3))

        # if self.using_V_regularisation:
        #     if np.shape(self.V_invcov) == ():
        #         A += self.V_invcov * np.eye(m*n).reshape((m, n, m, n))
        #     else:
        #         A += self.V_invcov
        #     B += self.V_mean

        newV = self.solve_tensor(A, B)
        return newV

    @staticmethod
    def solve_tensor(A, B):
        m, n = B.shape
        A = A.reshape((m*n, m*n))
        B = B.reshape((m*n,))
        V, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return V.reshape((m, n))
