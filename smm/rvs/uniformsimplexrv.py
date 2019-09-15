"""
Implements the uniform random variable associated to a simplex.
It isn't actually uniform for degenerate simplices.
"""

import numpy as np
from smm.rvs.basesimplexrv import BaseSimplexRV


class UniformSimplexRV(BaseSimplexRV):
    r"""
    A class to generate points randomly from a simplex in :math:`\mathbb{R}^m`.
    """

    def __init__(self, m, S):
        """
        Initialises the UniformSimplexRV class

        Parameters
        ----------
        m : int
            the number of hidden features.
        S : list of elements in range(m)
            the vertices of the simplex.
        """
        BaseSimplexRV.__init__(self, m, S)
        self.k = len(self.S) - 1

        # Setup defining matrix
        self.Moffset = np.zeros((self.m,))
        self.Moffset[S[-1]] = 1
        self.M = np.zeros((self.k, m))
        for i, v in enumerate(S):
            if i < self.k:
                self.M[i, v] += 1
            if i > 0:
                self.M[i-1, v] += -1

    def U_sample(self, d, rnd):
        r"""
        Samples from the simplex :math:`\{0 < x_1 < \cdots < x_k < 1\}` by
        first choosing :math:`x_i` uniformly from the unit interval [0, 1]
        and then sorting the sequence.

        Parameters
        ----------
        d : int
            the number of samples.
        rnd : np.random.RandomState
            random sampling.

        Returns
        -------
        sample : (d, r) ndarray
            an array of samples, here r is the number of vertices of self.S.
        """
        return np.sort(rnd.random_sample((d, self.k)), axis=-1)

    def U_to_Z(self, U):
        """
        Converts the latent variable U to the feature variable Z.

        Parameters
        ----------
        U : (N, k) ndarray
            input.

        Returns
        -------
        Z : (N, m) ndarray
            output.
        """
        return U.dot(self.M) + self.Moffset

    def log_PX(self, X):
        constant = -np.log(1)
        return np.zeros([1]*len(X.shape)) + constant

    @classmethod
    def MCstep(cls, U, rnd, delta=0.1, type='gaussian'):
        """
        Implements a step in a Markov process through the latent space with
        stable distribution the latent variable.

        One may specify the size of the step and the type, if 'default' is
        given we use a Gaussian step.

        Parameters
        ----------
        U : (N, k) ndarray
            positions prior to step.
        rnd : np.random.RandomState
            random sampling.
        delta : float
            the size of the step.
        type : {'uniform', 'gaussian', 'default'}, optional
            the type of step to use.

        Returns
        -------
        newU : (N, k) ndarray
            positions after step.
        """
        if type == 'uniform' or type == 'default':
            step = rnd.uniform(low=-delta, high=delta, size=U.shape)
        elif type == 'gaussian':
            step = rnd.standard_normal(size=U.shape) * delta
        else:
            raise ValueError(f"Unrecognised step type: {type}")

        newU = np.add(U, step, out=step)
        newU[newU < 0] += 1
        newU[newU > 1] -= 1
        newU.sort(axis=1)
        return newU

    def diff_entropy_U(self):
        """
        The logarithm of the volume, V = 1 / k!, of the simplex
        0 < u_1 < ... < u_k < 1
        """
        return -np.sum(np.log(np.array(range(2, self.k+1))))

    def diff_entropy_Z(self):
        """
        The logarithm of the volume, V = sqrt(k+1) / k!, of the probability
        simplex.

        Note that this is incorrect for the degenerate case.
        """
        return 1/2 * np.log(self.k+1) + self.diff_entropy_U()


__all__ = ["UniformSimplexRV"]
