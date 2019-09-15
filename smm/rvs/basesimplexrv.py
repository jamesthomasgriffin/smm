"""
Implements the base class for random variables with simplex like structures.
"""

import numpy as np
from smm.rvs.baserv import BaseRV
from itertools import combinations_with_replacement


class BaseSimplexRV(BaseRV):
    r"""
    A class to generate points randomly from a simplex in :math:`\mathbb{R}^m`.
    """

    def __init__(self, m, S):
        """
        Initialises the BaseSimplexRV class

        Parameters
        ----------
        m : int
            the number of hidden features.
        S : list of elements in range(m)
            the vertices of the simplex.
        """
        BaseRV.__init__(self, m)
        self.S = sorted(S)
        self.support = set(S)
        self.degree_sequence = [-1] * self.m
        for v in S:
            if v not in range(m):
                raise ValueError("Simplex vertex {} not in range".format(v))
            self.degree_sequence[v] += 1

        # Formula for mean
        self.mean = np.zeros((m,))
        for i, v in enumerate(S):
            self.mean[v] += 1 / len(S)

        # Formula for covariance matrix
        self.covar = np.zeros((self.m, self.m))
        for v in S:
            for w in S:
                self.covar[v, w] = -1.0 / (len(S)**3)
            self.covar[v, v] += 1.0 / (len(S)**2)

    def __str__(self):
        return str(self.S)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.m,
                                   repr(self.S))

    def __eq__(self, other):
        """
        Two simplices are equal if they have the same ambient dimension and
        the same set of vertices, notice that self.S is sorted in __init__.
        """
        return self.m == other.m and self.S == other.S

    def __hash__(self):
        """
        Two simplices hash to the same value if they have the same ambient
        dimension and the same set of vertices, notice that self.S is sorted
        in __init__.
        """
        return hash((self.m, self.S))

    def __lt__(self, other):
        if not self.m == other.m:
            return False
        for a, b in zip(self.degree_sequence, other.degree_sequence):
            if b > a:
                return False
        return True

    def __contains__(self, item):
        return item in self.S


def k_dim_degenerate_l_simplices(k, support_dims, m, **kw_args):
    """
    Creates a list of tuples of vertices consisting of all k simplices whose
    support dimension lies in the given list.

    Parameters
    ----------
    k : int
        the dimension of the simplices to generate.
    support_dims : list of ints
        the possible dimensions of the supports.
    m : int
        the number of vertices.

    Returns:
    rvs : list of tuples of vertices
    """

    return [S for S in combinations_with_replacement(range(m), k+1)
            if len(set(S))-1 in support_dims]


__all__ = ["BaseSimplexRV", "k_dim_degenerate_l_simplices"]
