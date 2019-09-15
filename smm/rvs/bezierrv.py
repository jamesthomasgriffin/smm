"""
Implements two random variables associated to Bezier curves.
"""

import numpy as np
from smm.rvs.baserv import BaseRV
from itertools import combinations


class CubicBezierRV(BaseRV):
    r"""
    A class to generate points randomly from a cubic Bezier curve in
    :math:`\mathbb{R}^m`.
    """

    def __init__(self, m, S):
        """
        Initialises the CubicBezierRV class

        Parameters
        ----------
        m : int
            the number of hidden features.
        S : 4-tuple of ints in range(m)
            the control points of the curve.
        """
        BaseRV.__init__(self, m)

        if len(S) != 4:
            raise ValueError("Incorrect number of features")

        self.S = S
        for v in S:
            if v not in range(m):
                raise ValueError("Control point {} not in range".format(v))

        self.k = 4
        M = np.zeros((self.k, m))
        for i, v in enumerate(S):
            M[i, v] = 1.

        self.B = np.array([[1, -3,  3, -1],
                           [0,  3, -6,  3],
                           [0,  0,  3, -3],
                           [0,  0,  0,  1]]).T

        self.M = self.B.dot(M)

        self.mean = np.array([1., 1/2, 1/3, 1/4]).dot(self.M)

        C = np.array([[1,   1/2, 1/3, 1/4],
                      [1/2, 1/3, 1/4, 1/5],
                      [1/3, 1/4, 1/5, 1/6],
                      [1/4, 1/5, 1/6, 1/7]])
        self.covar = self.M.T.dot(C).dot(M) - np.outer(self.mean, self.mean)

        self.Moffset = np.zeros_like(self.mean)

    def __str__(self):
        return str(self.S)

    def __repr__(self):
        return "CubicBezierRV({}, {})".format(self.m, repr(self.S))

    def __eq__(self, other):
        """
        Two Bezier curves are equal if they have the same ambient dimension and
        the same set of vertices, notice that self.S is sorted in __init__.
        """
        return self.m == other.m and self.S == other.S

    def __hash__(self):
        """
        Two Bezier curves hash to the same value if they have the same ambient
        dimension and the same set of vertices, notice that self.S is sorted
        in __init__.
        """
        return hash((self.m, self.S))

    def U_sample(self, d, rnd=np.random.RandomState()):
        """
        Samples uniformly from the interval and then computes the powers up
        to the cubic power.

        Parameters
        ----------
        d : int
            the number of samples.
        rnd : np.random.RandomState
            source for random samples.

        Returns
        -------
        sample : (d, 4) ndarray
            an array of samples.
        """
        t = rnd.random_sample((d,))
        powers_of_t = np.power(t[:, None], np.array([[0, 1, 2, 3]]))
        return powers_of_t

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
        return U.dot(self.M)

    @classmethod
    def MCstep(cls, U, rnd, delta=0.1, type='uniform'):
        """
        Implements a step in a Markov process through the latent space with
        stable distribution the latent variable.

        Parameters
        ----------
        U : (N, k) ndarray
            positions prior to step.
        rnd : np.random.RandomState
            random sampling.
        delta : float
            parameter determining step size.
        type : {'uniform', 'gaussian', 'default'}, optional
            the type of step to use.

        Returns
        -------
        new_U : (N, k) ndarray
            positions after step.
        """
        N = U.shape[0]

        if type == 'uniform' or type == 'default':
            step = rnd.uniform(low=-delta, high=delta, size=(N,))
        elif type == 'gaussian':
            step = rnd.standard_normal(size=(N,)) * delta
        else:
            raise ValueError(f"Unrecognised step type: {type}")

        new_t = np.add(U[:, 1], step, out=step)
        new_t[new_t < 0] += 1
        new_t[new_t > 1] -= 1
        new_U = np.power(new_t[:, None], np.array([[0, 1, 2, 3]]))
        return new_U


class QuadraticBezierRV(BaseRV):
    r"""
    A class to generate points randomly from a quadratic Bezier curve in
    :math:`\mathbb{R}^m`.
    """

    def __init__(self, m, S):
        """
        Initialises the QuadraticBezierRV class

        Parameters
        ----------
        m : int
            the number of hidden features.
        S : 3-tuple of ints in range(m)
            the (start, control point, end) of the curve.
        """
        BaseRV.__init__(self, m)

        if len(S) != 3:
            raise ValueError("Incorrect number of features")
        self.S = S
        for v in S:
            if v not in range(m):
                raise ValueError("Control point {} not in range".format(v))

        self.k = 3
        M = np.zeros((self.k, m))
        for i, v in enumerate(S):
            M[i, v] = 1.

        self.B = np.array([[1, -2,  1],
                           [0,  2, -2],
                           [0,  0,  1]]).T

        self.M = self.B.dot(M)

        self.mean = np.array([1., 1/2, 1/3]).dot(self.M)

        C = np.array([[1,   1/2, 1/3],
                      [1/2, 1/3, 1/4],
                      [1/3, 1/4, 1/5]])
        self.covar = self.M.T.dot(C).dot(M) - np.outer(self.mean, self.mean)

        self.Moffset = np.zeros_like(self.mean)

    def __str__(self):
        return str(self.S)

    def __repr__(self):
        return "QuadraticBezierRV({}, {})".format(self.m, repr(self.S))

    def __eq__(self, other):
        """
        Two Bezier curves are equal if they have the same ambient dimension and
        the same set of vertices, notice that self.S is sorted in __init__.
        """
        return self.m == other.m and self.S == other.S

    def __hash__(self):
        """
        Two Bezier curves hash to the same value if they have the same ambient
        dimension and the same set of vertices, notice that self.S is sorted
        in __init__.
        """
        return hash((self.m, self.S))

    def U_sample(self, d, rnd):
        """
        Samples from a Bezier curve with control points 0,1,2,3.

        Parameters
        ----------
        d : int
            the number of samples.
        rnd : np.random.RandomState
            random sampling.

        Returns
        -------
        sample : (d, 3) ndarray
            an array of samples.
        """
        t = rnd.random_sample((d,))
        powers_of_t = np.power(t[:, None], np.array([[0, 1, 2]]))
        return powers_of_t

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
        return U.dot(self.M)

    @classmethod
    def MCstep(cls, U, rnd, delta=0.1, type='uniform'):
        """
        Implements a step in a Markov process through the latent space with
        stable distribution the latent variable.

        Parameters
        ----------
        U : (N, k) ndarray
            positions prior to step.
        rnd : np.random.RandomState
            random sampling.
        delta : float
            parameter determining step size.
        type : {'uniform', 'gaussian', 'default'}, optional
            the type of step to use.

        Returns
        -------
        new_U : (N, k) ndarray
            positions after step.
        """
        N = U.shape[0]
        if type == 'uniform' or type == 'default':
            step = rnd.uniform(low=-delta, high=delta, size=(N,))
        elif type == 'gaussian':
            step = rnd.standard_normal(size=(N,)) * delta
        else:
            raise ValueError(f"Unrecognised step type: {type}")

        new_t = np.add(U[:, 1], step, out=step)
        new_t[new_t < 0] += 1
        new_t[new_t > 1] -= 1
        new_U = np.power(new_t[:, None], np.array([[0, 1, 2]]))
        return new_U


def complete_Bezier_graph(k, type="quadratic", V=None,
                          rnd=np.random.RandomState()):
    """
    Constructs a list of random variables representing the edges of a complete
    graph, also returning the number of features m needed to parameterise the
    graph.

    Parameters
    ----------
    k : int
        the number of vertices in the graph.
    type : {"quadratic", "cubic"}, optional
        the degree of the Bezier curves.
    V : (k, n) ndarray or None
        if provided this function will return positions for the control points
    rnd : np.random.RandomState, optional
        random number generator passed on to the simplices.

    Returns
    -------
    rvs : list of random variables
        the output random variables.
    m : int
        number of features.
    W : (m, n) ndarray or None
        if V was provided this contains feature positions derived from V
    """
    verts = list(range(k))  # list of vertices
    pairs = list(combinations(verts, 2))  # list of edges
    e = len(pairs)  # Number of edges in complete graph

    if type == 'quadratic':
        # dictionary to assign control point to an edge
        C = {p: i + k for i, p in enumerate(pairs)}
        m = k + e  # total number of features
    if type == 'cubic':
        # dictionaries to assign control points to an edge
        C1 = {p: i + k for i, p in enumerate(pairs)}
        C2 = {p: i + k + e for i, p in enumerate(pairs)}
        m = k + 2*e  # total number of features

    rvs = []
    if V is not None:
        assert V.shape[0] == k, \
            "Dimension mismatch, {}, {}".format(V.shape[0], k)
        W = np.zeros((m, V.shape[1]))
        W[:k] = V
    else:
        W = None
    for p in pairs:
        if type == 'quadratic':
            S = (p[0], C[p], p[1])
            if V is not None:
                W[C[p]] = (V[p[0]] + V[p[1]]) / 2
            rvs.append(QuadraticBezierRV(m, S))
        if type == 'cubic':
            S = (p[0], C1[p], C2[p], p[1])
            if V is not None:
                W[C1[p]] = V[p[0]]
                W[C2[p]] = V[p[1]]
            rvs.append(CubicBezierRV(m, S))
    return rvs, m, W


__all__ = ["CubicBezierRV", "QuadraticBezierRV", "complete_Bezier_graph"]
