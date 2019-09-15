"""
A few helpful functions that are used across the project.
"""

import numpy as np


def invsqrtS_logdetS_from_S(S, n=None):
    """
    Given a positive definite covariance S, determine its type from its shape
    and then compute the inverse square root and logarithm of the determinant.

    To be precise we are not really taking the square root, we take the
    inverse of the upper-triangular Cholesky factor.

    Parameters
    ----------
    S : {(n, n) ndarray, (n,) ndarray, float}
        covariance in either matrix form, diagonal form, or spherical form.
    n : int, optional
        ambient dimension, required if S is a float

    Returns
    -------
    invsqrtS : S.shape ndarray
        inverse of the square root of the covariance in either matrix form,
        diagonal form, or spherical form.
    logdetS : float
        the logarithm of the determinant of S
    """
    if len(np.shape(S)) == 2:  # full matrix case
        sqrtS = np.linalg.cholesky(S)
        invsqrtS = np.linalg.inv(sqrtS)
        logdetS = np.sum(np.log(np.diag(sqrtS)))
    if len(np.shape(S)) <= 1:  # spherical or diagonal case
        invsqrtS = 1. / np.sqrt(S)
        if len(np.shape(S)) == 0:  # spherical case
            assert n is not None, \
                "For spherical covariance, need ambient dimension."
            logdetS = n * np.log(S)
        else:  # diagonal case
            logdetS = np.sum(np.log(S))

    return invsqrtS, logdetS


def shuffle_arrays(arrs, rnd=np.random.RandomState()):
    """
    Simultaneously shuffle the given arrays in the first axis.
    The shuffles should be the same if the first axis dimensions are equal.

    Parameters
    ----------
    arrs : list of ndarrays
        the arrays to be shuffled.
    rnd : np.random.RandomState, optional
        random sampling.
    """
    shuffle = np.arange(arrs[0].shape[0])
    rnd.shuffle(shuffle)
    for arr in arrs:
        arr[:] = arr[shuffle]


def initialise_V(X, m, n_iter=100, n_repl=100, rnd=np.random.RandomState()):
    """
    Initialise vertex positions randomly, spacing out the points.
    """
    # Start with a random selection of points from X
    N = X.shape[0]
    V = X[rnd.choice(N, m, replace=False)].copy()

    # Each iteration change a vertex chosen at random
    for v_to_change in rnd.randint(m, size=(n_iter,)):
        # Move that vertex far away so it wont bother us
        V[v_to_change] = 1e10

        # Choose a random selection of X-values of length n_repl
        X_to_try = X[rnd.choice(N, n_repl, replace=False)]

        # Choose the one that has the maximum distance from the current set
        dist_sq = np.sum(np.square(V[:, None, :]-X_to_try), axis=2)
        rating = np.min(dist_sq, axis=0)
        v_to_use = np.argmax(rating)
        V[v_to_change] = X_to_try[v_to_use]
    return V


def logsumexp(a, axis=None, keepdims=False):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2, scipy.logsumexp

    Notes
    -----
    A simplified version of the scipy function

    """
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out
