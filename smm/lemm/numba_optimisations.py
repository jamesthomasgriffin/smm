from numba import guvectorize, float64, int64, njit
import numpy as np
from smm import numba_target as target_preset


@guvectorize([(int64, float64, float64[:, :], float64[:])],
             '(),(),(M,n)->(n)',
             nopython=True,
             target=target_preset)
def indexed_x_axpy(ix, a, x, res):  # pragma: no cover
    """Computes y <-- ax + y, but where x is indexed, i.e. it is
    chosen by the index, ix.  So it is equivalent to inplace a*x[ix,:]+y.

    N.B. y cannot be indexed because this is an inplace operation."""
    for i in range(res.shape[0]):
        res[i] += a * x[ix, i]


@guvectorize([(int64, float64, float64[:, :, :], float64[:], float64,
               float64[:])],
             '(),(),(M,m,n),(n),()->(m)',
             nopython=True,
             target=target_preset)
def indexed_A_gemv(ix, a, A, x, b, res):  # pragma: no cover
    """Computes y <-- aAx + by, where A is indexed."""
    for i in range(res.shape[0]):
        res[i] *= b
        for j in range(x.shape[0]):
            res[i] += a * A[ix, i, j] * x[j]


@guvectorize([(int64, float64[:], float64[:, :, :], float64[:, :],
               float64[:])],
             '(),(m),(M,m,n),(M,n)->(n)',
             nopython=True,
             target=target_preset)
def indexed_Ab_xApb(ix, x, A, b, res):  # pragma: no cover
    """Computes y <-- xA + b, where A and b are indexed."""
    for i in range(res.shape[0]):
        res[i] = b[ix, i]
        for j in range(x.shape[0]):
            res[i] += x[j] * A[ix, j, i]


@guvectorize([(int64, float64[:], float64[:, :, :], float64[:], float64[:])],
             '(),(n),(M,n,m),(m)->()',
             nopython=True,
             target=target_preset)
def indexed_A_xAy(ix, x, A, y, res):  # pragma: no cover
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            res[0] += x[i] * A[ix, i, j] * y[j]


@njit
def count_C(C, M):  # pragma: no cover
    C_total = np.zeros((M,))
    N = C.shape[0]
    for i in range(N):
        C_total[C[i]] += 1
    return C_total


@njit
def sum_per_C(C, M, P, out):  # pragma: no cover
    N = C.shape[0]
    for i in range(N):
        out[C[i]] += P[i]
    return out
