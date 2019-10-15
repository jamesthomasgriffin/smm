from smm.lemm.linearlyembeddedmm import LinearlyEmbeddedMM
from smm.rvs.normalsimplexrv import NormalSimplexRV
import numpy as np


class MPPCA(LinearlyEmbeddedMM):
    """
    A class implementing mixtures of PPCA, described in
    ...

    Parameters
    ----------
    components : integer
        the number of components
    d : integer
        the dimension of components
    n : integer
        the ambient dimension
    rnd : np.random.RandomState
        a choice of random number generator
    """

    def __init__(self, components, d, n, **kwargs):

        m = (d+1) * components
        self.d = d
        self.components = components

        simplices = [tuple(range(i*(d+1), (i+1)*(d+1))) for i in range(components)]

        rvs = [NormalSimplexRV(m, S) for S in simplices]

        LinearlyEmbeddedMM.__init__(self, m, n, rvs, **kwargs)

    def initial_V_from_means(self, means, scale):
        means_shape = (self.components, self.n)
        if means.shape != means_shape:
            raise ValueError(f"means has wrong shape, should be {means_shape}")

        V = np.zeros((self.m, self.n), dtype=means.dtype)
        d = self.d
        for i in range(d+1):
            V[i::d+1, :] = means + \
                scale * self.rnd.standard_normal(size=means.shape)

        return V
