from smm.lemm.linearlyembeddedmm import LinearlyEmbeddedMM
from smm.rvs.uniformsimplexrv import UniformSimplexRV
from smm.rvs.basesimplexrv import k_dim_degenerate_l_simplices


class SimplicialMM(LinearlyEmbeddedMM):
    """
    A class for a simplicial mixture model, implementing sampling etc.
    Assumes a full set of k-simplices, with an optional restriction on the
    dimensions of the support simplices.

    Parameters
    ----------
    m : integer
        the number of feature vectors
    n : integer
        the ambient dimension
    k : integer
        the latent dimension of the simplices
    support_dims : list of integers, optional
        the dimensions of the supports of simplices to allow
    rnd : np.random.RandomState
        a choice of random number generator
    """

    def __init__(self, m, n, k, support_dims=None, **kwargs):
        if support_dims is None:
            support_dims = list(range(k+1))

        self.k = k

        simplices = k_dim_degenerate_l_simplices(k, support_dims, m)

        rvs = [UniformSimplexRV(m, S) for S in simplices]

        LinearlyEmbeddedMM.__init__(self, m, n, rvs, **kwargs)
