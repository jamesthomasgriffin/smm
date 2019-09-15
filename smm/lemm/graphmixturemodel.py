from smm.lemm.linearlyembeddedmm import LinearlyEmbeddedMM
from smm.rvs.edgerv import EdgeRV
from smm.rvs.pointrv import PointRV
from smm.rvs.basesimplexrv import k_dim_degenerate_l_simplices


class GraphMM(LinearlyEmbeddedMM):
    """
    A class for a simplicial mixture model with only 0- and 1-simplices, i.e.
    only points and lines.

    Parameters
    ----------
    m : integer
        the number of feature vectors
    n : integer
        the ambient dimension
    include_nodes : bool, optional
        whether to include 0-simplices (nodes/vertices of the graph)
    rnd : np.random.RandomState
        a choice of random number generator
    """

    def __init__(self, m, n, include_nodes=False, **kw_args):

        edges = k_dim_degenerate_l_simplices(1, [1], m)
        rvs = [EdgeRV(m, S) for S in edges]

        if include_nodes:
            rvs = rvs + [PointRV(m, v) for v in range(m)]

        LinearlyEmbeddedMM.__init__(self, m, n, rvs, **kw_args)
