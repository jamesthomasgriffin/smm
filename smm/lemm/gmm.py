from smm.lemm.linearlyembeddedmm import LinearlyEmbeddedMM
from smm.rvs.pointrv import PointRV


class GMM(LinearlyEmbeddedMM):
    """
    A class implementing Gaussian mixture models.  Note that this is not an
    efficient implementation; the formulae for solving the maximisation problem
    are much simpler in the GMM case and this code does not take advantage of
    that.

    Parameters
    ----------
    components : integer
        the number of components
    n : integer
        the ambient dimension
    rnd : np.random.RandomState
        a choice of random number generator
    """

    def __init__(self, components, n, **kwargs):

        self.components = components

        rvs = [PointRV(components, v) for v in range(components)]

        LinearlyEmbeddedMM.__init__(self, components, n, rvs, **kwargs)
