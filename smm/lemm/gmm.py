from smm.lemm.linearlyembeddedmm import LinearlyEmbeddedMM
from smm.rvs.pointrv import PointRV
import numpy as np


class GMM(LinearlyEmbeddedMM):
    """
    A class implementing Gaussian mixture models.

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
