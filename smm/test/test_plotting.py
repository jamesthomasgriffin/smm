# import numpy.testing as npt
# import pytest

import numpy as np
from ..plotting import dist_colours
from ..lemm import LinearlyEmbeddedMM, GLEMM_Parameters
from ..rvs import UniformSimplexRV


class TestPlotting:
    def test_plotting(self):
        m, n = 3, 2
        simplices = [(0, 1), (1, 2)]
        rvs = [UniformSimplexRV(m, S) for S in simplices]
        L = LinearlyEmbeddedMM(m, n, rvs)

        TH = GLEMM_Parameters(np.random.uniform(size=(m, n)), L.M, None,
                              'spherical', 0.1)

        N = 100
        Y, indices = L.noisy_sample(TH, N, in_order=True)

        colours = dist_colours(indices)
        assert len(colours) == N
        for col in colours:
            assert len(col) == 4
            assert col[3] == 1.0

        colours = dist_colours(indices, cm='viridis', alpha=0.5)
        assert len(colours) == N
        for col in colours:
            assert len(col) == 4
            assert col[3] == 0.5
