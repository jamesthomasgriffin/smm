import numpy as np
from ..lemm import (SimplicialMM,
                    GLEMM_Parameters)


class TestSimplicialMM:
    def test_simplicialmm(self):
        m = 4
        n = 2
        rnd = np.random.RandomState(123)
        k = 2
        L = SimplicialMM(m, n, k, rnd=rnd)
        assert L.m == 4
        assert L.n == 2
        assert L.M == 4 + 12 + 4

        V = np.array([[0, 1], [1, 0], [1, 1], [1, -1]])
        TH = GLEMM_Parameters(V, L.M, None, 'spherical', 0.1)

        L.mean_and_covar(TH)

        L.top_components(TH)
        L.entropy_of_distribution(np.array([0.5, 0.5]), log_given=False)
        L.encoding_rate(TH)
        L.number_to_keep(TH.logp)
