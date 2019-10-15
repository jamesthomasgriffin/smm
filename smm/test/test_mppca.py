import numpy as np
from ..lemm import (MPPCA,
                    GLEMM_Parameters)


class TestMPPCA:
    def test_mppca(self):
        c = 3
        n = 2
        d = 2
        rnd = np.random.RandomState(123)
        L = MPPCA(c, d, n, rnd=rnd)
        assert L.d == 2
        assert L.components == 3
        assert L.M == 3
        assert L.m == 9

        means = np.array([[0, 1], [1, 0], [1, 1]])
        V = L.initial_V_from_means(means, 0.1)
        TH = GLEMM_Parameters(V, L.M, None, 'spherical', 0.1)

        L.mean_and_covar(TH)

        L.top_components(TH)
        L.entropy_of_distribution(np.array([0.5, 0.5]), log_given=False)

        X = np.random.standard_normal(size=(10, 2))
        L.log_likelihood(TH, X)
        L.expected_C_and_Z_given_X(TH, X)
        L.expected_Z_given_X(TH, X)
        L.expected_C_given_X(TH, X)
