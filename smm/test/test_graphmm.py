import numpy as np
from ..lemm import (GraphMM,
                    GLEMM_Parameters)


class TestGraphMM:
    def test_graphmm(self):
        m = 3
        n = 2
        rnd = np.random.RandomState(123)
        L = GraphMM(m, n, rnd=rnd)
        assert L.m == 3
        assert L.n == 2
        assert L.M == 3

        L = GraphMM(m, n, include_nodes=True, rnd=rnd)
        assert L.M == 6

        V = np.array([[0, 1], [1, 0], [1, 1]])
        TH = GLEMM_Parameters(V, L.M, None, 'spherical', 0.1)

        L.mean_and_covar(TH)

        L.top_components(TH)
        L.entropy_of_distribution(np.array([0.5, 0.5]), log_given=False)
        L.number_to_keep(TH.logp)

        X = np.random.standard_normal(size=(10, 2))
        L.log_likelihood(TH, X)
        L.expected_C_and_Z_given_X(TH, X)
        L.expected_Z_given_X(TH, X)
        L.expected_C_given_X(TH, X)
