import numpy as np
from ..lemm import (GMM,
                    GLEMM_Parameters,
                    GLEMM_Parameters_Untied)


class TestGMM:
    def test_gmm(self):
        c = 3
        n = 2
        rnd = np.random.RandomState(123)
        L = GMM(c, n, rnd=rnd)
        assert L.components == 3
        assert L.M == 3
        assert L.m == 3

        means = np.array([[0, 1], [1, 0], [1, 1]])
        TH = GLEMM_Parameters(means, L.M, None, 'spherical', 0.1)
        GLEMM_Parameters_Untied(means, L.M,
                                covar_type='diagonal',
                                covar=np.zeros((L.M, L.n))+0.1)

        L.mean_and_covar(TH)

        L.top_components(TH)
        L.entropy_of_distribution(np.array([0.5, 0.5]), log_given=False)

        X = np.random.standard_normal(size=(10, 2))
        L.log_likelihood(TH, X)
        L.expected_C_and_Z_given_X(TH, X)
        L.expected_Z_given_X(TH, X)
        L.expected_C_given_X(TH, X)

        L.step(TH, X)
