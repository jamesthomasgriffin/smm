import pytest

import numpy.testing as npt

import numpy as np
from ..rvs import NormalRV


class TestNormalRVs:
    def test_NormalRV(self):
        with pytest.raises(ValueError):
            # Testing mismatched dimensions, note M acts on the right
            mean = np.array([0, 1])
            M = np.array([[0, 1, 2], [1, 1, -1]])
            NormalRV(mean, M)

        mean = np.array([0, 0, 1])
        M = np.array([[0, 1, 2], [1, 1, -1]])
        T = NormalRV(mean, M)
        assert T.k == 2
        assert T.m == 3
        npt.assert_allclose(T.M, M)
        npt.assert_allclose(T.mean, mean)
        npt.assert_allclose(M.T.dot(M), T.covar)

        Y = T.U_sample(10, np.random.RandomState(123))
        assert Y.shape == (10, T.k)

        Y = T.sample(10)
        assert Y.shape == (10, T.m)

        X = mean[None, :]
        n = T.m
        log_PX = T.log_prob_X(np.eye(T.m), np.eye(n), X)
        assert log_PX.shape == (1,)
        Y = T.mean_given_X()
        npt.assert_allclose(Y, X)

        qZZ, qZX = T.moments_marg_over_X(log_PX, X)
        assert qZZ.shape == (T.m, T.m)
        assert qZX.shape == (T.m, n)

    def test_NormalRV_estimates(self):
        mean = np.array([0, 0, 1])
        M = np.array([[0, 1, 2], [1, 1, -1]])
        T = NormalRV(mean, M)
        rnd = np.random.RandomState(123)
        est_mean, est_cov = T.estimated_mean_and_covariance(10000, rnd=rnd)

        npt.assert_allclose(est_mean, T.mean, atol=0.1)
        npt.assert_allclose(est_cov, T.covar, atol=0.1)
