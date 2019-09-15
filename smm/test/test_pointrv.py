import pytest

import numpy.testing as npt

import numpy as np
from ..rvs import PointRV


class TestPointRVs:
    def test_PointRV(self):
        with pytest.raises(ValueError):
            # Testing mean of wrong dimension
            mean = np.array([0, 1])
            PointRV(3, mean)

        with pytest.raises(ValueError):
            # Test vertex out of range
            PointRV(3, 3)

        mean = np.array([0, 0, 1])
        T = PointRV(3, mean)
        assert T == PointRV(3, 2)
        assert T.k == 0
        assert T.m == 3
        npt.assert_allclose(T.mean, mean)
        npt.assert_allclose(T.covar, np.zeros((3, 3)))

        Y = T.sample(10)
        assert Y.shape == (10, T.m)

        X = mean[None, :]
        n = T.m

        log_PX = T.log_prob_X(np.eye(T.m), 1.0, X)
        log_PX = T.log_prob_X(np.eye(T.m), np.ones((T.m,)), X)
        log_PX = T.log_prob_X(np.eye(T.m), np.eye(T.m), X)
        assert log_PX.shape == (1,)
        Y = T.mean_given_X()
        npt.assert_allclose(Y, X)

        qZZ, qZX = T.moments_marg_over_X(log_PX, X)
        assert qZZ.shape == (T.m, T.m)
        assert qZX.shape == (T.m, n)

        T.rate_fn_bound(1.0)
        T.rate_fn_bound(1.0, V=np.eye(T.m))
        with pytest.raises(ValueError):
            T.rate_fn_bound(1.0, V=np.eye(5))
