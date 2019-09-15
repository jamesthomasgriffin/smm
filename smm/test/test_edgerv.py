import pytest

import numpy.testing as npt

import numpy as np
from ..rvs.edgerv import EdgeRV


class TestEdgeRVs:
    def test_EdgeRV(self):
        with pytest.raises(ValueError):
            EdgeRV(4, (0, 4))

        with pytest.raises(ValueError):
            EdgeRV(4, (0, 1, 2))

        T = EdgeRV(5, (1, 3))
        assert T.k == 1
        assert T.m == 5
        npt.assert_allclose(T.mean, np.array([0, 1, 0, 1, 0])/2)
        npt.assert_allclose(T.Moffset, np.array([0, 0, 0, 1, 0]))
        npt.assert_allclose(T.M, np.array([[0, 1, 0, -1, 0]]))

        sample = T.sample(10)

        # All samples have entries that sum to 1
        npt.assert_allclose(sample.sum(axis=1), np.ones((10,)))

        # All samples are contained in the correct support
        npt.assert_allclose(sample[:, (0, 2, 4)], np.zeros((10, 3)))

        # All samples have positive entries
        assert sample.min() >= 0

        U = np.random.uniform(size=(100, T.k))
        U.sort(axis=1)
        newU = T.MCstep(U, np.random.RandomState(123), delta=0.1)
        assert newU.max() <= 1
        assert newU.min() >= 0
        assert (np.diff(newU, axis=-1) >= 0).all()

        rnd = np.random.RandomState(123)
        X = rnd.standard_normal(size=(10, 5))
        X[0] = [0, 2, 0, 2, 0]
        n = T.m
        log_PX = T.log_prob_X(np.eye(T.m), np.eye(n), X)
        assert log_PX.shape == (10,)
        npt.assert_array_less(np.zeros_like(T.saved_E_UgX), T.saved_E_UgX)
        npt.assert_array_less(T.saved_E_UgX, np.ones_like(T.saved_E_UgX))

        E_UgX = T.mean_given_X()
        npt.assert_allclose(E_UgX[0], T.mean)
