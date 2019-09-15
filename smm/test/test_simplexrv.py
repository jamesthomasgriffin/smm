import pytest

import numpy.testing as npt

import numpy as np
from ..rvs import (UniformSimplexRV, k_dim_degenerate_l_simplices,
                   NormalSimplexRV)


class TestSimplexRVs:
    def test_UniformSimplexRV(self):
        with pytest.raises(ValueError):
            UniformSimplexRV(4, (0, 4))

        T = UniformSimplexRV(5, (1, 3, 4))
        assert T.k == 2
        assert T.m == 5
        npt.assert_allclose(T.mean, np.array([0, 1, 0, 1, 1])/3)

        sample = T.sample(10)

        # All samples have entries that sum to 1
        npt.assert_allclose(sample.sum(axis=1), np.ones((10,)))

        # All samples are contained in the correct support
        npt.assert_allclose(sample[:, (0, 2)], np.zeros((10, 2)))

        # All samples have positive entries
        assert sample.min() >= 0

        U = np.random.uniform(size=(1000, T.k))
        U.sort(axis=1)
        newU = T.MCstep(U, np.random.RandomState(123), delta=0.1)
        assert newU.max() <= 1
        assert newU.min() >= 0
        assert (np.diff(newU, axis=-1) >= 0).all()

        V = np.random.standard_normal(size=(5, 2))
        T.rate_fn_bound(1.0)
        T.rate_fn_bound(1.0, V=np.eye(T.m))
        with pytest.raises(ValueError):
            T.rate_fn_bound(1.0, V=np.eye(2))

        S = np.eye(2)
        X = np.random.standard_normal(size=(10, 2))
        for S in [np.eye(2), np.array([0.1, 0.1]), 0.1]:
            T.estimated_expected_moments_given_X(V, S, X, n_samples=10)

    def test_NormalSimplexRV(self):
        with pytest.raises(ValueError):
            NormalSimplexRV(4, (0, 4))

        T = NormalSimplexRV(5, (1, 3, 4))
        assert T.k == 2
        assert T.m == 5
        npt.assert_allclose(T.mean, np.array([0, 1, 0, 1, 1])/3)

        sample = T.sample(10)

        # All samples have entries that sum to 1
        npt.assert_allclose(sample.sum(axis=1), np.ones((10,)))

        # All samples are contained in the correct support
        npt.assert_allclose(sample[:, (0, 2)], np.zeros((10, 2)))

        T.rate_fn_bound(1.0)
        T.rate_fn_bound(1.0, V=np.eye(T.m))
        with pytest.raises(ValueError):
            T.rate_fn_bound(1.0, V=np.eye(2))

    def test_generating_lists_of_simplices(self):
        # The three vertices and edges of a triangle
        rvs = k_dim_degenerate_l_simplices(1, [0, 1], 3)
        assert len(rvs) == 6

        # The four faces of a tetrahedron
        rvs = k_dim_degenerate_l_simplices(2, [2], 4)
        assert len(rvs) == 4

        # The full, directed simple graph on 6 vertices
        rvs = k_dim_degenerate_l_simplices(2, [1], 6)
        assert len(rvs) == 30
