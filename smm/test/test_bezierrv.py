import pytest

import numpy.testing as npt

import numpy as np
from ..rvs import CubicBezierRV, QuadraticBezierRV, complete_Bezier_graph


class TestBezierRVs:
    def test_BezierRV(self):
        for RV, S in [(CubicBezierRV, (0, 1, 2, 3)),
                      (QuadraticBezierRV, (0, 1, 2))]:
            self.do_RV(RV, S)

    def do_RV(self, RV, S):
        m = 5
        with pytest.raises(ValueError):
            RV(m, (1, 2))
        with pytest.raises(ValueError):
            RV(m, [10 + s for s in S])

        T = RV(m, S)
        assert T.k == len(S)
        assert T.m == 5

        Y = T.sample(10)
        assert Y.shape == (10, T.m)
        npt.assert_allclose(Y.sum(axis=1), np.ones((10,)))

        rnd = np.random.RandomState(123)
        U = T.U_sample(10, rnd)
        T.MCstep(U, rnd)

    def test_complete_graph(self):
        k, n = 4, 5
        V = np.random.standard_normal(size=(k, n))

        rvs, m, W = complete_Bezier_graph(k, type="quadratic")
        rvs, m, W = complete_Bezier_graph(k, type="quadratic", V=V)
        assert len(rvs) + k == m
        assert W.shape == (m, n)
        rvs, m, W = complete_Bezier_graph(k, type="cubic", V=V)
        assert 2 * len(rvs) + k == m
        assert W.shape == (m, n)
