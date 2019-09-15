import numpy.testing as npt
import pytest

import numpy as np
from ..rvs import NormalSimplexRV, UniformSimplexRV, \
                  k_dim_degenerate_l_simplices
from ..lemm import (LinearlyEmbeddedMM,
                    GLEMM_Parameters)
from ..helpfulfunctions import initialise_V


class TestEMAlgorithm:

    @staticmethod
    def circle_data(n_points=100, r_std=0.0):
        r = np.random.randn(n_points) * r_std + 1.0
        θ = np.random.rand(n_points) * 2 * np.pi
        return np.vstack([r*np.cos(θ), r*np.sin(θ)]).T

    @pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
    def test_tied(self):

        n_data = 50
        n = 2

        np.random.seed(10)
        X = self.circle_data(n_data, 0.2)

        m = 5
        simplices = k_dim_degenerate_l_simplices(2, [1, 2], m)
        rvs = [NormalSimplexRV(m, S) for S in simplices]

        rnd = np.random.RandomState(123)

        L = LinearlyEmbeddedMM(m, n, rvs, rnd=rnd)

        TH1 = GLEMM_Parameters(
            initialise_V(X, m, n_iter=3, n_repl=10, rnd=rnd),
            L.M, None,
            'spherical', 0.1
        )

        TH2 = GLEMM_Parameters(
            X[:m].copy(),
            L.M, None,
            'diagonal', 0.1 + np.zeros(n)
        )

        TH3 = GLEMM_Parameters(
            X[:m].copy(),
            L.M, None,
            'full', 0.1 * np.eye(n)
        )

        npt.assert_allclose(TH1.cv_logdet, TH2.cv_logdet)
        npt.assert_allclose(TH1.cv_logdet, TH3.cv_logdet)
        npt.assert_allclose(TH1.cv_invchol*np.ones((L.n,)), TH2.cv_invchol)

        for TH in (TH1, TH2, TH3):
            L.step(TH, X)

    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
    def test_stochastic_EM(self):
        n_data = 50
        n = 2

        np.random.seed(10)
        X = self.circle_data(n_data, 0.2)

        m = 5
        simplices = k_dim_degenerate_l_simplices(2, [1, 2], m)
        rvs = [UniformSimplexRV(m, S) for S in simplices]

        rnd = np.random.RandomState(123)

        L = LinearlyEmbeddedMM(m, n, rvs, rnd=rnd)

        V = initialise_V(X, m, n_iter=3, n_repl=10, rnd=rnd)
        logp = np.zeros((L.M,))-np.log(L.M)

        TH1 = GLEMM_Parameters(V, L.M, logp, 'spherical', 0.1)

        TH2 = GLEMM_Parameters(V, L.M, logp, 'full',
                               0.1*np.eye(L.n))
        TH3 = GLEMM_Parameters(V, L.M, logp, 'diagonal',
                               0.01*np.ones(L.n))
        TH4 = TH1.untie()
        TH5 = TH2.untie()
        TH6 = TH3.untie()

        for TH in [TH1, TH2, TH3, TH4, TH5, TH6]:
            L.stochastic_step(TH, X, n_samples=100, batch_size=25)
            L.estimated_log_likelihood(TH, X, n_samples=50)
            L.estimated_expected_ZgivenX(TH, X, n_samples=20)
            L.estimated_expected_CgivenX(TH, X, n_samples=20)
            L.estimated_expected_CandZgivenX(TH, X, n_samples=20)
