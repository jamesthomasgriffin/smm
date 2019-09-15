import numpy.testing as npt
import pytest

import numpy as np
from ..rvs import UniformSimplexRV, NormalSimplexRV
from ..lemm import LinearlyEmbeddedMM, GLEMM_Parameters
from ..lemm.mcmc_integrator import MCMC_Integrator
from ..lemm.mcmc_integrator import EstimatedValuesAccumulator


class TestMCMC:
    def test_MCMC_Integrator(self):
        for RV in [UniformSimplexRV, NormalSimplexRV]:
            S = 0.1
            self.MCMC_Integrator_RV(RV, 'spherical', S)
            self.MCMC_Integrator_RV(RV, 'spherical', S, tied=False)

            S = np.array([0.1, 0.2])
            self.MCMC_Integrator_RV(RV, 'diagonal', S)
            self.MCMC_Integrator_RV(RV, 'diagonal', S, tied=False)

            S = np.array([[0.2, 0.1], [0.1, 0.2]])
            self.MCMC_Integrator_RV(RV, 'full', S)
            self.MCMC_Integrator_RV(RV, 'full', S, tied=False)

    # @pytest.mark.filterwarnings("ignore:importing the ABCs from")
    def MCMC_Integrator_RV(self, RV, covar_type, covar, tied=True):
        m = 3
        n = 2
        simplices = [(0, 1), (0, 2)]
        rvs = [RV(m, S) for S in simplices]
        rnd = np.random.RandomState(123)
        L = LinearlyEmbeddedMM(m, n, rvs, rnd=rnd)

        V = np.array([[0, 1], [1, 0], [1, 1]])
        TH = GLEMM_Parameters(V, L.M, None, covar_type, covar)
        if not tied:
            TH = TH.untie()

        def circle_data(rnd, n_points=50, r_std=0.0):
            r = rnd.randn(n_points) * r_std + 1.0
            θ = rnd.rand(n_points) * 2 * np.pi
            return np.vstack([r*np.cos(θ), r*np.sin(θ)]).T

        rnd = np.random.RandomState(123)
        N = 50
        X = circle_data(rnd, n_points=N)

        mc = MCMC_Integrator(L, TH, X)
        mc.change_parameters(TH)

        k = mc.k
        assert k == 1
        # assert mc.c.shape == (N,)
        assert mc.b.shape == (N, k)
        # assert mc.A.shape == (N, k, k)

        mc.U_step()
        mc.C_step()

        assert mc.exp_values.weight['U_step_acceptance'] == mc.N
        assert mc.exp_values.weight['C_step_acceptance'] == mc.N

        mc.contribute_q()
        mc.contribute_Z_given_X()
        mc.contribute_C_given_X()

        q, qZZ, qZX, qXX = mc.estimated_expected_q_values()
        Z_given_X = mc.estimated_exp_Z_given_X()
        C_given_X = mc.estimated_exp_C_given_X()

        if tied:
            assert q.shape == (mc.L.M,)
            assert qZZ.shape == (mc.L.m, mc.L.m)
            assert qZX.shape == (mc.L.m, mc.L.n)
            assert qXX.shape == (mc.L.n, mc.L.n)
        else:
            assert q.shape == (mc.L.M,)
            assert qZZ.shape == (mc.L.M, mc.L.m, mc.L.m)
            assert qZX.shape == (mc.L.M, mc.L.m, mc.L.n)
            assert qXX.shape == (mc.L.M, mc.L.n, mc.L.n)

        assert Z_given_X.shape == (mc.N, mc.L.m)
        assert C_given_X.shape == (mc.N, mc.L.M)

        mc.perform("UCqmz")

    def test_accumulator(self):
        eva = EstimatedValuesAccumulator()

        eva.record("score", 10, 1)
        eva.record("score", 35, 2)

        npt.assert_allclose(eva.expected_value("score"), 15)

        eva.reweight("score", weight=1)
        npt.assert_allclose(eva.expected_value("score"), 15)

        eva.record_interpolate("score", 25, 0.5)
        npt.assert_allclose(eva.expected_value("score"), 20)
        npt.assert_allclose(eva.weight["score"], 1)

        eva.record_interpolate("score", 40, 0.5, weight=2)
        npt.assert_allclose(eva.expected_value("score"), 20)
        npt.assert_allclose(eva.weight["score"], 1.5)

        eva.reweight("score", factor=1/1.5)
        npt.assert_allclose(eva.expected_value("score"), 20)
        npt.assert_allclose(eva.weight["score"], 1)

        with pytest.raises(KeyError):
            eva.record_interpolate("cores", 20, 0.5)

        with pytest.raises(KeyError):
            eva.expected_value("cores")

        with pytest.raises(KeyError):
            eva.reweight("cores", factor=0.5)

        with pytest.raises(ValueError):
            eva.reweight("score", factor=None, weight=None)

        # Reset used to set counter to zero, even if it hasn't been
        # encountered before
        eva.reset("score")
        eva.reset("scare")
