import numpy.testing as npt
import pytest

from scipy.special import logsumexp
import numpy as np

from ..lemm import LEMM_Parameters, GLEMM_Parameters, GLEMM_Parameters_Untied


class Test_Parameters:
    def test_lemm_parameters(self):
        M = 10
        m = 4
        n = 2
        logp = np.random.standard_normal((M,))
        logp -= logsumexp(logp, keepdims=True)
        V = np.random.randn(m, n)

        TH = LEMM_Parameters(V, M, logp)
        assert TH.M == M
        assert TH.m == m
        assert TH.n == n
        assert not TH.gaussian

        TH2 = LEMM_Parameters(V, M, None)
        assert TH2.logp.shape == (M,)
        npt.assert_allclose(np.sum(np.exp(TH2.logp)), 1.0)

        with pytest.raises(ValueError):
            LEMM_Parameters(V, M-1, logp)

    def test_glemm_parameters(self):
        M = 10
        m = 4
        n = 2
        V = np.random.randn(m, n)

        covars = [
            ('spherical', 1.0),
            ('diagonal', np.ones(n)),
            ('full', np.eye(n)),
        ]
        for cv_type, covar in covars:
            GLEMM_Parameters(V, M, None, cv_type, covar)

        TH = GLEMM_Parameters(V, M, None, 'spherical', 1.0)
        TH.relax_type('diagonal')
        assert TH.covar_type == 'diagonal'
        assert TH.cv_chol.shape == (n,)

        TH.relax_type('full')
        assert TH.covar_type == 'full'
        assert TH.cv_invchol.shape == (n, n)

        TH.restrict_type('diagonal')
        assert TH.covar_type == 'diagonal'

        TH.restrict_type('spherical')
        assert TH.covar_type == 'spherical'

        TH.relax_type('full')
        TH.restrict_type('spherical')

        TH.untie()

        covar = np.random.standard_exponential(size=(M,))
        TH = GLEMM_Parameters_Untied(V, M, None, 'spherical', covar)

        TH.relax_type('diagonal')
        assert TH.covar_type == 'diagonal'
        assert TH.cv_chol.shape == (M, n)

        TH.relax_type('full')
        assert TH.covar_type == 'full'
        assert TH.cv_invchol.shape == (M, n, n)

        TH.restrict_type('diagonal')
        assert TH.covar_type == 'diagonal'

        TH.restrict_type('spherical')
        assert TH.covar_type == 'spherical'

        TH.relax_type('full')
        TH.restrict_type('spherical')
