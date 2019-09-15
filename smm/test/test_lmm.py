import numpy.testing as npt

import numpy as np
from ..rvs import UniformSimplexRV, NormalSimplexRV
from ..lemm import (LinearlyEmbeddedMM,
                    GLEMM_Parameters,
                    GLEMM_Parameters_Untied)


class TestLEMM:
    def test_lemm(self):
        m = 3
        n = 2
        simplices = [(0, 1), (0, 2)]
        rvs = [NormalSimplexRV(m, S) for S in simplices]
        rnd = np.random.RandomState(123)
        L = LinearlyEmbeddedMM(m, n, rvs, rnd=rnd)
        assert L.m == 3
        assert L.n == 2
        assert L.M == 2

        V = np.array([[0, 1], [1, 0], [1, 1]])
        TH = GLEMM_Parameters(V, L.M, None, 'spherical', 0.1)

        L2 = L.copy()
        L2.mean_and_covar(TH)

        L2.top_components(TH)
        L2.entropy_of_distribution(np.array([0.5, 0.5]), log_given=False)
        L2.number_to_keep(TH.logp)

        X = np.random.standard_normal(size=(10, 2))
        L2.log_likelihood(TH, X)
        L2.expected_C_and_Z_given_X(TH, X)
        L2.expected_Z_given_X(TH, X)
        L2.expected_C_given_X(TH, X)
        L2.truncate(TH, number=1)

    def test_sampling(self):
        m = 3
        n = 2
        simplices = [(0, 1), (0, 2)]
        rvs = [UniformSimplexRV(m, S) for S in simplices]
        rnd = np.random.RandomState(123)
        L = LinearlyEmbeddedMM(m, n, rvs, rnd=rnd)

        par_tied = [True]
        par_type = ['spherical']
        par_S = [0.1]

        par_tied.append(False)
        par_type.append('diagonal')
        par_S.append(np.array([[0.1, 0.2], [0.3, 0.4]]))

        par_tied.append(True)
        par_type.append('full')
        par_S.append(np.array([[0.6, 0.1], [0.1, 0.5]]))

        for tied, covar_type, covar in zip(par_tied, par_type, par_S):
            print(tied, covar_type, covar)
            V = np.array([[0, 0], [1, 0], [0, 1]])
            if tied:
                TH = GLEMM_Parameters(V, L.M, None, covar_type, covar)
            else:
                TH = GLEMM_Parameters_Untied(V, L.M, None, covar_type, covar)

            L.encoding_rate(TH)

            for samp_fn, pm, dim in zip([L.sample, L.Z_sample, L.noisy_sample],
                                        [TH, TH.logp, TH],
                                        [L.n, L.m, L.n]):
                # Check sampling - ordered case
                n_samples = 20
                print(samp_fn.__name__)
                Y, indices, log_sw = samp_fn(pm, n_samples, in_order=True,
                                             output_weights=True)
                assert Y.shape == (n_samples, dim)
                assert log_sw.shape == (n_samples,)

                # Check that the indices all make sense
                assert len(indices) == L.M

                assert indices[0].start == 0
                last_stop = indices[0].stop
                for i in range(1, L.M):
                    assert indices[i].start == last_stop
                    last_stop = indices[i].stop
                assert last_stop == n_samples

                npt.assert_allclose(np.exp(log_sw).sum(), 2)

                # Check sampling - unordered case
                n_samples = 20
                Y, C_sample, log_sw = samp_fn(pm, n_samples, in_order=False,
                                              output_weights=True)
                assert Y.shape == (n_samples, dim)
                assert C_sample.shape == (n_samples,)
                assert log_sw.shape == (n_samples,)

                assert C_sample.dtype == np.int
                for c in C_sample:
                    assert c in range(L.M)

                npt.assert_allclose(np.exp(log_sw).sum(), 2)

            # Check sampling - ordered case
            n_samples = 20
            indices, log_sw = L.C_sample(TH.logp, n_samples, in_order=True,
                                         output_weights=True)
            assert log_sw.shape == (n_samples,)
            npt.assert_allclose(np.exp(log_sw).sum(), 2)

            # Check that the indices all make sense
            assert len(indices) == L.M

            assert indices[0].start == 0
            last_stop = indices[0].stop
            for i in range(1, L.M):
                assert indices[i].start == last_stop
                last_stop = indices[i].stop
            assert last_stop == n_samples

            # Check sampling - unordered case
            n_samples = 20
            C_sample, log_sw = L.C_sample(TH.logp, n_samples,
                                          in_order=False, output_weights=True)
            assert C_sample.shape == (n_samples,)
            assert log_sw.shape == (n_samples,)
            npt.assert_allclose(np.exp(log_sw).sum(), 2)

            assert C_sample.dtype == np.int
            for c in C_sample:
                assert c in range(L.M)
