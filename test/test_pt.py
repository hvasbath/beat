import pymc3 as pm
import numpy as num
import os
from beat.sampler import pt, metropolis
from beat.backend import TextStage
from tempfile import mkdtemp
import shutil
import logging
import theano.tensor as tt
import unittest
from pyrocko import util


logger = logging.getLogger('test_pt')


class TestPT(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.test_folder_multi = mkdtemp(prefix='PT_TEST')

        logger.info('Test result in: \n %s' % self.test_folder_multi)

        self.n_chains = 8
        self.n_samples = 2e3
        self.tune_interval = 25
        self.beta_tune_interval = 5e1
        self.swap_interval = (100, 300)
        self.buffer_size = self.n_samples / 20.
        self.burn = 0.5
        self.thin = 2

    def _test_sample(self, n_jobs, test_folder):
        logger.info('Running on %i cores...' % n_jobs)

        n = 4

        mu1 = num.ones(n) * (1. / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = num.power(stdev, 2) * num.eye(n)
        isigma = num.linalg.inv(sigma)
        dsigma = num.linalg.det(sigma)

        w1 = stdev
        w2 = (1 - stdev)

        def two_gaussians(x):
            log_like1 = - 0.5 * n * tt.log(2 * num.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            log_like2 = - 0.5 * n * tt.log(2 * num.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as PT_test:
            X = pm.Uniform('X',
                           shape=n,
                           lower=-2. * num.ones_like(mu1),
                           upper=2. * num.ones_like(mu1),
                           testval=-1. * num.ones_like(mu1),
                           transform=None)
            like = pm.Deterministic('tmp', two_gaussians(X))
            llk = pm.Potential('like', like)

        with PT_test:
            step = metropolis.Metropolis(
                n_chains=n_jobs,
                likelihood_name=PT_test.deterministics[0].name,
                proposal_name='Normal',
                tune_interval=self.tune_interval)

        pt.pt_sample(
            step,
            n_chains=n_jobs,
            n_samples=self.n_samples,
            swap_interval=self.swap_interval,
            beta_tune_interval=self.beta_tune_interval,
            homepath=test_folder,
            progressbar=False,
            buffer_size=self.buffer_size,
            model=PT_test,
            rm_flag=False,
            keep_tmp=False)

        stage_handler = TextStage(test_folder)

        mtrace = stage_handler.load_multitrace(-1, model=PT_test)

        n_steps = self.n_samples
        burn = self.burn
        thin = self.thin

        def burn_sample(x):
            if n_steps == 1:
                return x
            else:
                nchains = int(x.shape[0] / n_steps)
                xout = []
                for i in range(nchains):
                    nstart = int((n_steps * i) + (n_steps * burn))
                    nend = int(n_steps * (i + 1) - 1)
                    xout.append(x[nstart:nend:thin])

                return num.vstack(xout)

        from pymc3 import traceplot
        from matplotlib import pyplot as plt
        traceplot(mtrace, transform=burn_sample)
        plt.show()

        d = mtrace.get_values('X', combine=True, squeeze=True)
        mu1d = num.abs(d).mean(axis=0)

        num.testing.assert_allclose(mu1, mu1d, rtol=0., atol=0.03)

    def test_multicore(self):
        self._test_sample(self.n_chains, self.test_folder_multi)

    def tearDown(self):
        shutil.rmtree(self.test_folder_multi)


if __name__ == '__main__':
    util.setup_logging('test_pt', 'debug')
    unittest.main()
