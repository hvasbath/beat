import pymc3 as pm
import numpy as num
import os
from beat import smc, utility, backend
from tempfile import mkdtemp
import shutil
import theano.tensor as tt
import multiprocessing as mp
import unittest
from pyrocko import util


class TestSMC(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.test_folder = mkdtemp(prefix='ATMIP_TEST')
        self.n_cpu = mp.cpu_count()

    def test_sample(self):
        n_chains = 300
        n_steps = 100
        tune_interval = 25
        n_jobs = utility.biggest_common_divisor(n_chains, self.n_cpu)

        n = 4

        mu1 = num.ones(n) * (1. / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = num.power(stdev, 2) * num.eye(n)
        isigma = num.linalg.inv(sigma)
        dsigma = num.linalg.det(sigma)

        w1 = stdev
        w2 = (1 - stdev)

        def last_sample(x):
            return x[(n_steps - 1)::n_steps]

        def two_gaussians(x):
            log_like1 = - 0.5 * n * tt.log(2 * num.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            log_like2 = - 0.5 * n * tt.log(2 * num.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as ATMIP_test:
            X = pm.Uniform('X',
                           shape=n,
                           lower=-2. * num.ones_like(mu1),
                           upper=2. * num.ones_like(mu1),
                           testval=-1. * num.ones_like(mu1),
                           transform=None)
            like = pm.Deterministic('like', two_gaussians(X))
            llk = pm.Potential('like', like)

        with ATMIP_test:
            step = smc.SMC(
                n_chains=n_chains,
                tune_interval=tune_interval,
                likelihood_name=ATMIP_test.deterministics[0].name)

        smc.ATMIP_sample(
            n_steps=n_steps,
            step=step,
            n_jobs=n_jobs,
            progressbar=False,
            stage='0',
            homepath=self.test_folder,
            model=ATMIP_test,
            rm_flag=False)

        stage_path = os.path.join(self.test_folder, 'stage_final')

        mtrace = backend.load(stage_path, model=ATMIP_test)

        d = mtrace.get_values('X', combine=True, squeeze=True)
        x = last_sample(d)
        mu1d = num.abs(x).mean(axis=0)

        num.testing.assert_allclose(mu1, mu1d, rtol=0., atol=0.03)

    def tearDown(self):
        shutil.rmtree(self.test_folder)

if __name__ == '__main__':
    util.setup_logging('test_smc', 'info')
    unittest.main()
