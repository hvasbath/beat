import logging
import multiprocessing as mp
import os
import shutil
import unittest
from tempfile import mkdtemp

import numpy as num
import pymc3 as pm
import theano.tensor as tt
from pyrocko import util

from beat import backend, smc, utility

logger = logging.getLogger("test_smc")


class TestSMC(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.test_folder_one = mkdtemp(prefix="SMC_TEST")
        self.test_folder_multi = mkdtemp(prefix="SMC_TEST")

        logger.info(
            "Test result in: \n %s, \n %s "
            % (self.test_folder_one, self.test_folder_multi)
        )

        self.n_cpu = mp.cpu_count()
        self.n_chains = 300
        self.n_steps = 100
        self.tune_interval = 25

    def _test_sample(self, n_jobs, test_folder):
        logger.info("Running on %i cores..." % n_jobs)

        n = 4

        mu1 = num.ones(n) * (1.0 / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = num.power(stdev, 2) * num.eye(n)
        isigma = num.linalg.inv(sigma)
        dsigma = num.linalg.det(sigma)

        w1 = stdev
        w2 = 1 - stdev

        def last_sample(x):
            return x[(self.n_steps - 1) :: self.n_steps]

        def two_gaussians(x):
            log_like1 = (
                -0.5 * n * tt.log(2 * num.pi)
                - 0.5 * tt.log(dsigma)
                - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            )
            log_like2 = (
                -0.5 * n * tt.log(2 * num.pi)
                - 0.5 * tt.log(dsigma)
                - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            )
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as SMC_test:
            X = pm.Uniform(
                "X",
                shape=n,
                lower=-2.0 * num.ones_like(mu1),
                upper=2.0 * num.ones_like(mu1),
                testval=-1.0 * num.ones_like(mu1),
                transform=None,
            )
            like = pm.Deterministic("like", two_gaussians(X))
            llk = pm.Potential("like", like)

        with SMC_test:
            step = smc.SMC(
                n_chains=self.n_chains,
                tune_interval=self.tune_interval,
                likelihood_name=SMC_test.deterministics[0].name,
            )

        smc.smc_sample(
            n_steps=self.n_steps,
            step=step,
            n_jobs=n_jobs,
            progressbar=True,
            stage=0,
            homepath=test_folder,
            model=SMC_test,
            rm_flag=False,
        )

        stage_handler = backend.SampleStage(test_folder)

        mtrace = stage_handler.load_multitrace(-1, model=SMC_test)

        d = mtrace.get_values("X", combine=True, squeeze=True)
        x = last_sample(d)
        mu1d = num.abs(x).mean(axis=0)

        num.testing.assert_allclose(mu1, mu1d, rtol=0.0, atol=0.03)

    def test_one_core(self):
        n_jobs = 1
        self._test_sample(n_jobs, self.test_folder_one)

    def test_multicore(self):
        n_jobs = utility.biggest_common_divisor(self.n_chains, self.n_cpu)
        self._test_sample(n_jobs, self.test_folder_multi)

    def tearDown(self):
        shutil.rmtree(self.test_folder_one)
        shutil.rmtree(self.test_folder_multi)


if __name__ == "__main__":
    util.setup_logging("test_smc", "info")
    unittest.main()
