import logging
import os
import shutil
import unittest
from tempfile import mkdtemp

import numpy as num
import pymc3 as pm
import theano.tensor as tt
from pyrocko import util
from pyrocko.plot import mpl_papersize

from beat.backend import SampleStage
from beat.config import sample_p_outname
from beat.sampler import metropolis, pt
from beat.sampler.pt import SamplingHistory
from beat.utility import load_objects, mod_i

logger = logging.getLogger("test_pt")


class TestPT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.test_folder_multi = mkdtemp(prefix="PT_TEST")

        logger.info("Test result in: \n %s" % self.test_folder_multi)

        self.n_chains = 8
        self.n_workers_posterior = 2
        self.n_samples = int(3e4)
        self.tune_interval = 50
        self.beta_tune_interval = 3000
        self.swap_interval = (10, 15)
        self.buffer_size = self.n_samples / 10.0
        self.burn = 0.5
        self.thin = 1

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

        with pm.Model() as PT_test:
            X = pm.Uniform(
                "X",
                shape=n,
                lower=-2.0 * num.ones_like(mu1),
                upper=2.0 * num.ones_like(mu1),
                testval=-1.0 * num.ones_like(mu1),
                transform=None,
            )
            like = pm.Deterministic("tmp", two_gaussians(X))
            llk = pm.Potential("like", like)

        with PT_test:
            step = metropolis.Metropolis(
                n_chains=n_jobs,
                likelihood_name=PT_test.deterministics[0].name,
                proposal_name="MultivariateCauchy",
                tune_interval=self.tune_interval,
            )

        pt.pt_sample(
            step,
            n_chains=n_jobs,
            n_samples=self.n_samples,
            swap_interval=self.swap_interval,
            beta_tune_interval=self.beta_tune_interval,
            n_workers_posterior=self.n_workers_posterior,
            homepath=test_folder,
            progressbar=False,
            buffer_size=self.buffer_size,
            model=PT_test,
            rm_flag=False,
            keep_tmp=False,
        )

        stage_handler = SampleStage(test_folder)

        mtrace = stage_handler.load_multitrace(-1, varnames=PT_test.vars)
        history = load_objects(
            os.path.join(stage_handler.stage_path(-1), sample_p_outname)
        )

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

        from matplotlib import pyplot as plt
        from pymc3 import traceplot

        with PT_test:
            traceplot(mtrace, transform=burn_sample)

        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=mpl_papersize("a5", "portrait")
        )
        axes[0].plot(history.acceptance, "r")
        axes[0].set_ylabel("Acceptance ratio")
        axes[0].set_xlabel("Update interval")
        axes[1].plot(num.array(history.t_scales), "k")
        axes[1].set_ylabel("Temperature scaling")
        axes[1].set_xlabel("Update interval")

        n_acceptances = len(history)
        ncol = 3
        nrow = int(num.ceil(n_acceptances / float(ncol)))

        fig2, axes1 = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=mpl_papersize("a4", "portrait")
        )
        axes1 = num.atleast_2d(axes1)
        fig3, axes2 = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=mpl_papersize("a4", "portrait")
        )
        axes2 = num.atleast_2d(axes2)

        acc_arrays = history.get_acceptance_matrixes_array()
        sc_arrays = history.get_sample_counts_array()
        scvmin = sc_arrays.min(0).min(0)
        scvmax = sc_arrays.max(0).max(0)
        accvmin = acc_arrays.min(0).min(0)
        accvmax = acc_arrays.max(0).max(0)

        for i in range(ncol * nrow):
            rowi, coli = mod_i(i, ncol)
            # if i == n_acceptances:
            #   pass
            # plt.colorbar(im, axes1[rowi, coli])
            # plt.colorbar(im2, axes2[rowi, coli])

            if i > n_acceptances - 1:
                try:
                    fig2.delaxes(axes1[rowi, coli])
                    fig3.delaxes(axes2[rowi, coli])
                except KeyError:
                    pass
            else:
                axes1[rowi, coli].matshow(
                    history.acceptance_matrixes[i],
                    vmin=accvmin[i],
                    vmax=accvmax[i],
                    cmap="hot",
                )
                axes1[rowi, coli].set_title("min %i, max%i" % (accvmin[i], accvmax[i]))
                axes1[rowi, coli].get_xaxis().set_ticklabels([])
                axes2[rowi, coli].matshow(
                    history.sample_counts[i], vmin=scvmin[i], vmax=scvmax[i], cmap="hot"
                )
                axes2[rowi, coli].set_title("min %i, max%i" % (scvmin[i], scvmax[i]))
                axes2[rowi, coli].get_xaxis().set_ticklabels([])

        fig2.suptitle("Accepted number of samples")
        fig2.tight_layout()
        fig3.tight_layout()
        fig3.suptitle("Total number of samples")
        plt.show()

        # d = mtrace.get_values('X', combine=True, squeeze=True)
        # mu1d = num.abs(d).mean(axis=0)

        # num.testing.assert_allclose(mu1, mu1d, rtol=0., atol=0.03)

    def test_multicore(self):
        self._test_sample(self.n_chains, self.test_folder_multi)

    def tearDown(self):
        shutil.rmtree(self.test_folder_multi)


if __name__ == "__main__":
    util.setup_logging("test_pt", "info")
    unittest.main()
