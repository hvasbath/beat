import logging
import os
import shutil
import unittest
from copy import deepcopy
from tempfile import mkdtemp

import numpy as num
import theano.tensor as tt
from numpy.testing import assert_allclose
from pyrocko import orthodrome, plot, trace, util
from theano import function, shared

from beat import models

logger = logging.getLogger("test_heart")
km = 1000.0


class RundirectoryError(Exception):
    pass


def get_run_directory():
    cwd = os.getcwd()
    if os.path.basename(cwd) != "beat":
        raise RundirectoryError(
            "The test suite has to be run in the beat main-directory! "
            "Current work directory: %s" % cwd
        )
    else:
        return cwd


def load_problem(dirname, mode):
    beat_dir = get_run_directory()
    project_dir = os.path.join(beat_dir, "data/examples", dirname)
    return models.load_model(project_dir, mode=mode)


def _get_mt_source_params():
    source_point = {
        "magnitude": 4.8,
        "mnn": 0.84551376,
        "mee": -0.75868967,
        "mdd": -0.08682409,
        "mne": 0.51322155,
        "mnd": 0.14554675,
        "med": -0.25767963,
        "east_shift": 10.0,
        "north_shift": 20.0,
        "depth": 8.00,
        "time": -2.5,
        "duration": 5.0,
    }
    return {k: num.atleast_1d(num.asarray(v)) for k, v in source_point.items()}


class TestSeisComposite(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dirname = "FullMT"
        self.mode = "geometry"

    @classmethod
    def setUpClass(cls):
        dirname = "FullMT"
        mode = "geometry"
        cls.problem = load_problem(dirname, mode)
        cls.sc = cls.problem.composites["seismic"]

    def test_synths(self):
        logger.info("Test synth")
        synths, obs = self.sc.get_synthetics(
            self.problem.model.test_point, outmode="data"
        )

        for st, ob in zip(synths, obs):
            assert_allclose(st.ydata, ob.ydata, rtol=1e-03, atol=0)

    def test_results(self):
        logger.info("Test results")
        results = self.sc.assemble_results(self.problem.model.test_point)

        for result in results:
            assert_allclose(
                result.processed_obs.ydata,
                result.processed_syn.ydata,
                rtol=1e-03,
                atol=0,
            )
            assert_allclose(
                result.filtered_obs.ydata, result.filtered_syn.ydata, rtol=1e-03, atol=0
            )

    def test_weights(self):
        logger.info("Test weights")
        for wmap in self.sc.wavemaps:
            for w, d in zip(wmap.weights, wmap.datasets):
                assert_allclose(
                    w.get_value(), d.covariance.chol_inverse, rtol=1e-08, atol=0
                )

    def test_lognorm_factor(self):
        logger.info("Test covariance factor")
        cov = deepcopy(self.sc.datasets[0].covariance)

        f = function([], [cov.slnf])

        cov.pred_v += num.ones_like(cov.data) * 1e-20
        cov.update_slnf()

        assert_allclose(cov.slnf.get_value(), f(), rtol=1e-06, atol=0)
        assert_allclose(cov.log_norm_factor, f(), rtol=1e-06, atol=0)


class TestGeoComposite(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dirname = "Mogi"
        self.mode = "geometry"

    @classmethod
    def setUpClass(cls):
        dirname = "Mogi"
        mode = "geometry"
        cls.problem = load_problem(dirname, mode)
        cls.sc = cls.problem.composites["geodetic"]

    def test_synths(self):
        logger.info("Test synth")
        synths = self.sc.get_synthetics(
            self.problem.model.test_point, outmode="stacked_arrays"
        )

        for st, ds in zip(synths, self.sc.datasets):
            assert_allclose(st, ds, rtol=1e-03, atol=0)

    def test_results(self):
        logger.info("Test results")
        results = self.sc.assemble_results(self.problem.model.test_point)

        for result in results:
            assert_allclose(
                result.processed_obs, result.processed_syn, rtol=1e-05, atol=0
            )

    def test_weights(self):
        logger.info("Test weights")
        for w, d in zip(self.sc.weights, self.sc.datasets):
            assert_allclose(
                w.get_value(), d.covariance.chol_inverse, rtol=1e-08, atol=0
            )


if __name__ == "__main__":
    util.setup_logging("test_heart", "warning")
    unittest.main()
