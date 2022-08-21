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
from theano import config

from beat import heart, models

config.mode = "FAST_COMPILE"


logger = logging.getLogger("test_geodetic")
km = 1000.0

project_dir = "/home/vasyurhm/BEATS/LaquilaJointPonlyUPDATE_nf"


class TestGeodeticComposite(unittest.TestCase):
    def setUp(self):

        self.mode = "geometry"
        self.problem = models.load_model(project_dir, self.mode)

    def test_step(self):

        step = self.problem.init_sampler()
        rp = self.problem.get_random_point()
        rp1 = deepcopy(rp)
        rp1["Laquila_ascxn_offset"] = num.array([0.05])

        _, lp0 = step.step(rp)
        _, lp1 = step.step(rp1)
        assert lp0[-1] != lp1[-1]


if __name__ == "__main__":
    util.setup_logging("test_geodetic", "info")
    unittest.main()
