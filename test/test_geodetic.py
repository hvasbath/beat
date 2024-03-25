import logging
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as num
from pyrocko import util
from pytensor import config as tconfig
from pytest import mark

from beat import models

tconfig.mode = "FAST_COMPILE"


logger = logging.getLogger("test_geodetic")
km = 1000.0

# TODO update with version 2.0.0 compliant setup
project_dir = Path("/home/vasyurhm/BEATS/LaquilaJointPonlyUPDATE_nf")


class TestGeodeticComposite(unittest.TestCase):
    def setUp(self):
        self.mode = "geometry"
        mark.skipif(project_dir.is_dir() is False, reason="Needs project dir")
        self.problem = models.load_model(project_dir, self.mode)

    @mark.skipif(project_dir.is_dir() is False, reason="Needs project dir")
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
