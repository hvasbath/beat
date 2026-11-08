from copy import deepcopy

import numpy as num
import pytest
from conftest import require_external
from pytensor import config as tconfig

from beat import models

# TODO update with version 2.0.0 compliant setup
PROJECT = ("BEATS", "LaquilaJointPonlyUPDATE_nf")


@pytest.fixture(scope="module")
def problem():
    project_dir = require_external(*PROJECT)
    with tconfig.change_flags(mode="FAST_COMPILE"):
        yield models.load_model(str(project_dir), "geometry")


def test_step(problem):
    step = problem.init_sampler()
    rp = problem.get_random_point()
    rp1 = deepcopy(rp)
    rp1["Laquila_ascxn_offset"] = num.array([0.05])

    _, lp0 = step.step(rp)
    _, lp1 = step.step(rp1)
    assert lp0[-1] != lp1[-1]
