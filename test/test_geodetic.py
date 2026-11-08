"""
Geodetic geometry problem with corrections (orbital ramp for SAR, euler pole
and strain rate for GNSS) on a synthetic project. The data were synthesized
without corrections, i.e. all corrections are zero at the true point.
"""

from copy import deepcopy

import numpy as num
import pytest
from numpy.testing import assert_allclose


@pytest.fixture(scope="module")
def project(synthetic_dir):
    from synthetic import make_project

    return make_project(
        synthetic_dir,
        "geodetic_corrections",
        ["geodetic"],
        "RectangularSource",
        corrections=True,
    )


@pytest.fixture(scope="module")
def problem(project):
    return project.load_problem()


@pytest.fixture
def true_point(project, problem):
    pc = problem.config.problem_config
    point = deepcopy(project.point)
    point.update({k: v.testvalue.copy() for k, v in pc.hierarchicals.items()})
    return point


def test_hierarchicals(problem):
    assert sorted(problem.hierarchicalnames) == sorted(
        [
            "S1_dsc_azimuth_ramp",
            "S1_dsc_range_ramp",
            "S1_dsc_offset",
            "1_pole_lat",
            "1_pole_lon",
            "1_omega",
            "2_exx",
            "2_eyy",
            "2_exy",
            "2_rotation",
        ]
    )
    for name in problem.hierarchicalnames:
        assert name in problem.model.named_vars


def test_ramp_offset_correction(problem, true_point):
    gc = problem.composites["geodetic"]
    synths = gc.get_synthetics(true_point)

    offset = 0.02
    point = deepcopy(true_point)
    point["S1_dsc_offset"] = num.array([offset])
    synths_offset = gc.get_synthetics(point)

    assert gc.datasets[0].name == "S1_dsc"
    assert_allclose(synths_offset[0], synths[0] + offset, rtol=0, atol=1e-10)
    for synth, synth_offset in zip(synths[1:], synths_offset[1:]):
        assert_allclose(synth_offset, synth, rtol=0, atol=1e-10)


def test_strain_rate_correction(problem, true_point):
    gc = problem.composites["geodetic"]
    synths = gc.get_synthetics(true_point)

    point = deepcopy(true_point)
    point["2_exx"] = num.array([100.0])
    synths_strain = gc.get_synthetics(point)

    assert_allclose(synths_strain[0], synths[0], rtol=0, atol=1e-10)
    changed = [
        not num.allclose(synth, synth_strain)
        for synth, synth_strain in zip(synths[1:], synths_strain[1:])
    ]
    assert any(changed)


def test_corrections_change_likelihood(problem, true_point):
    logp = problem.model.compile_logp()
    point = problem.model.initial_point()
    for name in problem.hierarchicalnames:
        assert_allclose(point[name], true_point[name])

    lp_true = logp(point)
    assert num.isfinite(lp_true)

    for name, value in [("S1_dsc_offset", 0.02), ("S1_dsc_range_ramp", 0.002)]:
        perturbed = deepcopy(point)
        perturbed[name] = num.array([value])
        assert logp(perturbed) < lp_true, name


def test_step(problem):
    step = problem.init_sampler()
    rp = problem.get_random_point()
    rp1 = deepcopy(rp)
    rp1["S1_dsc_offset"] = num.array([0.05])

    _, lp0 = step.step(rp)
    _, lp1 = step.step(rp1)
    assert num.isfinite(lp0[-1])
    assert lp0[-1] != lp1[-1]
