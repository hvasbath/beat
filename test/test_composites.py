"""
Consistency tests of the seismic and geodetic composites on the example
projects in data/examples. The example data was synthesized with the
test point of the respective problem, hence synthetics have to match the data.

Needs the Green's function stores referenced in the example configs,
otherwise the tests are skipped.
"""

from copy import deepcopy

import numpy as num
import pytest
from conftest import load_example_problem
from numpy.testing import assert_allclose
from pytensor import function


@pytest.fixture(scope="module")
def seismic_problem(tmp_path_factory):
    return load_example_problem(
        "FullMT", "geometry", tmp_path_factory.mktemp("composites")
    )


@pytest.fixture(scope="module")
def geodetic_problem(tmp_path_factory):
    # TODO: Mogi example project is not part of data/examples
    return load_example_problem("Mogi", "geometry", tmp_path_factory.mktemp("mogi"))


class TestSeisComposite:
    def test_synths(self, seismic_problem):
        sc = seismic_problem.composites["seismic"]
        synths, obs = sc.get_synthetics(
            seismic_problem.model.test_point, outmode="data"
        )

        assert len(synths) == len(obs) > 0
        for st, ob in zip(synths, obs):
            assert_allclose(st.ydata, ob.ydata, rtol=1e-03, atol=0)

    def test_results(self, seismic_problem):
        sc = seismic_problem.composites["seismic"]
        results = sc.assemble_results(seismic_problem.model.test_point)

        assert len(results) > 0
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

    def test_weights(self, seismic_problem):
        sc = seismic_problem.composites["seismic"]
        for wmap in sc.wavemaps:
            for w, d in zip(wmap.weights, wmap.datasets):
                assert_allclose(
                    w.get_value(), d.covariance.chol_inverse, rtol=1e-08, atol=0
                )

    def test_lognorm_factor(self, seismic_problem):
        sc = seismic_problem.composites["seismic"]
        cov = deepcopy(sc.datasets[0].covariance)

        f = function([], [cov.slnf])

        cov.pred_v += num.ones_like(cov.data) * 1e-20
        cov.update_slnf()

        assert_allclose(cov.slnf.get_value(), f(), rtol=1e-06, atol=0)
        assert_allclose(cov.log_norm_factor, f(), rtol=1e-06, atol=0)


class TestGeoComposite:
    def test_synths(self, geodetic_problem):
        gc = geodetic_problem.composites["geodetic"]
        synths = gc.get_synthetics(
            geodetic_problem.model.test_point, outmode="stacked_arrays"
        )

        for st, ds in zip(synths, gc.datasets):
            assert_allclose(st, ds, rtol=1e-03, atol=0)

    def test_results(self, geodetic_problem):
        gc = geodetic_problem.composites["geodetic"]
        results = gc.assemble_results(geodetic_problem.model.test_point)

        for result in results:
            assert_allclose(
                result.processed_obs, result.processed_syn, rtol=1e-05, atol=0
            )

    def test_weights(self, geodetic_problem):
        gc = geodetic_problem.composites["geodetic"]
        for w, d in zip(gc.weights, gc.datasets):
            assert_allclose(
                w.get_value(), d.covariance.chol_inverse, rtol=1e-08, atol=0
            )
