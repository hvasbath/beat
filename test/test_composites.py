"""
Consistency tests of the seismic and geodetic composites on the synthetic
projects (see synthetic.py). The data were synthesized with the test point of
the respective problem plus white noise of known standard deviation, hence
the synthetics of the composites have to fit the data up to that noise.
"""

from copy import deepcopy

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pytensor import function


def rms(x):
    return num.sqrt(num.mean(num.asarray(x) ** 2))


@pytest.fixture(scope="module", params=["seismic_project", "joint_project"])
def seismic_setup(request):
    project = request.getfixturevalue(request.param)
    problem = project.load_problem()
    return project, problem


@pytest.fixture(scope="module", params=["geodetic_project", "joint_project"])
def geodetic_setup(request):
    project = request.getfixturevalue(request.param)
    problem = project.load_problem()
    return project, problem


@pytest.fixture(scope="module", params=["seismic_project", "geodetic_project"])
def setup(request):
    project = request.getfixturevalue(request.param)
    problem = project.load_problem()
    return project, problem


class TestSeisComposite:
    def test_wavemaps(self, seismic_setup):
        _, problem = seismic_setup
        sc = problem.composites["seismic"]
        assert [wmap.name for wmap in sc.wavemaps] == ["any_P", "any_S"]
        assert [wmap.channels for wmap in sc.wavemaps] == [["Z"], ["T"]]
        for wmap in sc.wavemaps:
            assert wmap.n_t == 8

    def test_synths(self, seismic_setup):
        project, problem = seismic_setup
        sc = problem.composites["seismic"]
        synths, obs = sc.get_synthetics(project.point, force=True)

        assert len(synths) == len(obs) == 16
        residuals = []
        for st, ob in zip(synths, obs):
            assert st.ydata.shape == ob.ydata.shape
            assert_allclose(st.tmin, ob.tmin)
            residuals.append(ob.ydata - st.ydata)

        residuals = num.concatenate(residuals)
        signal = num.concatenate([ob.ydata for ob in obs])
        sigma = project.noise_sigmas["seismic"]
        # filtered (band-limited) and tapered white noise
        assert rms(residuals) < sigma
        assert rms(signal) > 10 * rms(residuals)

    def test_results(self, seismic_setup):
        project, problem = seismic_setup
        sc = problem.composites["seismic"]
        results = sc.assemble_results(project.point)

        assert len(results) == 16
        for result in results:
            assert_allclose(
                result.processed_res.ydata,
                result.processed_obs.ydata - result.processed_syn.ydata,
                rtol=1e-6,
                atol=1e-12,
            )
            assert result.processed_syn.ydata.size == result.processed_obs.ydata.size

    def test_noise_estimate(self, seismic_setup):
        project, problem = seismic_setup
        sc = problem.composites["seismic"]
        sigma = project.noise_sigmas["seismic"]
        for dataset in sc.datasets:
            variance = num.diag(dataset.covariance.data)
            # variance of band-pass filtered white noise of std sigma, but the
            # traces are demeaned and filtered without taper, so the filter
            # transient of the static offset leaks into the pre-arrival window
            assert num.all(variance > 0.1 * sigma**2)
            assert num.all(variance < 2.0 * sigma**2)

    def test_weights(self, seismic_setup):
        _, problem = seismic_setup
        sc = problem.composites["seismic"]
        for wmap in sc.wavemaps:
            assert len(wmap.weights) == len(wmap.datasets)
            for w, d in zip(wmap.weights, wmap.datasets):
                assert_allclose(
                    w.get_value(), d.covariance.chol_inverse, rtol=1e-08, atol=0
                )

    def test_lognorm_factor(self, seismic_setup):
        _, problem = seismic_setup
        sc = problem.composites["seismic"]
        cov = deepcopy(sc.datasets[0].covariance)

        f = function([], [cov.slog_pdet])

        cov.pred_v += num.ones_like(cov.data) * 1e-20
        cov.update_slog_pdet()

        assert_allclose(cov.slog_pdet.get_value(), f(), rtol=1e-06, atol=0)
        assert_allclose(cov.log_pdet, f(), rtol=1e-06, atol=0)


class TestGeoComposite:
    def test_datasets(self, geodetic_setup):
        _, problem = geodetic_setup
        gc = problem.composites["geodetic"]
        assert [d.typ for d in gc.datasets] == ["SAR", "GNSS", "GNSS", "GNSS"]
        assert [d.id for d in gc.datasets] == [
            "S1_dsc",
            "synthetic_gnss_north",
            "synthetic_gnss_east",
            "synthetic_gnss_up",
        ]

    def test_synths(self, geodetic_setup):
        project, problem = geodetic_setup
        gc = problem.composites["geodetic"]
        synths = gc.get_synthetics(project.point)

        assert len(synths) == len(gc.datasets)
        sigma = project.noise_sigmas["geodetic"]
        residuals = num.concatenate(
            [ds.displacement - st for st, ds in zip(synths, gc.datasets)]
        )
        # white noise of std sigma, 471 samples
        assert_allclose(residuals.std(), sigma, rtol=0.15)
        assert_allclose(residuals.mean(), 0.0, atol=0.2 * sigma)

    def test_results(self, geodetic_setup):
        project, problem = geodetic_setup
        gc = problem.composites["geodetic"]
        results = gc.assemble_results(project.point)

        assert len(results) == len(gc.datasets)
        for result, dataset in zip(results, gc.datasets):
            assert_allclose(result.processed_obs, dataset.displacement)
            assert_allclose(
                result.processed_res,
                result.processed_obs - result.processed_syn,
                rtol=1e-6,
                atol=1e-12,
            )

    def test_weights(self, geodetic_setup):
        project, problem = geodetic_setup
        gc = problem.composites["geodetic"]
        sigma = project.noise_sigmas["geodetic"]
        for w, d in zip(gc.weights, gc.datasets):
            assert_allclose(
                w.get_value(), d.covariance.chol_inverse, rtol=1e-08, atol=0
            )
            assert_allclose(
                w.get_value(), num.eye(d.samples) / sigma, rtol=1e-5, atol=1e-8
            )


class TestProblem:
    @pytest.mark.parametrize(
        "project_name", ["seismic_project", "geodetic_project", "joint_project"]
    )
    def test_variance_reductions(self, request, project_name):
        project = request.getfixturevalue(project_name)
        # updates the weights inplace for other time windows, which would
        # break the model of the shared problem for other tests
        problem = project.load_problem(build=False)

        vrs = num.array(list(problem.get_variance_reductions(project.point).values()))
        assert vrs.size > 0

        # VR = 1 - |r|^2 / |d|^2 with residuals r of the white noise of std
        # sigma (at most, the seismic data are filtered), i.e. E(|r|^2) <= n sigma^2
        n_sigma2 = []
        for datatype, composite in problem.composites.items():
            sigma = project.noise_sigmas[datatype]
            if datatype == "geodetic":
                data = [d.displacement for d in composite.datasets]
            else:
                results = composite.assemble_results(project.point)
                data = [r.processed_obs.ydata for r in results]
            n_sigma2.extend(d.size * sigma**2 / (d**2).sum() for d in data)

        assert (vrs > 1 - 3 * num.array(n_sigma2)).all()

        perturbed = deepcopy(project.point)
        perturbed["north_shift"] = perturbed["north_shift"] + 1.0
        vrs_perturbed = num.array(
            list(problem.get_variance_reductions(perturbed).values())
        )
        assert vrs.mean() > vrs_perturbed.mean() + 0.05

    @pytest.mark.parametrize("varname", ["north_shift", "depth", "time"])
    def test_true_point_is_more_likely(self, setup, varname):
        project, problem = setup
        point = problem.model.initial_point()
        if varname not in point:
            pytest.skip(f"{varname} not a parameter of {project.source_type}")

        logp = problem.model.compile_logp()
        perturbed = deepcopy(point)
        perturbed[varname] = perturbed[varname] + 0.5

        assert num.isfinite(logp(point))
        assert logp(point) > logp(perturbed)
