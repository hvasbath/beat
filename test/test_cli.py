"""
End-to-end tests of the "beat" command line interface on synthetic projects:
import (raw data formats) -> sample -> summarize -> export / plot / check /
clone / update, and for the finite fault: init -> import results -> build_gfs
-> sample.

The projects have 5 % noise and are sampled with a small SMC setup to keep
the run times short, so posterior checks are coarse.

The plot functions are called directly instead of through "beat plot", which
only logs errors of the plot functions.
"""

import os
import time
from types import SimpleNamespace

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pyrocko.guts import load

from beat import config as bconfig
from pyrocko import util

from beat import heart, plotting, utility
from beat.apps import beat as cli
from beat.backend import load_multitrace
from beat.models import load_model

pytestmark = pytest.mark.slow

NOISE_LEVEL = 0.05


def beat(command, name, main_path, *args):
    """Run a beat subcommand like "beat <command> <name> [args]"."""
    command_func = getattr(cli, "command_%s" % command)
    return command_func(
        [name, "--main_path=%s" % main_path, "--loglevel=warning"] + list(args)
    )


def configure_small_sampler(project_dir, mode="geometry"):
    config = bconfig.load_config(project_dir, mode)
    config.sampler_config.progressbar = False
    sp = config.sampler_config.parameters
    sp.n_chains = 20
    sp.n_steps = 20
    sp.n_jobs = 1
    sp.tune_interval = 10

    hsp = config.hyper_sampler_config.parameters
    config.hyper_sampler_config.progressbar = False
    hsp.n_chains = 2
    hsp.n_steps = 100
    hsp.n_jobs = 1
    hsp.thin = 1
    hsp.tune_interval = 10
    bconfig.dump_config(config)


def load_final_stage(project_dir, mode, chains=None):
    """Final stage samples, chains=[-1] for the summarized trace only."""
    path = os.path.join(project_dir, mode, "stage_-1")
    problem = load_model(project_dir, mode, build=False)
    backend = problem.config.sampler_config.backend
    return load_multitrace(
        path, varnames=problem.varnames, backend=backend, chains=chains
    )


def assert_posterior_near_truth(project, varnames, fraction=0.25):
    """
    The small SMC setup underestimates the posterior spread, so check only
    that the posterior means are within a fraction of the prior width of the
    true values.
    """
    mtrace = load_final_stage(project.project_dir, "geometry")
    priors = project.config.problem_config.priors
    for varname in varnames:
        samples = mtrace.get_values(varname, combine=True, squeeze=True)
        prior = priors[varname]
        width = float(prior.upper[0] - prior.lower[0])
        assert (
            abs(samples.mean() - project.point[varname][0]) < fraction * width
        ), varname


def run_plot(problem, plotname, **kwargs):
    """Run plot function and return the newly written figure files."""
    po = plotting.PlotOptions(outformat="png", dpi=50, force=True, **kwargs)
    figure_dir = os.path.join(problem.outfolder, po.figure_dir)
    util.ensuredir(figure_dir)
    start = time.time() - 1.0
    plotting.plots_catalog[plotname](problem, po)
    return [
        fn
        for fn in os.listdir(figure_dir)
        if os.path.getmtime(os.path.join(figure_dir, fn)) >= start
    ]


@pytest.fixture(scope="module")
def main_path(tmp_path_factory):
    return tmp_path_factory.mktemp("cli")


@pytest.fixture(scope="module")
def geodetic_run(main_path, synthetic_dir):
    from synthetic import make_project, write_raw_data

    name = "geodetic"
    project = make_project(
        main_path,
        name,
        ["geodetic"],
        "RectangularSource",
        store_superdir=synthetic_dir / "GF",
        noise_level=NOISE_LEVEL,
        seed=1,
    )
    datapath = os.path.join(project.project_dir, bconfig.geodetic_data_name)
    original = utility.load_objects(datapath)

    write_raw_data(project, main_path / "geodetic_raw")
    beat(
        "import",
        name,
        main_path,
        "--datatypes=geodetic",
        "--geodetic_format=kite,ascii",
        "--force",
    )
    configure_small_sampler(project.project_dir)
    # module fixtures run before the function scoped seeding of conftest
    num.random.seed(10)
    beat("sample", name, main_path)
    beat("summarize", name, main_path, "--calc_derived")

    return SimpleNamespace(
        name=name,
        project=project,
        original=original,
        imported=utility.load_objects(datapath),
        problem=load_model(project.project_dir, "geometry", build=False),
    )


@pytest.fixture(scope="module")
def seismic_run(main_path, synthetic_dir):
    from synthetic import make_project, write_raw_data

    name = "seismic"
    project = make_project(
        main_path,
        name,
        ["seismic"],
        "MTSource",
        store_superdir=synthetic_dir / "GF",
        noise_level=NOISE_LEVEL,
        seed=2,
    )
    datapath = os.path.join(project.project_dir, bconfig.seismic_data_name)
    original = utility.load_objects(datapath)

    write_raw_data(project, main_path / "seismic_raw")
    beat(
        "import",
        name,
        main_path,
        "--datatypes=seismic",
        "--seismic_format=mseed",
        "--force",
    )
    configure_small_sampler(project.project_dir)
    # module fixtures run before the function scoped seeding of conftest
    num.random.seed(10)
    beat("sample", name, main_path)
    beat("summarize", name, main_path, "--calc_derived")

    return SimpleNamespace(
        name=name,
        project=project,
        original=original,
        imported=utility.load_objects(datapath),
        problem=load_model(project.project_dir, "geometry", build=False),
    )


@pytest.fixture(scope="module")
def ffi_run(main_path, geodetic_run):
    name = "geodetic_ffi"
    beat(
        "clone",
        geodetic_run.name,
        main_path,
        name,
        "--copy_data",
        "--datatypes=geodetic",
    )
    beat("init", name, main_path, "--mode=ffi", "--datatypes=geodetic")
    beat(
        "import",
        name,
        main_path,
        "--results=%s" % geodetic_run.project.project_dir,
        "--mode=ffi",
        "--import_from_mode=geometry",
        "--datatypes=geodetic",
    )

    project_dir = str(main_path / name)
    config = bconfig.load_config(project_dir, "ffi")
    dconfig = config.geodetic_config.gf_config.discretization_config
    dconfig.patch_widths = [1.0]
    dconfig.patch_lengths = [1.0]
    dconfig.extension_widths = [0.0]
    dconfig.extension_lengths = [0.0]
    config.problem_config.mode_config.regularization = "laplacian"
    config.update_hypers()
    bconfig.dump_config(config)
    configure_small_sampler(project_dir, mode="ffi")
    num.random.seed(11)

    beat(
        "build_gfs", name, main_path, "--mode=ffi", "--datatypes=geodetic", "--execute"
    )
    beat("sample", name, main_path, "--mode=ffi")
    beat("summarize", name, main_path, "--mode=ffi", "--calc_derived")

    return SimpleNamespace(
        name=name,
        project_dir=project_dir,
        problem=load_model(project_dir, "ffi", build=False),
    )


class TestGeodetic:
    def test_import_gnss(self, geodetic_run):
        original = [d for d in geodetic_run.original if d.typ == "GNSS"]
        imported = [d for d in geodetic_run.imported if d.typ == "GNSS"]
        assert [d.id for d in imported] == [d.id for d in original]
        for a, b in zip(original, imported):
            assert_allclose(b.displacement, a.displacement, rtol=0, atol=1e-10)
            assert_allclose(b.lats, a.lats, rtol=0, atol=1e-8)
            assert_allclose(b.lons, a.lons, rtol=0, atol=1e-8)
            assert_allclose(b.covariance.data, a.covariance.data, rtol=1e-8)

    def test_import_kite_scene(self, geodetic_run):
        sar = [d for d in geodetic_run.imported if d.typ == "SAR"]
        assert len(sar) == 1
        sar = sar[0]
        assert sar.name == "S1_dsc"
        assert sar.samples > 50
        assert sar.covariance.data.shape == (sar.samples, sar.samples)
        num.linalg.cholesky(sar.covariance.data)

        # BEAT synthetics (BEAT LOS from incidence and heading) have to fit
        # the data projected to LOS with the kite convention
        gc = geodetic_run.problem.composites["geodetic"]
        synths = gc.get_synthetics(geodetic_run.project.point)
        idx = [d.typ for d in gc.datasets].index("SAR")
        data = gc.datasets[idx].displacement
        residual = data - synths[idx]
        assert 1 - (residual**2).sum() / (data**2).sum() > 0.9

    def test_sample(self, geodetic_run):
        project_dir = geodetic_run.project.project_dir
        stage_dirs = os.listdir(os.path.join(project_dir, "geometry"))
        assert "stage_0" in stage_dirs
        assert "stage_-1" in stage_dirs

        assert_posterior_near_truth(
            geodetic_run.project,
            ["north_shift", "east_shift", "strike", "dip"],
        )

        # the maximum likelihood sample fits the SAR data
        mtrace = load_final_stage(project_dir, "geometry", chains=[-1])
        map_point = plotting.get_result_point(mtrace, "max")
        gc = geodetic_run.problem.composites["geodetic"]
        synths = gc.get_synthetics(map_point)
        idx = [d.typ for d in gc.datasets].index("SAR")
        data = gc.datasets[idx].displacement
        residual = data - synths[idx]
        assert 1 - (residual**2).sum() / (data**2).sum() > 0.9

    def test_summarize(self, geodetic_run):
        summary_file = os.path.join(
            geodetic_run.project.project_dir, "geometry", bconfig.summary_name
        )
        with open(summary_file) as f:
            summary = f.read()

        for varname in geodetic_run.problem.varnames:
            assert varname in summary

        # derived variables
        assert "magnitude" in summary

    @pytest.mark.parametrize("reference", [False, True])
    def test_export(self, geodetic_run, main_path, reference):
        args = ["--force"] + (["--reference"] if reference else [])
        beat("export", geodetic_run.name, main_path, *args)
        post_llk = "ref" if reference else "max"
        result = load(
            filename=os.path.join(
                geodetic_run.project.project_dir,
                "geometry",
                bconfig.results_dir_name,
                "solution_%s.yaml" % post_llk,
            )
        )
        assert isinstance(result, heart.ResultPoint)
        assert set(result.variance_reductions) == {d.id for d in geodetic_run.imported}
        assert set(geodetic_run.problem.varnames) <= set(result.point)
        assert len(result.events) == 1

    @pytest.mark.parametrize(
        "plotname",
        [
            "stage_posteriors",
            "correlation_hist",
            "scene_fits",
            "geodetic_covariances",
            "velocity_models",
            "fuzzy_beachball",
        ],
    )
    def test_plot(self, geodetic_run, plotname):
        assert len(run_plot(geodetic_run.problem, plotname, nensemble=3)) > 0

    @pytest.mark.gmt
    def test_plot_gnss_fits(self, geodetic_run):
        assert len(run_plot(geodetic_run.problem, "gnss_fits")) > 0

    def test_check_stores(self, geodetic_run, main_path):
        beat("check", geodetic_run.name, main_path, "--datatypes=geodetic")

    def test_clone(self, geodetic_run, main_path):
        beat(
            "clone",
            geodetic_run.name,
            main_path,
            "geodetic_clone",
            "--copy_data",
            "--datatypes=geodetic",
            "--sampler=Metropolis",
        )
        clone_dir = str(main_path / "geodetic_clone")
        config = bconfig.load_config(clone_dir, "geometry")
        assert config.name == "geodetic_clone"
        assert config.project_dir == clone_dir
        assert config.sampler_config.name == "Metropolis"
        assert os.path.exists(os.path.join(clone_dir, bconfig.geodetic_data_name))
        original = geodetic_run.problem.config.problem_config
        assert config.problem_config.priors.keys() == original.priors.keys()

    def test_update(self, geodetic_run, main_path):
        beat("update", geodetic_run.name, main_path, "--parameters=hypers")
        config = bconfig.load_config(geodetic_run.project.project_dir, "geometry")
        assert set(config.problem_config.hyperparameters) == {"h_SAR", "h_GNSS"}

    def test_sample_hypers(self, geodetic_run, main_path):
        project_dir = geodetic_run.project.project_dir
        old = bconfig.load_config(project_dir, "geometry").problem_config

        beat("sample", geodetic_run.name, main_path, "--hypers")
        hyper_dir = os.path.join(project_dir, "geometry", "hypers")
        assert os.path.exists(hyper_dir)
        assert len(os.listdir(hyper_dir)) > 0

        # hyperparameter bounds are updated with the sampled range
        new = bconfig.load_config(project_dir, "geometry").problem_config
        for name, param in new.hyperparameters.items():
            assert param.lower < param.testvalue < param.upper
            assert (param.lower, param.upper) != (
                old.hyperparameters[name].lower,
                old.hyperparameters[name].upper,
            )


class TestSeismic:
    def test_import_mseed(self, seismic_run):
        ostations, otraces = seismic_run.original
        istations, itraces = seismic_run.imported

        assert [s.station for s in istations] == [s.station for s in ostations]
        for os_, is_ in zip(ostations, istations):
            assert sorted(c.name for c in is_.channels) == sorted(
                c.name for c in os_.channels
            )

        assert len(itraces) == len(otraces)
        original = {tr.nslc_id: tr for tr in otraces}
        for tr in itraces:
            otr = original[tr.nslc_id]
            assert isinstance(tr, heart.SeismicDataset)
            assert abs(tr.tmin - otr.tmin) < 1e-3
            # the rotation to R, T uses the station coordinates of
            # stations.txt, which are rounded to 1e-5 deg
            atol = 0.0 if tr.channel == "Z" else 1e-4 * num.abs(otr.ydata).max()
            assert_allclose(tr.ydata, otr.ydata, rtol=0, atol=atol)

    def test_sample(self, seismic_run):
        # depth trades off with time and duration in the small SMC setup
        assert_posterior_near_truth(
            seismic_run.project, ["magnitude", "north_shift", "east_shift"]
        )

        # the maximum likelihood sample fits the waveforms almost as well as
        # the true model (5 % noise of the maximum amplitude of all traces)
        def median_variance_reduction(point):
            synths, obs = sc.get_synthetics(point, force=True)
            return num.median(
                [
                    1 - ((ob.ydata - st.ydata) ** 2).sum() / (ob.ydata**2).sum()
                    for st, ob in zip(synths, obs)
                ]
            )

        mtrace = load_final_stage(
            seismic_run.project.project_dir, "geometry", chains=[-1]
        )
        map_point = plotting.get_result_point(mtrace, "max")
        sc = seismic_run.problem.composites["seismic"]
        vr_true = median_variance_reduction(seismic_run.project.point)
        assert vr_true > 0.5
        assert median_variance_reduction(map_point) > vr_true - 0.15

    def test_summarize_derived(self, seismic_run):
        summary_file = os.path.join(
            seismic_run.project.project_dir, "geometry", bconfig.summary_name
        )
        with open(summary_file) as f:
            summary = f.read()

        for varname in seismic_run.problem.varnames:
            assert varname in summary

        # derived moment tensor parameters
        for varname in ["strike1", "dip1", "rake1"]:
            assert varname in summary

    def test_export(self, seismic_run, main_path):
        beat("export", seismic_run.name, main_path, "--force")
        results_dir = os.path.join(
            seismic_run.project.project_dir, "geometry", bconfig.results_dir_name
        )
        result = load(filename=os.path.join(results_dir, "solution_max.yaml"))
        assert len(result.variance_reductions) == 16
        assert all(
            isinstance(target, heart.DynamicTarget)
            for target in result.variance_reductions
        )
        files = os.listdir(results_dir)
        assert any(fn.startswith("processed_obs") for fn in files)
        assert any(fn.startswith("processed_syn") for fn in files)

    @pytest.mark.parametrize(
        "plotname",
        [
            "stage_posteriors",
            "correlation_hist",
            "waveform_fits",
            "fuzzy_mt_decomp",
            "hudson",
            "station_variance_reductions",
            "fuzzy_beachball",
            "velocity_models",
        ],
    )
    def test_plot(self, seismic_run, plotname):
        assert len(run_plot(seismic_run.problem, plotname, nensemble=3)) > 0

    @pytest.mark.gmt
    def test_plot_station_map(self, seismic_run):
        assert len(run_plot(seismic_run.problem, "station_map")) > 0

    def test_check_stores(self, seismic_run, main_path):
        beat("check", seismic_run.name, main_path, "--datatypes=seismic")


class TestFFI:
    def test_import_geometry_results(self, ffi_run, geodetic_run):
        mtrace = load_final_stage(
            geodetic_run.project.project_dir, "geometry", chains=[-1]
        )
        map_point = plotting.get_result_point(mtrace, "max")

        config = ffi_run.problem.config
        sources = config.geodetic_config.gf_config.reference_sources
        assert len(sources) == 1
        for varname in ["strike", "dip", "rake"]:
            assert_allclose(getattr(sources[0], varname), map_point[varname][0])
        assert_allclose(sources[0].depth, map_point["depth"][0] * 1000.0)

    def test_build_gfs(self, ffi_run):
        gc = ffi_run.problem.composites["geodetic"]
        fault = gc.load_fault_geometry()
        assert (
            fault.npatches == ffi_run.problem.config.problem_config.mode_config.npatches
        )
        assert fault.npatches > 1

        gfdir = os.path.join(ffi_run.project_dir, "ffi", bconfig.linear_gf_dir_name)
        for varname in gc.slip_varnames:
            assert os.path.exists(
                os.path.join(gfdir, "geodetic_%s_static_0.yaml" % varname)
            )

    def test_sample(self, ffi_run):
        mtrace = load_final_stage(ffi_run.project_dir, "ffi")
        npatches = ffi_run.problem.config.problem_config.mode_config.npatches
        uparr = mtrace.get_values("uparr", combine=True)
        assert uparr.shape[1] == npatches
        assert num.isfinite(uparr).all()

    def test_export(self, ffi_run, main_path):
        beat("export", ffi_run.name, main_path, "--mode=ffi", "--force")
        results_dir = os.path.join(ffi_run.project_dir, "ffi", bconfig.results_dir_name)
        assert os.path.exists(os.path.join(results_dir, "solution_max.yaml"))

    @pytest.mark.parametrize(
        "plotname",
        ["stage_posteriors", "slip_distribution", "scene_fits", "geodetic_covariances"],
    )
    def test_plot(self, ffi_run, plotname):
        assert len(run_plot(ffi_run.problem, plotname)) > 0

    @pytest.mark.gmt
    def test_plot_slip_distribution_3d(self, ffi_run):
        assert len(run_plot(ffi_run.problem, "slip_distribution_3d")) > 0
