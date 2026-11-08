"""
Polarity problem on the MTQT_polarity example of data/examples (real P-wave
polarities, the GF "store" only provides ray tracing / take-off angle tables,
calculated with cake, i.e. no external modelling codes are needed).
"""

import os
import shutil

import numpy as num
import pytest
from conftest import EXAMPLES_DIR
from numpy.testing import assert_allclose

from beat import config as bconfig
from beat import heart, plotting
from beat.apps import beat as cli
from beat.models import load_model

NAME = "MTQT_polarity"


def setup_project(main_path):
    project_dir = os.path.join(main_path, NAME)
    shutil.copytree(EXAMPLES_DIR / NAME, project_dir)
    config = bconfig.load_config(project_dir, "geometry")
    config.project_dir = project_dir
    config.polarity_config.gf_config.store_superdir = os.path.join(main_path, "GF")
    for wc in config.polarity_config.waveforms:
        wc.polarities_marker_path = os.path.join(
            project_dir, os.path.basename(wc.polarities_marker_path)
        )

    config.sampler_config.progressbar = False
    sp = config.sampler_config.parameters
    sp.n_chains = 20
    sp.n_steps = 20
    sp.n_jobs = 1
    sp.tune_interval = 10
    bconfig.dump_config(config)
    return project_dir


@pytest.fixture(scope="module")
def project_dir(tmp_path_factory):
    main_path = str(tmp_path_factory.mktemp("polarity"))
    project_dir = setup_project(main_path)
    cli.command_build_gfs(
        [
            NAME,
            "--main_path=%s" % main_path,
            "--datatypes=polarity",
            "--execute",
            "--loglevel=warning",
        ]
    )
    return project_dir


@pytest.fixture(scope="module")
def problem(project_dir):
    return load_model(project_dir, "geometry")


def test_data(problem):
    composite = problem.composites["polarity"]
    assert len(composite.wavemaps) == 1
    pmap = composite.wavemaps[0]

    blacklist = problem.config.polarity_config.waveforms[0].blacklist
    station_names = ["%s.%s" % (s.network, s.station) for s in pmap.stations]
    assert not set(blacklist) & set(station_names)
    assert pmap.n_t == len(station_names) == 25
    assert set(num.unique(pmap._prepared_data)) <= {-1, 1}


def test_synthetics_radiation_pattern(problem):
    composite = problem.composites["polarity"]
    point = problem.config.problem_config.get_test_point()
    synths, obs = composite.get_synthetics(point)
    assert len(synths) == len(obs) == 25

    source = composite.sources[0]
    m9 = source.pyrocko_moment_tensor().m()
    m9 = num.asarray(m9) / (num.sqrt(num.sum(num.asarray(m9) ** 2)) / num.sqrt(2.0))

    # P amplitude: gamma^T M gamma, gamma: ray direction at the source (NED)
    reference = []
    for target in composite.wavemaps[0].targets:
        takeoff = target.takeoff_angle_rad
        azimuth = num.deg2rad(source.azibazi_to(target)[0])
        gamma = num.array(
            [
                num.sin(takeoff) * num.cos(azimuth),
                num.sin(takeoff) * num.sin(azimuth),
                num.cos(takeoff),
            ]
        )
        reference.append(gamma.dot(m9).dot(gamma))

    assert_allclose(synths, reference, rtol=1e-6, atol=1e-10)


def test_takeoff_angle_tables(problem, project_dir, tmp_path):
    """Tabulated take-off angles have to agree with ray tracing."""
    config = bconfig.load_config(project_dir, "geometry")
    gfc = config.polarity_config.gf_config
    gfc.always_raytrace = False
    gfc.store_superdir = str(tmp_path)
    gfc.source_depth_min = 0.5
    gfc.source_depth_max = 7.5
    gfc.source_depth_spacing = 0.5
    gfc.source_distance_spacing = 2.0
    heart.polarity_construct_gf(
        [gfc.reference_location],
        config.event,
        config.polarity_config,
        execute=True,
        force=True,
    )

    from pyrocko.gf import LocalEngine

    engine = LocalEngine(store_superdirs=[str(tmp_path)])
    composite = problem.composites["polarity"]
    source = composite.sources[0]
    differences = []
    for target in composite.wavemaps[0].targets:
        store = engine.get_store(target.store_id)
        table = target.get_takeoff_angle_table(source, store)
        raytraced = target.get_takeoff_angle_cake(source, store)
        differences.append(abs(table - raytraced))

    # coarse tables, take-off angles of near horizontal rays vary rapidly
    assert num.median(differences) < 0.5
    assert max(differences) < 3.0


def test_likelihood(problem):
    logp = problem.model.compile_logp()
    point = problem.model.initial_point()
    assert num.isfinite(logp(point))


def test_sample_and_plot(project_dir):
    main_path = os.path.dirname(project_dir)
    args = ["--main_path=%s" % main_path, "--loglevel=warning"]
    num.random.seed(5)
    cli.command_sample([NAME] + args)
    cli.command_summarize([NAME, "--calc_derived"] + args)

    assert os.path.exists(os.path.join(project_dir, "geometry", bconfig.summary_name))

    problem = load_model(project_dir, "geometry", build=False)
    po = plotting.PlotOptions(outformat="png", dpi=50, force=True, nensemble=3)
    figure_dir = os.path.join(problem.outfolder, po.figure_dir)
    os.makedirs(figure_dir, exist_ok=True)
    for plotname in [
        "stage_posteriors",
        "fuzzy_mt_decomp",
        "hudson",
        "fuzzy_beachball",
    ]:
        plotting.plots_catalog[plotname](problem, po)

    assert len(os.listdir(figure_dir)) >= 4
