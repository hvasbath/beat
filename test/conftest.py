"""
Shared pytest configuration and fixtures for the BEAT test suite.

Synthetic projects
------------------
The session fixtures ``seismic_project``, ``geodetic_project``,
``joint_project`` and ``ffi_project`` provide complete BEAT projects with
synthetic data and Green's function stores calculated with the pyrocko
ahfullgreen backend, i.e. without external data or modelling codes. They are
created once per session in a temporary directory, see synthetic.py.

Environment variables
---------------------
BEAT_TEST_DATA
    Root directory of additional (non-public) test data. It is expected to
    contain the sub-directories ``BEATS`` (project directories) and ``GF``
    (Green's function stores), i.e. the layout of the original developer
    machine. Tests that need such data are skipped if it is not available.
BEAT_TEST_SHOW
    If set to "1" matplotlib figures are shown interactively. By default
    figures are rendered with the non-interactive "Agg" backend and closed
    after every test.
"""

import os
import shutil
from pathlib import Path

import matplotlib
import pytest

SHOW = os.environ.get("BEAT_TEST_SHOW", "0") == "1"

if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as num  # noqa: E402
from pytensor import config as tconfig  # noqa: E402

from beat.info import project_root  # noqa: E402

EXAMPLES_DIR = Path(project_root) / "data" / "examples"
TEST_DIR = Path(__file__).parent


def _have_mpi():
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False

    return shutil.which("mpiexec") is not None


def _have_gmt():
    try:
        from pyrocko.plot import gmtpy

        gmtpy.GMT()
    except Exception:
        return False

    return shutil.which("gmtpy-epstopdf") is not None


def _external_data_dir():
    path = os.environ.get("BEAT_TEST_DATA", "")
    if path and Path(path).is_dir():
        return Path(path)

    return None


def pytest_collection_modifyitems(config, items):
    skip_mpi = pytest.mark.skip(reason="needs mpi4py and mpiexec")
    skip_gmt = pytest.mark.skip(reason="needs GMT and gmtpy-epstopdf")

    have_mpi = None
    have_gmt = None
    for item in items:
        if "mpi" in item.keywords:
            if have_mpi is None:
                have_mpi = _have_mpi()
            if not have_mpi:
                item.add_marker(skip_mpi)

        if "gmt" in item.keywords:
            if have_gmt is None:
                have_gmt = _have_gmt()
            if not have_gmt:
                item.add_marker(skip_gmt)


@pytest.fixture(autouse=True)
def _pytensor_test_values_raise():
    """
    beat.models.problems sets ``compute_test_value = "pdb"`` globally on
    import, which drops into the debugger (and blocks on stdin) on any graph
    construction error. BEAT needs test values to infer variable shapes, so
    they stay switched on, but tests have to fail instead of opening pdb.
    """
    with tconfig.change_flags(compute_test_value="raise"):
        yield


@pytest.fixture(autouse=True)
def _seed_random():
    num.random.seed(42)
    yield


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if SHOW:
        plt.show()

    plt.close("all")


@pytest.fixture(scope="session")
def test_dir():
    return TEST_DIR


@pytest.fixture(scope="session")
def external_data():
    """
    Root of the external test data given by $BEAT_TEST_DATA, skips otherwise.
    """
    path = _external_data_dir()
    if path is None:
        pytest.skip("needs external test data, set BEAT_TEST_DATA")

    return path


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """
    Session wide directory with the GF stores of the synthetic projects.
    """
    return tmp_path_factory.mktemp("synthetic")


@pytest.fixture(scope="session")
def seismic_project(synthetic_dir):
    """
    Synthetic project: seismic waveforms (P on Z, S on T) of a MTSource.

    The project is shared by all tests of the session, tests that write into
    the project directory need to work on a copy (SyntheticProject.copy).
    """
    from synthetic import make_project

    return make_project(synthetic_dir, "seismic_mt", ["seismic"], "MTSource")


@pytest.fixture(scope="session")
def geodetic_project(synthetic_dir):
    """Synthetic project: SAR and GNSS data of a RectangularSource."""
    from synthetic import make_project

    return make_project(
        synthetic_dir, "geodetic_rectangular", ["geodetic"], "RectangularSource"
    )


@pytest.fixture(scope="session")
def joint_project(synthetic_dir):
    """Synthetic project: geodetic and seismic data of a RectangularSource."""
    from synthetic import make_project

    return make_project(
        synthetic_dir, "joint_rectangular", ["geodetic", "seismic"], "RectangularSource"
    )


@pytest.fixture(scope="session")
def ffi_project(joint_project, tmp_path_factory):
    """
    Synthetic finite fault project (mode "ffi") derived from joint_project
    with linear GF libraries for geodetic and seismic data.
    """
    from synthetic import make_ffi_project

    return make_ffi_project(joint_project, tmp_path_factory.mktemp("ffi"))


def require_external(*parts):
    """
    Return path below $BEAT_TEST_DATA or skip the test if it does not exist.
    To be used inside of tests or fixtures.
    """
    root = _external_data_dir()
    if root is None:
        pytest.skip("needs external test data, set BEAT_TEST_DATA")

    path = root.joinpath(*parts)
    if not path.exists():
        pytest.skip("external test data not found: %s" % path)

    return path
