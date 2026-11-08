import numpy as num
import pytensor.tensor as tt
import pytest
from numpy.testing import assert_allclose
from pytensor import config as tconfig
from pytensor import function

from beat import pytensorf
from beat.fast_sweeping import fast_sweep

km = 1000.0
PATCH_SIZE = 10.0 * km
NUC_X = 2
NUC_Y = 3
N_PATCH_STRIKE = 4
N_PATCH_DIP = 6


@pytest.fixture
def slownesses():
    velo1 = num.ones((N_PATCH_DIP, N_PATCH_STRIKE // 2))
    velo2 = num.ones((N_PATCH_DIP, N_PATCH_STRIKE // 2)) * 3.5
    return 1.0 / num.concatenate((velo1, velo2), axis=1)


def numpy_implementation(slownesses):
    return fast_sweep.get_rupture_times_numpy(
        slownesses, PATCH_SIZE / km, N_PATCH_STRIKE, N_PATCH_DIP, NUC_X, NUC_Y
    ).flatten()


def pytensor_implementation(slownesses):
    t_slownesses = tt.dmatrix("slownesses")
    t_slownesses.tag.test_value = slownesses

    nuc_x = tt.lscalar("nuc_x")
    nuc_x.tag.test_value = NUC_X

    nuc_y = tt.lscalar("nuc_y")
    nuc_y.tag.test_value = NUC_Y

    patch_size = tt.cast(PATCH_SIZE / km, "float64")

    # scan inner graph variables have no test values
    with tconfig.change_flags(compute_test_value="off"):
        start_times = fast_sweep.get_rupture_times_pytensor(
            t_slownesses, patch_size, nuc_x, nuc_y
        )
        f = function([t_slownesses, nuc_x, nuc_y], start_times)
    return f(slownesses, NUC_X, NUC_Y).flatten()


def c_implementation(slownesses):
    return fast_sweep.get_rupture_times_c(
        slownesses.flatten(),
        PATCH_SIZE / km,
        N_PATCH_STRIKE,
        N_PATCH_DIP,
        NUC_X,
        NUC_Y,
    )


def pytensor_c_wrapper(slownesses):
    t_slownesses = tt.dvector("slownesses")
    t_slownesses.tag.test_value = slownesses.flatten()

    nuc_x = tt.lscalar("nuc_x")
    nuc_x.tag.test_value = NUC_X

    nuc_y = tt.lscalar("nuc_y")
    nuc_y.tag.test_value = NUC_Y

    sweeper = pytensorf.Sweeper(PATCH_SIZE / km, N_PATCH_DIP, N_PATCH_STRIKE, "c")
    start_times = sweeper(t_slownesses, nuc_y, nuc_x)

    f = function([t_slownesses, nuc_y, nuc_x], start_times)
    return f(slownesses.flatten(), NUC_Y, NUC_X)


def test_numpy_rupture_times_properties(slownesses):
    times = numpy_implementation(slownesses).reshape((N_PATCH_DIP, N_PATCH_STRIKE))

    assert times[NUC_Y, NUC_X] == pytest.approx(0.0)
    assert (times >= 0.0).all()
    # direct neighbour along strike in the same velocity domain
    assert times[NUC_Y, NUC_X + 1] == pytest.approx(
        PATCH_SIZE / km * slownesses[NUC_Y, NUC_X + 1], rel=0.5
    )
    # rupture times increase away from the nucleation point along dip
    assert (num.diff(times[NUC_Y:, NUC_X]) > 0).all()
    assert (num.diff(times[: NUC_Y + 1, NUC_X]) < 0).all()


@pytest.mark.parametrize(
    "implementation",
    [pytensor_implementation, c_implementation, pytensor_c_wrapper],
    ids=["pytensor", "c", "pytensor_c_wrapper"],
)
def test_implementations_match_numpy(implementation, slownesses):
    assert_allclose(
        implementation(slownesses),
        numpy_implementation(slownesses),
        rtol=0.0,
        atol=1e-6,
    )
