import numpy as num
from numpy.testing import assert_allclose

from beat.utility import get_random_uniform
from beat.voronoi import voronoi

km = 1000.0
N_GFS = 1000
N_VORO = 100
FAULT_LENGTH = 30.0 * km
FAULT_WIDTH = 10.0 * km


def test_voronoi_discretization():
    gf_dip = get_random_uniform(0.0, FAULT_WIDTH, dimension=N_GFS)
    gf_strike = get_random_uniform(0.0, FAULT_LENGTH, dimension=N_GFS)
    voro_dip = get_random_uniform(0.0, FAULT_WIDTH, dimension=N_VORO)
    voro_strike = get_random_uniform(0.0, FAULT_LENGTH, dimension=N_VORO)

    idxs_c = voronoi.get_voronoi_cell_indexes_c(
        gf_dip, gf_strike, voro_dip, voro_strike
    )
    idxs_numpy = voronoi.get_voronoi_cell_indexes_numpy(
        gf_dip, gf_strike, voro_dip, voro_strike
    )

    assert_allclose(idxs_c, idxs_numpy, rtol=0.0, atol=1e-6)

    # brute force nearest voronoi node as reference
    dists = num.hypot(
        gf_dip[:, None] - voro_dip[None, :], gf_strike[:, None] - voro_strike[None, :]
    )
    assert_allclose(idxs_numpy, dists.argmin(axis=1))
