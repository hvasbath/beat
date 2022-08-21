import numpy as num

import voronoi_ext


def get_voronoi_cell_indexes_c(
    gf_points_dip, gf_points_strike, voronoi_points_dip, voronoi_points_strike
):
    """
    Do voronoi cell discretization and return idxs to cells.

    Parameters
    ----------
    gf_points_dip : :class:`numpy.NdArray`
        2d array, positions of gf_points along fault-dip-direction [m]
    gf_points_strike : :class:`numpy.NdArray`
        2d array, positions of gf_points along fault-strike-direction [m]
    voronoi_points_dip : :class:`numpy.NdArray`
        2d array, positions of voronoi_points along fault-dip-direction [m]
    voronoi_points_strike : :class:`numpy.NdArray`
        2d array, positions of voronoi_points along fault-strike-direction [m]

    Returns
    -------
    :class:`numpy.NdArray` with indexes to voronoi cells
    """

    return voronoi_ext.voronoi(
        gf_points_dip, gf_points_strike, voronoi_points_dip, voronoi_points_strike
    )


def get_voronoi_cell_indexes_numpy(
    gf_points_dip, gf_points_strike, voronoi_points_dip, voronoi_points_strike
):

    n_voros = voronoi_points_dip.size
    n_gfs = gf_points_dip.size

    gfs_dip_arr = num.tile(gf_points_dip, n_voros)
    gfs_strike_arr = num.tile(gf_points_strike, n_voros)

    voro_dips_arr = num.repeat(voronoi_points_dip, n_gfs)
    voro_strike_arr = num.repeat(voronoi_points_strike, n_gfs)

    distances = num.sqrt(
        (gfs_dip_arr - voro_dips_arr) ** 2.0 + (gfs_strike_arr - voro_strike_arr) ** 2.0
    ).reshape((n_voros, n_gfs))

    return distances.argmin(axis=0)
