import voronoi_ext


def get_voronoi_cell_indexes(
        gf_points_dip, gf_points_strike,
        voronoi_points_dip, voronoi_points_strike):
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
        gf_points_dip, gf_points_strike,
        voronoi_points_dip, voronoi_points_strike)
