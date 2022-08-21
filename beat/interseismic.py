"""
Module for interseismic models.

Block-backslip model
--------------------
The fault is assumed to be locked above a certain depth "locking_depth" and
it is creeping with the rate of the defined plate- which is handled as a
rigid block.

STILL EXPERIMENTAL!

References
==========
Savage & Prescott 1978
Metzger et al. 2011
"""

import copy
import logging

import numpy as num
from matplotlib.path import Path
from pyrocko.gf import RectangularSource as RS
from pyrocko.orthodrome import earthradius, latlon_to_ne_numpy, latlon_to_xyz

from beat import utility
from beat.heart import geo_synthetics

logger = logging.getLogger("interseismic")

km = 1000.0
d2r = num.pi / 180.0
r2d = 180.0 / num.pi

non_source = set(["amplitude", "azimuth", "locking_depth"])


__all__ = ["geo_backslip_synthetics"]


def block_mask(easts, norths, sources, east_ref, north_ref):
    """
    Determine stable and moving observation points dependent on the input
    fault orientation.

    Parameters
    ----------
    easts : :class:`numpy.ndarray`
        east - local coordinates [m] of observations
    norths : :class:`numpy.ndarray`
        north - local coordinates [m] of observations
    sources : list
        of :class:`RectangularSource`
    east_ref : float
        east local coordinate [m] of stable reference
    north_ref : float
        north local coordinate [m] of stable reference

    Returns
    -------
    :class:`numpy.ndarray` with zeros at stable points, ones at moving points
    """

    def get_vertex(outlines, i, j):
        f1 = outlines[i]
        f2 = outlines[j]
        print(f1, f2)
        return utility.line_intersect(f1[0, :], f1[1, :], f2[0, :], f2[1, :])

    tol = 2.0 * km

    Eline = RS(
        east_shift=easts.max() + tol,
        north_shift=0.0,
        strike=0.0,
        dip=90.0,
        length=1 * km,
    )
    Nline = RS(
        east_shift=0.0,
        north_shift=norths.max() + tol,
        strike=90,
        dip=90.0,
        length=1 * km,
    )
    Sline = RS(
        east_shift=0.0,
        north_shift=norths.min() - tol,
        strike=90,
        dip=90.0,
        length=1 * km,
    )

    frame = [Nline, Eline, Sline]

    # collect frame lines
    outlines = []
    for source in sources + frame:
        outline = source.outline(cs="xy")
        outlines.append(utility.swap_columns(outline, 0, 1)[0:2, :])

    # get polygon vertices
    poly_vertices = []
    for i in range(len(outlines) - 1):
        poly_vertices.append(get_vertex(outlines, i, i + 1))
    else:
        poly_vertices.append(get_vertex(outlines, 0, -1))

    print(poly_vertices, outlines)
    polygon = Path(num.vstack(poly_vertices), closed=True)

    ens = num.vstack([easts.flatten(), norths.flatten()]).T
    ref_en = num.array([east_ref, north_ref]).flatten()
    print(ens)
    mask = polygon.contains_points(ens)

    if not polygon.contains_point(ref_en):
        return mask

    else:
        return num.logical_not(mask)


def block_geometry(lons, lats, sources, reference):
    """
    Construct block geometry determine stable and moving parts dependent
    on the reference location.

    Parameters
    ----------
    lons : :class:`num.ndarray`
        Longitudes [deg] of observation points
    lats : :class:`num.ndarray`
        Latitudes [deg] of observation points
    sources : list
        of RectangularFault objects
    reference : :class:`heart.ReferenceLocation`
        reference location that determines the stable block

    Returns
    -------
    :class:`num.ndarray`
        mask with zeros/ones for stable/moving observation points, respectively
    """

    norths, easts = latlon_to_ne_numpy(reference.lat, reference.lon, lats, lons)

    return block_mask(easts, norths, sources, east_ref=0.0, north_ref=0.0)


def block_movement(bmask, amplitude, azimuth):
    """
    Get block movements. Assumes one side of the model stable, therefore
    the moving side is moving 2 times the given amplitude.

    Parameters
    ----------
    bmask : :class:`numpy.ndarray`
        masked block determining stable and moving observation points
    amplitude : float
        slip [m] of the moving block
    azimuth : float
        azimuth-angle[deg] ergo direction of moving block towards North

    Returns
    -------
    :class:`numpy.ndarray`
         (n x 3) [North, East, Down] displacements [m]
    """

    tmp = num.repeat(bmask * 2.0 * float(amplitude), 3).reshape((bmask.shape[0], 3))
    sv = utility.strike_vector(float(azimuth), order="NEZ")
    return tmp * sv


def geo_block_synthetics(lons, lats, sources, amplitude, azimuth, reference):
    """
    Block model: forward model for synthetic displacements(n,e,d) [m] caused by
    a rigid moving block defined by the bounding geometry of rectangular
    faults. The reference location determines the stable regions.
    The amplitude and azimuth determines the amount and direction of the
    moving block.

    Parameters
    ----------
    lons : :class:`num.ndarray`
        Longitudes [deg] of observation points
    lats : :class:`num.ndarray`
        Latitudes [deg] of observation points
    sources : list
        of RectangularFault objects
    amplitude : float
        slip [m] of the moving block
    azimuth : float
        azimuth-angle[deg] ergo direction of moving block towards North
    reference : :class:`heart.ReferenceLocation`
        reference location that determines the stable block

    Returns
    -------
    :class:`numpy.ndarray`
         (n x 3) [North, East, Down] displacements [m]
    """
    bmask = block_geometry(lons, lats, sources, reference)
    return block_movement(bmask, amplitude, azimuth)


def backslip_params(azimuth, strike, dip, amplitude, locking_depth):
    """
    Transforms the interseismic blockmodel parameters to fault input parameters
    for the backslip model.

    Parameters
    ----------
    azimuth : float
        azimuth [deg] of the block-motion towards the North
    strike : float
        strike-angle[deg] of the backslipping fault
    dip : float
        dip-angle[deg] of the back-slipping fault
    amplitude : float
        slip rate of the blockmodel [m/yr]
    locking_depth : float
        locking depth [km] of the fault

    Returns
    -------
    dict of parameters for the back-slipping RectangularSource
    """
    if dip == 0.0:
        raise ValueError("Dip must not be zero!")

    az_vec = utility.strike_vector(azimuth)
    strike_vec = utility.strike_vector(strike)
    alpha = num.arccos(az_vec.dot(strike_vec))
    alphad = alpha * r2d

    sdip = num.sin(dip * d2r)

    # assuming dip-slip is zero --> strike slip = slip
    slip = num.abs(amplitude * num.cos(alpha))
    opening = -amplitude * num.sin(alpha) * sdip

    if alphad < 90.0 and alphad >= 0.0:
        rake = 0.0
    elif alphad >= 90.0 and alphad <= 180.0:
        rake = 180.0
    else:
        raise Exception("Angle between vectors inconsistent!")

    width = locking_depth * km / sdip

    return dict(
        slip=float(slip),
        opening=float(opening),
        width=float(width),
        depth=0.0,
        rake=float(rake),
    )


def geo_backslip_synthetics(
    engine, sources, targets, lons, lats, reference, amplitude, azimuth, locking_depth
):
    """
    Interseismic backslip model: forward model for synthetic
    displacements(n,e,d) [m] caused by a rigid moving block defined by the
    bounding geometry of rectangular faults. The reference location determines
    the stable regions. The amplitude and azimuth determines the amount and
    direction of the moving block.
    Based on this block-movement the upper part of the crust that is not locked
    is assumed to slip back. Thus the final synthetics are the superposition
    of the block-movement and the backslip.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
    sources : list
        of :class:`pyrocko.gf.seismosizer.RectangularSource`
        Sources to calculate synthetics for
    targets : list
        of :class:`pyrocko.gf.targets.StaticTarget`
    lons : list of floats, or :class:`numpy.ndarray`
        longitudes [deg] of observation points
    lats : list of floats, or :class:`numpy.ndarray`
        latitudes [deg] of observation points
    amplitude : float
        slip [m] of the moving block
    azimuth : float
        azimuth-angle[deg] ergo direction of moving block towards North
    locking_depth : :class:`numpy.ndarray`
        locking_depth [km] of the fault(s) below there is no movement
    reference : :class:`heart.ReferenceLocation`
        reference location that determines the stable block

    Returns
    -------
    :class:`numpy.ndarray`
         (n x 3) [North, East, Down] displacements [m]
    """

    disp_block = geo_block_synthetics(
        lons, lats, sources, amplitude, azimuth, reference
    )

    for source, ld in zip(sources, locking_depth):
        source_params = backslip_params(
            azimuth=azimuth,
            amplitude=amplitude,
            locking_depth=ld,
            strike=source.strike,
            dip=source.dip,
        )
        source.update(**source_params)

    disp_block += geo_synthetics(
        engine=engine, targets=targets, sources=sources, outmode="stacked_array"
    )

    return disp_block


def seperate_point(point):
    """
    Separate point into source object related components and the rest.
    """
    tpoint = copy.deepcopy(point)

    interseismic_point = {}
    for var in non_source:
        if var in tpoint.keys():
            interseismic_point[var] = tpoint.pop(var)

    return tpoint, interseismic_point
