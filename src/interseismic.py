"""
Module for interseismic models. Block-backslip model.
"""

from beat import utility
from beat.heart import geo_layer_synthetics

import numpy as num
import logging
import copy

from pyrocko import orthodrome


logger = logging.getLogger('interseismic')

km = 1000.
d2r = num.pi / 180.
r2d = 180. / num.pi

non_source = set(['amplitude', 'azimuth', 'locking_depth'])


def block_mask(easts, norths, strike, east_ref, north_ref):
    """
    Determine stable and moving observation points dependend on the input
    fault orientation.

    Parameters
    ----------
    easts : :class:`numpy.array`
        east - local coordinates [m] of observations
    norths : :class:`numpy.array`
        north - local coordinates [m] of observations
    strike : float
        fault strike [deg]
    east_ref : float
        east local coordinate [m] of stable reference
    north_ref : float
        north local coordinate [m] of stable reference

    Returns
    -------
    :class:`numpy.array` with zeros at stable points, ones at moving points
    """

    sv = utility.strike_vector(-strike)[0:2]
    nes = num.vstack([norths.flatten(), easts.flatten()]).T
    ref_ne = num.array([north_ref, east_ref]).flatten()
    reference = num.dot(ref_ne, sv)
    mask = num.dot(nes, sv)

    if reference < 0:
        mask[mask < 0.] = 0.
        mask[mask > 0.] = 1.
    elif reference > 0:
        mask[mask > 0.] = 0.
        mask[mask < 0.] = 1.
    else:
        logger.warn(
            'The stable reference location lies on the prolongation of a fault'
            'ambiguous stability! Assuming stable! Re-check necessary!')
        mask[mask < 0.] = 0.
        mask[mask > 0.] = 1.

    return mask


def block_geometry(lons, lats, sources, reference):
    """
    Construct block geometry determine stable and moving parts dependend
    on the reference location.

    Parameters
    ----------
    lons : :class:`num.array`
        Longitudes [deg] of observation points
    lats : :class:`num.array`
        Latitudes [deg] of observation points
    sources : list
        of RectangularFault objects
    reference : :class:`heart.ReferenceLocation`
        reference location that determines the stable block

    Returns
    -------
    :class:`num.array`
        mask with zeros/ones for stable/moving observation points, respectively
    """

    bmask = num.zeros_like(lons)
    for source in sources:
        norths, easts = orthodrome.latlon_to_ne_numpy(
            source.effective_lat, source.effective_lon, lats, lons)
        north_ref, east_ref = orthodrome.latlon_to_ne_numpy(
            source.effective_lat,
            source.effective_lon,
            reference.lat,
            reference.lon)
        bmask += block_mask(easts, norths, source.strike, east_ref, north_ref)

    # reset points that are moving to one
    bmask[bmask > 0] = 1
    return bmask


def block_movement(bmask, amplitude, azimuth):
    """
    Get block movements. Assumes one side of the model stable, therefore
    the moving side is moving 2 times the given amplitude.

    Parameters
    ----------
    bmask : :class:`numpy.array`
        masked block determining stable and moving observation points
    amplitude : float
        slip [m] of the moving block
    azimuth : float
        azimuth-angle[deg] ergo direction of moving block towards North

    Returns
    -------
    :class:`numpy.array`
         (n x 3) [North, East, Down] displacements [m]
    """

    tmp = num.repeat(
        bmask * 2. * float(amplitude), 3).reshape((bmask.shape[0], 3))
    sv = utility.strike_vector(float(azimuth), order='NEZ')
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
    lons : :class:`num.array`
        Longitudes [deg] of observation points
    lats : :class:`num.array`
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
    :class:`numpy.array`
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
    if dip == 0.:
        raise ValueError('Dip must not be zero!')

    az_vec = utility.strike_vector(azimuth)
    strike_vec = utility.strike_vector(strike)
    alpha = num.arccos(az_vec.dot(strike_vec))
    alphad = alpha * r2d

    sdip = num.sin(dip * d2r)

    # assuming dip-slip is zero --> strike slip = slip
    slip = num.abs(amplitude * num.cos(alpha))
    opening = -amplitude * num.sin(alpha) * sdip

    if alphad < 90. and alphad >= 0.:
        rake = 0.
    elif alphad >= 90. and alphad <= 180.:
        rake = 180.
    else:
        raise Exception('Angle between vectors inconsistent!')

    width = locking_depth * km / sdip

    return dict(
        slip=float(slip), opening=float(opening), width=float(width),
        depth=0., rake=float(rake))


def geo_backslip_synthetics(
    store_superdir, crust_ind, sources, lons, lats, reference,
    amplitude, azimuth, locking_depth):
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
    store_superdir : str
        main path to directory containing the different Greensfunction stores
    crust_ind : int
        index of Greens Function store to use
    lons : List of floats
        Longitudes [decimal deg] of observation points
    lats : List of floats
        Latitudes [decimal deg] of observation points
    sources : List of :class:`pscmp.PsCmpRectangularSource`
        Sources to calculate synthetics for
    amplitude : float
        slip [m] of the moving block
    azimuth : float
        azimuth-angle[deg] ergo direction of moving block towards North
    locking_depth : :class:`numpy.array`
        locking_depth [km] of the fault(s) below there is no movement
    reference : :class:`heart.ReferenceLocation`
        reference location that determines the stable block

    Returns
    -------
    :class:`numpy.array`
         (n x 3) [North, East, Down] displacements [m]
    """

    disp_block = geo_block_synthetics(
        lons, lats, sources, amplitude, azimuth, reference)

    for source, ld in zip(sources, locking_depth):
        source_params = backslip_params(
            azimuth=azimuth, amplitude=amplitude, locking_depth=ld,
            strike=source.strike, dip=source.dip)
        source.update(**source_params)

    disp_block += geo_layer_synthetics(
        store_superdir, crust_ind, lons, lats, sources)

    return disp_block


def seperate_point(point):
    """
    Seperate point into source object related components and the rest.
    """
    tpoint = copy.deepcopy(point)

    interseismic_point = {}
    for var in non_source:
        if var in tpoint.keys():
            interseismic_point[var] = tpoint.pop(var)

    return tpoint, interseismic_point
