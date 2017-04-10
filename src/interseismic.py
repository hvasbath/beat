"""
Module for interseismic models. Block-backslip model.
"""

from beat import utility
import numpy as num
import logging


logger = logging.getLogger('interseismic')

km = 1000.
d2r = num.pi / 180.
r2d = 180. / num.pi


def block_mask(x, y, strike):
    """
    Determine stable and moving observation points dependend on the input
    fault orientation.

    Parameters
    ----------
    x : :class:`numpy.array`
        east - local coordinates [m]
    y : :class:`numpy.array`
        north - local coordinates [m]
    strike : float
        fault strike [deg]

    Returns
    -------
    :class:`numpy.array` with zeros at stable points, ones at moving points
    """
    sv = utility.strike_vector(strike)[0:2]
    C = num.vstack([y.flatten(), x.flatten()]).T
    dots = num.dot(C, sv)
    dots[dots < 0.] = 0.
    dots[dots > 0.] = 1.
    return dots


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
    reference : ::class:`heart.ReferenceLocation`
        reference location that determines the stable block
    """



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
    return num.repeat(
        bmask * 2. * amplitude, 3).reshape((bmask.shape[0], 3)) * \
        utility.strike_vector(azimuth)


def geo_block_forward(lons, lats, sources, amplitude, azimuth, reference):
    return


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
    opening = amplitude * num.sin(alpha) * sdip

    if alphad < 90. and alphad >= 0.:
        rake = 0.
    elif alphad >= 90. and alphad <= 180.:
        rake = 180.
    else:
        raise Exception('Angle between vectors inconsistent!')

    width = locking_depth * km / sdip

    return dict(
        slip=slip, opening=opening, width=width, depth=0., rake=rake)
