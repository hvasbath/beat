"""
Module that contains customized sources that can be used by the
pyrocko.gf.seismosizer.Engine.
Other specialized sources may be implemented here.
"""
from pyrocko.guts import Float
from pyrocko import gf
from pyrocko.gf import meta
from pyrocko import moment_tensor as mtm
from pyrocko.gf.seismosizer import outline_rect_source, Source, Cloneable
from pyrocko.orthodrome import ne_to_latlon

import math
import copy
import numpy as num
import logging

km = 1000.
d2r = num.pi / 180.
r2d = 180. / num.pi

logger = logging.getLogger('sources')


class RectangularSource(gf.DCSource, Cloneable):
    """
    Source for rectangular fault that unifies the necessary different source
    objects for teleseismic and geodetic computations.
    Reference point of the depth attribute is the top-center of the fault.

    Many of the methods of the RectangularSource have been modified from
    the HalfspaceTool from GertjanVanZwieten.
    """

    width = Float.T(help='width of the fault [m]',
                    default=1. * km)
    length = Float.T(help='length of the fault [m]',
                    default=1. * km)
    slip = Float.T(help='slip of the fault [m]',
                    default=1.)
    opening = Float.T(help='opening of the fault [m]',
                    default=0.)

    @property
    def dipvector(self):
        """
        Get 3 dimensional dip-vector of the planar fault.

        Parameters
        ----------
        dip : scalar, float
            dip-angle [deg] of the fault
        strike : scalar, float
            strike-abgle [deg] of the fault

        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.array(
            [num.cos(self.dip * d2r) * num.cos(self.strike * d2r),
             -num.cos(self.dip * d2r) * num.sin(self.strike * d2r),
              num.sin(self.dip * d2r)])

    @property
    def strikevector(self):
        """
        Get 3 dimensional strike-vector of the planar fault.

        Parameters
        ----------
        strike : scalar, float
            strike-abgle [deg] of the fault

        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.array(
            [num.sin(self.strike * d2r),
             num.cos(self.strike * d2r),
             0.])

    def center(self, width):
        """
        Get 3d fault center coordinates. Depth attribute is top depth!

        Parameters
        ----------
        width : scalar, float
            width [m] of the fault (dip-direction)

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the center of the
        fault
        """

        return num.array([self.east_shift, self.north_shift, self.depth]) + \
            0.5 * width * self.dipvector

    def center2top_depth(self, center):
        """
        Get top depth of the fault [m] given a potential center point.
        (Patches method needs input depth to
        be top_depth.)

        Parameters
        ----------
        center : vector[x, y, z], float
            coordinates [m] of the center of the fault

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        upper edge of the fault
        """

        return num.array([center[0], center[1], center[2]]) - \
            0.5 * self.width * self.dipvector

    def bottom_depth(self, depth):
        """
        Get bottom depth of the fault [m].
        (Patches method needs input depth to be top_depth.)

        Parameters
        ----------
        depth : scalar, float
            depth [m] of the center of the fault

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        lower edge of the fault
        """

        return num.array([self.east_shift, self.north_shift, depth]) + \
            0.5 * self.width * self.dipvector

    def trace_center(self, depth):
        """
        Get trace central coordinates of the fault [m] at the surface of the
        halfspace.

        Parameters
        ----------
        depth : scalar, float
            depth [m] of the center of the fault

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        lower edge of the fault
        """

        bd = self.bottom_depth(depth)
        xtrace = bd[0] - \
            (bd[2] * num.cos(d2r * self.strike) / num.tan(d2r * self.dip))
        ytrace = bd[1] + \
            (bd[2] * num.sin(d2r * self.strike) / num.tan(d2r * self.dip))
        return num.array([xtrace, ytrace, 0.])

    def patches(self, nl, nw, datatype):
        """
        Cut source into n by m sub-faults and return n times m
        :class:`RectangularSource` Objects.
        Discretization starts at shallow depth going row-wise deeper.
        REQUIRES: self.depth to be TOP DEPTH!!!

        Parameters
        ----------
        nl : int
            number of patches in length direction (strike)
        nw : int
            number of patches in width direction (dip)
        datatype : string
            'geodetic' or 'seismic' determines the source to be returned

        Returns
        -------
        :class:`pscmp.PsCmpRectangularSource` or
        :class:`pyrocko.gf.seismosizer.RectangularSource` depending on
        datatype. Depth is being updated from top_depth to center_depth.
        """

        length = self.length / float(nl)
        width = self.width / float(nw)

        patches = []
        for j in range(nw):
            for i in range(nl):
                sub_center = self.center(self.width) + \
                    self.strikevector * ((i + 0.5 - 0.5 * nl) * length) + \
                    self.dipvector * ((j + 0.5 - 0.5 * nw) * width)

                patch = gf.RectangularSource(
                    lat=float(self.lat),
                    lon=float(self.lon),
                    east_shift=float(sub_center[0]),
                    north_shift=float(sub_center[1]),
                    depth=float(sub_center[2]),
                    strike=self.strike, dip=self.dip, rake=self.rake,
                    length=length, width=width, stf=self.stf,
                    time=self.time, slip=self.slip, anchor='center')

                if nw == 1 and nl == 1:
                    logger.warn(
                        'RectangularSource for fault-geometry inversion'
                        ' decimated!')
                    if datatype == 'seismic':
                        patch.decimation_factor = 20

                    elif datatype == 'geodetic':
                        patch.decimation_factor = 7

                else:
                    raise TypeError(
                        "Datatype not supported either: 'seismic/geodetic'")

                patches.append(patch)

        return patches

    def outline(self, cs='xyz'):
        points = outline_rect_source(self.strike, self.dip, self.length,
                                     self.width)
        center = self.center(self.width)
        points[:, 0] += center[0]
        points[:, 1] += center[1]
        points[:, 2] += center[2]
        if cs == 'xyz':
            return points
        elif cs == 'xy':
            return points[:, :2]
        elif cs in ('latlon', 'lonlat'):
            latlon = ne_to_latlon(
                self.lat, self.lon, points[:, 0], points[:, 1])

            latlon = num.array(latlon).T
            if cs == 'latlon':
                return latlon
            else:
                return latlon[:, ::-1]

    def get_n_patches(self, patch_size=1000., dimension='length'):
        """
        Return number of patches along dimension of the fault.

        Parameters
        ----------
        patch_size : float
            patch size [m] of desired sub-patches
        dimension : str

        Returns
        -------
        int
        """
        if dimension not in ['length', 'width']:
            raise Exception('Invalid dimension!')

        return int(num.ceil(self[dimension] / patch_size))

    def extent_source(self, extension_width, extension_length,
                     patch_width, patch_length):
        """
        Extend fault into all directions. Rounds dimensions to have no
        half-patches.

        Parameters
        ----------
        extension_width : float
            factor to extend source in width (dip-direction)
        extension_length : float
            factor extend source in length (strike-direction)
        patch_width : float
            Width [m] of subpatch in dip-direction
        patch_length : float
            Length [m] of subpatch in strike-direction

        Returns
        -------
        dict with list of :class:`pscmp.PsCmpRectangularSource` or
            :class:`pyrocko.gf.seismosizer.RectangularSource`
        """
        s = copy.deepcopy(self)

        l = self.length
        w = self.width

        new_length = num.ceil((l + (2. * l * extension_length)) / km) * km
        new_width = num.ceil((w + (2. * w * extension_length)) / km) * km

        npl = int(num.ceil(new_length / patch_length))
        npw = int(num.ceil(new_width / patch_width))

        new_length = float(npl * patch_length)
        new_width = float(npw * patch_width)
        logger.info(
            'Fault extended to length=%f, width=%f!' % (new_length, new_width))

        orig_center = s.center(s.width)
        s.update(length=new_length, width=new_width)

        top_center = s.center2top_depth(orig_center)

        if top_center[2] < 0.:
            logger.info('Fault would intersect surface!'
                        ' Setting top center to 0.!')
            trace_center = s.trace_center(s.depth)
            s.update(east_shift=float(trace_center[0]),
                     north_shift=float(trace_center[1]),
                     depth=float(trace_center[2]))
        else:
            s.update(east_shift=float(top_center[0]),
                     north_shift=float(top_center[1]),
                     depth=float(top_center[2]))

        return s


class MTSourceWithMagnitude(gf.SourceWithMagnitude):
    '''
    A moment tensor point source.
    '''

    discretized_source_class = meta.DiscretizedMTSource

    mnn = Float.T(
        default=1.,
        help='north-north component of moment tensor')

    mee = Float.T(
        default=1.,
        help='east-east component of moment tensor')

    mdd = Float.T(
        default=1.,
        help='down-down component of moment tensor')

    mne = Float.T(
        default=0.,
        help='north-east component of moment tensor')

    mnd = Float.T(
        default=0.,
        help='north-down component of moment tensor')

    med = Float.T(
        default=0.,
        help='east-down component of moment tensor')

    def __init__(self, **kwargs):
        if 'm6' in kwargs:
            for (k, v) in zip('mnn mee mdd mne mnd med'.split(),
                              kwargs.pop('m6')):
                kwargs[k] = float(v)

        Source.__init__(self, **kwargs)

    @property
    def m6(self):
        return num.array(self.m6_astuple)

    @property
    def scaled_m6(self):
        m9 = mtm.symmat6(*self.m6)
        m0_unscaled = math.sqrt(num.sum(m9.A ** 2)) / math.sqrt(2.)
        m9 /= m0_unscaled
        m6 = mtm.to6(m9)
        return m6

    @property
    def scaled_m6_dict(self):
        keys = ['mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']
        return {k: m for k, m in zip(keys, self.scaled_m6.tolist())}

    @property
    def m6_astuple(self):
        return (self.mnn, self.mee, self.mdd, self.mne, self.mnd, self.med)

    @m6.setter
    def m6(self, value):
        self.mnn, self.mee, self.mdd, self.mne, self.mnd, self.med = value

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, 0.0)
        m0 = mtm.magnitude_to_moment(self.magnitude)
        m6s = self.scaled_m6 * m0
        return meta.DiscretizedMTSource(
            m6s=m6s[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times))

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            d.update(m6=map(float, mt.m6()))

        d.update(kwargs)
        return super(MTSourceWithMagnitude, cls).from_pyrocko_event(ev, **d)
