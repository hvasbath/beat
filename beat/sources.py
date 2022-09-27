"""
Module that contains customized sources that can be used by the
pyrocko.gf.seismosizer.Engine.
Other specialized sources may be implemented here.
"""
import copy
import logging
import math

import numpy as num
from pyrocko import gf
from pyrocko import moment_tensor as mtm
from pyrocko.gf import meta
from pyrocko.gf.seismosizer import Source
from pyrocko.guts import Float

from beat.utility import get_rotation_matrix

# MTQT constants
pi = num.pi
pi4 = pi / 4.0
km = 1000.0
d2r = pi / 180.0
r2d = 180.0 / pi

SQRT3 = num.sqrt(3.0)
SQRT2 = num.sqrt(2.0)
SQRT6 = num.sqrt(6.0)

n = 1000
BETA_MAPPING = num.linspace(0, pi, n)
U_MAPPING = (
    (3.0 / 4.0 * BETA_MAPPING)
    - (1.0 / 2.0 * num.sin(2.0 * BETA_MAPPING))
    + (1.0 / 16.0 * num.sin(4.0 * BETA_MAPPING))
)

LAMPBDA_FACTOR_MATRIX = num.array(
    [[SQRT3, -1.0, SQRT2], [0.0, 2.0, SQRT2], [-SQRT3, -1.0, SQRT2]], dtype="float64"
)


logger = logging.getLogger("sources")


class RectangularSource(gf.RectangularSource):
    """
    Source for rectangular fault that unifies the necessary different source
    objects for teleseismic and geodetic computations.
    Reference point of the depth attribute is the top-center of the fault.

    Many of the methods of the RectangularSource have been modified from
    the HalfspaceTool from GertjanVanZwieten.
    """

    @property
    def dipvector(self):
        """
        Get 3 dimensional dip-vector of the planar fault.

        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.array(
            [
                num.cos(self.dip * d2r) * num.cos(self.strike * d2r),
                -num.cos(self.dip * d2r) * num.sin(self.strike * d2r),
                num.sin(self.dip * d2r),
            ]
        )

    @property
    def strikevector(self):
        """
        Get 3 dimensional strike-vector of the planar fault.

        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.array([num.sin(self.strike * d2r), num.cos(self.strike * d2r), 0.0])

    @property
    def normalvector(self):
        """
        Get 3 dimensional normal-vector of the planar fault.
        Returns
        -------
        :class:`numpy.ndarray`
        """

        return num.cross(self.strikevector, self.dipvector)

    @property
    def center(self):
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

        return (
            num.array([self.east_shift, self.north_shift, self.depth])
            + 0.5 * self.width * self.dipvector
        )

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

        return num.array([center[0], center[1], center[2]]) - (
            0.5 * self.width * self.dipvector
        )

    @property
    def bottom_center(self):
        """
        Bottom depth of the fault [m].
        (Patches method needs input depth to be top_depth.)

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        lower edge of the fault
        """

        return num.array([self.east_shift, self.north_shift, self.depth]) + (
            self.width * self.dipvector
        )

    @property
    def bottom_depth(self):
        return float(self.bottom_center[2])

    @property
    def bottom_left(self):
        return self.bottom_center - (0.5 * self.strikevector * self.length)

    @property
    def bottom_right(self):
        return self.bottom_center + (0.5 * self.strikevector * self.length)

    @property
    def top_left(self):
        return num.array([self.east_shift, self.north_shift, self.depth]) - (
            0.5 * self.strikevector * self.length
        )

    @property
    def top_right(self):
        return num.array([self.east_shift, self.north_shift, self.depth]) + (
            0.5 * self.strikevector * self.length
        )

    @property
    def corners(self):
        return num.vstack(
            [self.top_left, self.top_right, self.bottom_left, self.bottom_right]
        )

    def trace_center(self):
        """
        Get trace central coordinates of the fault [m] at the surface of the
        halfspace.

        Returns
        -------
        :class:`numpy.ndarray` with x, y, z coordinates of the central
        lower edge of the fault
        """

        bc = self.bottom_center
        xtrace = bc[0] - (bc[2] * num.cos(d2r * self.strike) / num.tan(d2r * self.dip))
        ytrace = bc[1] + (bc[2] * num.sin(d2r * self.strike) / num.tan(d2r * self.dip))
        return num.array([xtrace, ytrace, 0.0])

    def patches(self, nl, nw, datatype):
        """
        Cut source into n by m sub-faults and return n times m
        :class:`RectangularSource` Objects.
        Discretization starts at shallow depth going row-wise deeper.
        REQUIRES: self.depth to be TOP DEPTH!!! Returned faults also have depth
        reference at the top!

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
                sub_top = (
                    self.center2top_depth(self.center)
                    + self.strikevector * ((i + 0.5 - 0.5 * nl) * length)
                    + self.dipvector * (j * width)
                )

                patch = RectangularSource(
                    lat=float(self.lat),
                    lon=float(self.lon),
                    east_shift=float(sub_top[0]),
                    north_shift=float(sub_top[1]),
                    depth=float(sub_top[2]),
                    strike=self.strike,
                    dip=self.dip,
                    rake=self.rake,
                    length=length,
                    width=width,
                    stf=self.stf,
                    time=self.time,
                    slip=self.slip,
                    anchor="top",
                    opening_fraction=self.opening_fraction,
                )

                patches.append(patch)

        return patches

    def get_n_patches(self, patch_size=1000.0, dimension="length"):
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
        if dimension not in ["length", "width"]:
            raise ValueError("Invalid dimension!")

        n_p = num.round(self[dimension] / patch_size, decimals=4)
        return int(num.ceil(n_p))

    def extent_source(
        self, extension_width, extension_length, patch_width, patch_length
    ):
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

        length = self.length
        width = self.width

        if extension_length:
            new_length = (
                num.ceil((length + (2.0 * length * extension_length)) / km) * km
            )
            npl = int(num.ceil(new_length / patch_length))
            new_length = float(npl * patch_length)
        else:
            new_length = length

        if extension_width:
            new_width = num.ceil((width + (2.0 * width * extension_width)) / km) * km
            npw = int(num.ceil(new_width / patch_width))
            new_width = float(npw * patch_width)
        else:
            new_width = width

        logger.info("Fault extended to length=%f, width=%f!" % (new_length, new_width))

        orig_center = s.center

        s.update(length=new_length, width=new_width)

        top_center = s.center2top_depth(orig_center)

        if top_center[2] < 0.0:
            logger.info("Fault would intersect surface!" " Setting top center to 0.!")
            trace_center = s.trace_center()
            s.update(
                east_shift=float(trace_center[0]),
                north_shift=float(trace_center[1]),
                depth=float(trace_center[2]),
            )
        else:
            s.update(
                east_shift=float(top_center[0]),
                north_shift=float(top_center[1]),
                depth=float(top_center[2]),
            )

        return s

    @classmethod
    def from_kite_source(cls, source, kwargs):

        d = dict(
            lat=source.lat,
            lon=source.lon,
            north_shift=source.northing,
            east_shift=source.easting,
            depth=source.depth,
            width=source.width,
            length=source.length,
            strike=source.strike,
            dip=source.dip,
            rake=source.rake,
            slip=source.slip,
            anchor="top",
            **kwargs
        )

        if hasattr(source, "decimation_factor"):
            d["decimation_factor"] = source.decimation_factor

        return cls(**d)


def v_to_gamma(v):
    """
    Converts from v parameter (Tape2015) to lune longitude [rad]
    """
    return (1.0 / 3.0) * num.arcsin(3.0 * v)


def w_to_beta(w, u_mapping=None, beta_mapping=None, n=1000):
    """
    Converts from  parameter w (Tape2015) to lune co-latitude
    """
    if beta_mapping is None:
        beta_mapping = num.linspace(0, pi, n)

    if u_mapping is None:
        u_mapping = (
            (3.0 / 4.0 * beta_mapping)
            - (1.0 / 2.0 * num.sin(2.0 * beta_mapping))
            + (1.0 / 16.0 * num.sin(4.0 * beta_mapping))
        )
    return num.interp(3.0 * pi / 8.0 - w, u_mapping, beta_mapping)


def w_to_delta(w, n=1000):
    """
    Converts from parameter w (Tape2015) to lune latitude
    """
    beta = w_to_beta(w)
    return pi / 2.0 - beta


class MTQTSource(gf.SourceWithMagnitude):
    """
    A moment tensor point source.

    Notes
    -----
    Following Q-T parameterization after Tape & Tape 2015
    """

    discretized_source_class = meta.DiscretizedMTSource

    w = Float.T(
        default=0.0,
        help="Lune latitude delta transformed to grid. "
        "Defined: -3/8pi <= w <=3/8pi. "
        "If fixed to zero the MT is deviatoric.",
    )

    v = Float.T(
        default=0.0,
        help="Lune co-longitude transformed to grid. "
        "Defined: -1/3 <= v <= 1/3. "
        "If fixed to zero together with w the MT is pure DC.",
    )

    kappa = Float.T(
        default=0.0,
        help="Strike angle equivalent of moment tensor plane."
        "Defined: 0 <= kappa <= 2pi",
    )

    sigma = Float.T(
        default=0.0,
        help="Rake angle equivalent of moment tensor slip angle."
        "Defined: -pi/2 <= sigma <= pi/2",
    )

    h = Float.T(
        default=0.0,
        help="Dip angle equivalent of moment tensor plane." "Defined: 0 <= h <= 1",
    )

    def __init__(self, **kwargs):

        self.R = get_rotation_matrix()
        self.roty_pi4 = self.R["y"](-pi4)
        self.rotx_pi = self.R["x"](pi)

        self._lune_lambda_matrix = num.zeros((3, 3), dtype="float64")

        Source.__init__(self, **kwargs)

    @property
    def u(self):
        """
        Lunar co-latitude(beta), dependent on w
        """
        return (3.0 / 8.0) * num.pi - self.w

    @property
    def gamma(self):
        """
        Lunar longitude, dependent on v
        """
        return v_to_gamma(self.v)

    @property
    def beta(self):
        """
        Lunar co-latitude, dependent on u
        """
        return w_to_beta(self.w, u_mapping=U_MAPPING, beta_mapping=BETA_MAPPING)

    def delta(self):
        """
        From Tape & Tape 2012, delta measures departure of MT being DC
        Delta = Gamma = 0 yields pure DC
        """
        return (pi / 2.0) - self.beta

    @property
    def rho(self):
        return mtm.magnitude_to_moment(self.magnitude) * SQRT2

    @property
    def theta(self):
        return num.arccos(self.h)

    @property
    def rot_theta(self):
        return self.R["x"](self.theta)

    @property
    def rot_kappa(self):
        return self.R["z"](-self.kappa)

    @property
    def rot_sigma(self):
        return self.R["z"](self.sigma)

    @property
    def lune_lambda(self):
        sin_beta = num.sin(self.beta)
        cos_beta = num.cos(self.beta)
        sin_gamma = num.sin(self.gamma)
        cos_gamma = num.cos(self.gamma)
        vec = num.array([sin_beta * cos_gamma, sin_beta * sin_gamma, cos_beta])
        return 1.0 / SQRT6 * LAMPBDA_FACTOR_MATRIX.dot(vec) * self.rho

    @property
    def lune_lambda_matrix(self):
        num.fill_diagonal(self._lune_lambda_matrix, self.lune_lambda)
        return self._lune_lambda_matrix

    @property
    def rot_V(self):
        return self.rot_kappa.dot(self.rot_theta).dot(self.rot_sigma)

    @property
    def rot_U(self):
        return self.rot_V.dot(self.roty_pi4)

    @property
    def m9_nwu(self):
        """
        MT orientation is in NWU
        """
        return self.rot_U.dot(self.lune_lambda_matrix).dot(num.linalg.inv(self.rot_U))

    @property
    def m9(self):
        """
        Pyrocko MT in NEED
        """
        return self.rotx_pi.dot(self.m9_nwu).dot(self.rotx_pi.T)

    @property
    def m6(self):
        return mtm.to6(self.m9)

    @property
    def m6_astuple(self):
        return tuple(self.m6.ravel().tolist())

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, self.time
        )
        return meta.DiscretizedMTSource(
            m6s=self.m6[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times)
        )

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs
        )

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            logger.warning(
                "From event will ignore MT components initially. "
                "Needs mapping from NEED to QT space!"
            )
            # d.update(m6=list(map(float, mt.m6())))

        d.update(kwargs)
        return super(MTQTSource, cls).from_pyrocko_event(ev, **d)

    def get_derived_parameters(self, point=None, store=None, target=None, event=None):
        """
        Returns array with mt components and dc component conversions
        """
        scaled_m6 = self.m6 / self.moment
        mt = mtm.MomentTensor.from_values(scaled_m6)
        return num.hstack((scaled_m6, num.hstack(mt.both_strike_dip_rake())))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["R"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.R = get_rotation_matrix()


class MTSourceWithMagnitude(gf.SourceWithMagnitude):
    """
    A moment tensor point source.
    """

    discretized_source_class = meta.DiscretizedMTSource

    mnn = Float.T(default=1.0, help="north-north component of moment tensor")

    mee = Float.T(default=1.0, help="east-east component of moment tensor")

    mdd = Float.T(default=1.0, help="down-down component of moment tensor")

    mne = Float.T(default=0.0, help="north-east component of moment tensor")

    mnd = Float.T(default=0.0, help="north-down component of moment tensor")

    med = Float.T(default=0.0, help="east-down component of moment tensor")

    def __init__(self, **kwargs):
        if "m6" in kwargs:
            for (k, v) in zip("mnn mee mdd mne mnd med".split(), kwargs.pop("m6")):
                kwargs[k] = float(v)

        Source.__init__(self, **kwargs)

    @property
    def m6(self):
        return num.array(self.m6_astuple)

    @property
    def scaled_m6(self):
        m9 = mtm.symmat6(*self.m6)
        if isinstance(m9, num.matrix):
            m9 = m9.A

        m0_unscaled = math.sqrt(num.sum(m9**2)) / math.sqrt(2.0)
        m9 /= m0_unscaled
        m6 = mtm.to6(m9)
        return m6

    @property
    def scaled_m6_dict(self):
        keys = ["mnn", "mee", "mdd", "mne", "mnd", "med"]
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
            store.config.deltat, self.time
        )
        m0 = mtm.magnitude_to_moment(self.magnitude)
        m6s = self.scaled_m6 * m0
        return meta.DiscretizedMTSource(
            m6s=m6s[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times)
        )

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs
        )

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            d.update(m6=list(map(float, mt.m6())))

        d.update(kwargs)
        return super(MTSourceWithMagnitude, cls).from_pyrocko_event(ev, **d)

    def get_derived_parameters(self, point=None, store=None, target=None, event=None):
        mt = mtm.MomentTensor.from_values(self.scaled_m6)
        return num.hstack(mt.both_strike_dip_rake())
