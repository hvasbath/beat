import numpy as num
from pyrocko.guts import Float, Int, Object, Dict, String, Tuple, load, dump
from pyrocko.config import expand
from pyrocko import util
import logging
import os

logger = logging.getLogger("pyrocko.config")

guts_prefix = "pf"

beat_dir_tmpl = os.environ.get("BEAT_DIR", os.path.join("~", ".beat"))


SQRT2 = num.sqrt(2)


def hypername(varname):
    if varname in plot_units.keys():
        return varname
    else:
        return "h_"


class Bounds(Object):

    default_bounds = Tuple.T(2, Float.T(), default=(0, 1))
    physical_bounds = Tuple.T(2, Float.T(), default=(0, 1))
    unit = String.T(default="$[m]$")


class ParameterDefaults(Object):

    parameters = Dict.T(String.T(), Bounds.T())

    def __getitem__(self, k):
        if k not in self.parameters.keys():
            raise KeyError(k)

        return self.parameters[k]


sf_force = (0, 1e10)
moffdiag = (-1.0, 1.0)
mdiag = (-SQRT2, SQRT2)
# TODO cleanup dicts into a single dict
default_bounds = dict(
    east_shift=(-10.0, 10.0),
    north_shift=(-10.0, 10.0),
    depth=(0.0, 5.0),
    strike=(0, 180.0),
    strike1=(0, 180.0),
    strike2=(0, 180.0),
    dip=(45.0, 90.0),
    dip1=(45.0, 90.0),
    dip2=(45.0, 90.0),
    rake=(-90.0, 90.0),
    rake1=(-90.0, 90.0),
    rake2=(-90.0, 90.0),
    length=(5.0, 30.0),
    width=(5.0, 20.0),
    slip=(0.1, 8.0),
    nucleation_x=(-1.0, 1.0),
    nucleation_y=(-1.0, 1.0),
    opening_fraction=(0.0, 0.0),
    magnitude=(4.0, 7.0),
    mnn=mdiag,
    mee=mdiag,
    mdd=mdiag,
    mne=moffdiag,
    mnd=moffdiag,
    med=moffdiag,
    fn=sf_force,
    fe=sf_force,
    fd=sf_force,
    exx=(-200.0, 200.0),
    eyy=(-200.0, 200.0),
    exy=(-200.0, 200.0),
    rotation=(-200.0, 200.0),
    kappa=(0.0, 2 * num.pi),
    sigma=(-num.pi / 2.0, num.pi / 2.0),
    h=(0.0, 1.0),
    v=(-1.0 / 3, 1.0 / 3.0),
    w=(-3.0 / 8.0 * num.pi, 3.0 / 8.0 * num.pi),
    volume_change=(1e8, 1e10),
    diameter=(5.0, 10.0),
    sign=(-1.0, 1.0),
    mix=(0, 1),
    time=(-5.0, 5.0),
    time_shift=(-5.0, 5.0),
    delta_time=(0.0, 10.0),
    delta_depth=(0.0, 10.0),
    distance=(0.0, 10.0),
    duration=(1.0, 30.0),
    peak_ratio=(0.0, 1.0),
    durations=(0.5, 29.5),
    uparr=(-0.05, 6.0),
    uperp=(-0.3, 4.0),
    utens=(0.0, 0.0),
    nucleation_strike=(0.0, 10.0),
    nucleation_dip=(0.0, 7.0),
    velocities=(0.5, 4.2),
    azimuth=(0, 180),
    amplitude=(1e10, 1e20),
    locking_depth=(1.0, 10.0),
    hypers=(-2.0, 6.0),
    ramp=(-0.005, 0.005),
    offset=(-0.05, 0.05),
    lat=(30.0, 30.5),
    lon=(30.0, 30.5),
    omega=(0.5, 0.6),
    strike_traction=(0, 1e9),
    dip_traction=(0, 1e9),
    tensile_traction=(0, 1e9),
    major_axis=(0.01, 10),
    minor_axis=(0.01, 10),
    major_axis_bottom=(0.01, 10),
    minor_axis_bottom=(0.01, 10),
    plunge=(0, 20),
    delta_east_shift_bottom=(-10, 10),
    delta_north_shift_bottom=(-10, 10),
    delta_depth_bottom=(-1, 1),
)

default_seis_std = 1.0e-6
default_geo_std = 1.0e-3

default_decimation_factors = {"polarity": 1, "geodetic": 4, "seismic": 2}


physical_bounds = dict(
    east_shift=(-500.0, 500.0),
    north_shift=(-500.0, 500.0),
    depth=(0.0, 1000.0),
    strike=(-90.0, 420.0),
    strike1=(-90.0, 420.0),
    strike2=(-90.0, 420.0),
    dip=(-45.0, 135.0),
    dip1=(-45.0, 135.0),
    dip2=(-45.0, 135.0),
    rake=(-180.0, 270.0),
    rake1=(-180.0, 270.0),
    rake2=(-180.0, 270.0),
    mix=(0, 1),
    diameter=(0.0, 100.0),
    sign=(-1.0, 1.0),
    volume_change=(-1e12, 1e12),
    fn=(-1e20, 1e20),
    fe=(-1e20, 1e20),
    fd=(-1e20, 1e20),
    mnn=(-SQRT2, SQRT2),
    mee=(-SQRT2, SQRT2),
    mdd=(-SQRT2, SQRT2),
    mne=(-1.0, 1.0),
    mnd=(-1.0, 1.0),
    med=(-1.0, 1.0),
    exx=(-500.0, 500.0),
    eyy=(-500.0, 500.0),
    exy=(-500.0, 500.0),
    rotation=(-500.0, 500.0),
    kappa=(0.0, 2 * num.pi),
    sigma=(-num.pi / 2.0, num.pi / 2.0),
    h=(0.0, 1.0),
    length=(0.0, 7000.0),
    width=(0.0, 500.0),
    slip=(0.0, 150.0),
    nucleation_x=(-1.0, 1.0),
    nucleation_y=(-1.0, 1.0),
    opening_fraction=(-1.0, 1.0),
    magnitude=(-5.0, 10.0),
    time=(-300.0, 300.0),
    time_shift=(-40.0, 40.0),
    delta_time=(0.0, 100.0),
    delta_depth=(0.0, 300.0),
    distance=(0.0, 300.0),
    duration=(0.0, 600.0),
    peak_ratio=(0.0, 1.0),
    durations=(0.0, 600.0),
    uparr=(-1.0, 150.0),
    uperp=(-150.0, 150.0),
    utens=(-150.0, 150.0),
    nucleation_strike=(0.0, num.inf),
    nucleation_dip=(0.0, num.inf),
    velocities=(0.0, 20.0),
    azimuth=(0, 360),
    amplitude=(1.0, 10e25),
    locking_depth=(0.1, 100.0),
    hypers=(-20.0, 20.0),
    ramp=(-0.01, 0.01),
    offset=(-1.0, 1.0),
    lat=(-90.0, 90.0),
    lon=(-180.0, 180.0),
    omega=(-10.0, 10.0),
    v=(-1.0 / 3, 1.0 / 3.0),
    w=(-3.0 / 8.0 * num.pi, 3.0 / 8.0 * num.pi),
    strike_traction=(0, 1e15),
    dip_traction=(0, 1e15),
    tensile_traction=(0, 1e15),
    major_axis=(0.01, 100),
    minor_axis=(0.01, 100),
    major_axis_bottom=(0.01, 100),
    minor_axis_bottom=(0.01, 100),
    plunge=(0, 90),
    delta_east_shift_bottom=(-500, 500),
    delta_north_shift_bottom=(-500, 500),
    delta_depth_bottom=(-20, 20),
)

u_n = "$[N]$"
u_nm = "$[Nm]$"
u_km = "$[km]$"
u_km_s = "$[km/s]$"
u_deg = "$[^{\circ}]$"
u_deg_myr = "$[^{\circ} / myr]$"
u_m = "$[m]$"
u_v = "$[m^3]$"
u_s = "$[s]$"
u_rad = "$[rad]$"
u_hyp = ""
u_percent = "[$\%$]"
u_nanostrain = "nstrain"
u_mpa = "[$[MPa]$]"

plot_units = {
    "east_shift": u_km,
    "north_shift": u_km,
    "depth": u_km,
    "width": u_km,
    "length": u_km,
    "dip": u_deg,
    "dip1": u_deg,
    "dip2": u_deg,
    "strike": u_deg,
    "strike1": u_deg,
    "strike2": u_deg,
    "rake": u_deg,
    "rake1": u_deg,
    "rake2": u_deg,
    "mix": u_hyp,
    "volume_change": u_v,
    "diameter": u_km,
    "slip": u_m,
    "opening_fraction": u_hyp,
    "azimuth": u_deg,
    "amplitude": u_nm,
    "locking_depth": u_km,
    "nucleation_dip": u_km,
    "nucleation_strike": u_km,
    "nucleation_x": u_hyp,
    "nucleation_y": u_hyp,
    "time_shift": u_s,
    "coupling": u_percent,
    "uperp": u_m,
    "uparr": u_m,
    "utens": u_m,
    "durations": u_s,
    "velocities": u_km_s,
    "fn": u_n,
    "fe": u_n,
    "fd": u_n,
    "mnn": u_nm,
    "mee": u_nm,
    "mdd": u_nm,
    "mne": u_nm,
    "mnd": u_nm,
    "med": u_nm,
    "magnitude": u_hyp,
    "eps_xx": u_nanostrain,
    "eps_yy": u_nanostrain,
    "eps_xy": u_nanostrain,
    "rotation": u_nanostrain,
    "pole_lat": u_deg,
    "pole_lon": u_deg,
    "omega": u_deg_myr,
    "w": u_rad,
    "v": u_rad,
    "kappa": u_deg,
    "sigma": u_deg,
    "h": u_deg,
    "distance": u_km,
    "delta_depth": u_km,
    "delta_time": u_s,
    "time": u_s,
    "duration": u_s,
    "peak_ratio": u_hyp,
    "h_": u_hyp,
    "like": u_hyp,
    "strike_traction": u_mpa,
    "dip_traction": u_mpa,
    "tensile_traction": u_mpa,
    "major_axis": u_km,
    "minor_axis": u_km,
    "major_axis_bottom": u_km,
    "minor_axis_bottom": u_km,
    "plunge": u_deg,
    "delta_east_shift_bottom": u_km,
    "delta_north_shift_bottom": u_km,
    "delta_depth_bottom": u_km,
}


def make_path_tmpl(name="defaults"):
    return os.path.join(beat_dir_tmpl, "%s.pf" % name)


def init_parameter_defaults():

    defaults = ParameterDefaults()

    for parameter_name, dbound in default_bounds.items():
        pbound = physical_bounds[parameter_name]
        unit = plot_units[hypername(parameter_name)]
        defaults.parameters[parameter_name] = Bounds(
            physical_bounds=pbound, default_bounds=dbound, unit=unit
        )

    return defaults


def get_defaults(force=True):
    defaults_path = expand(make_path_tmpl())
    if not os.path.exists(defaults_path) or force:
        defaults = init_parameter_defaults()
        util.ensuredirs(defaults_path)
        dump(defaults, filename=defaults_path)
    else:
        defaults = load(filename=defaults_path)

    return defaults


defaults = get_defaults()
