import logging
import os

import numpy as num
from pyrocko import util
from pyrocko.config import expand
from pyrocko.guts import Dict, Float, Object, String, Tuple, dump, load

logger = logging.getLogger("pyrocko.config")

guts_prefix = "pf"

SQRT2 = num.sqrt(2)

default_seis_std = 1.0e-6
default_geo_std = 1.0e-3
default_decimation_factors = {"polarity": 1, "geodetic": 4, "seismic": 2}

beat_dir_tmpl = os.environ.get("BEAT_DIR", os.path.expanduser("~/.beat"))


class Bounds(Object):
    default_bounds = Tuple.T(2, Float.T(), default=(0, 1))
    physical_bounds = Tuple.T(2, Float.T(), default=(0, 1))
    unit = String.T(default=r"[m]")


class ParameterDefaults(Object):
    parameters = Dict.T(String.T(), Bounds.T())

    def __getitem__(self, k):
        if k not in self.parameters.keys():
            raise KeyError(k)
        return self.parameters[k]


sf_force = (0, 1e10)
moffdiag = (-1.0, 1.0)
mdiag = (-SQRT2, SQRT2)

u_n = r"$[N]$"
u_nm = r"$[Nm]$"
u_km = r"$[km]$"
u_km_s = r"$[km/s]$"
u_deg = r"$[^\circ]$"
u_deg_myr = r"$[^\circ/myr]$"
u_m = r"$[m]$"
u_v = r"$[m^3]$"
u_s = r"$[s]$"
u_rad = r"$[rad]$"
u_hyp = r""
u_percent = r"[$\%$]"
u_nanostrain = r"[nstrain]"
u_pa = r"$[MPa]$"

# Bounds and Units for all parameters
parameter_info = {
    "east_shift": Bounds(
        physical_bounds=(-500.0, 500.0), default_bounds=(-10.0, 10.0), unit=u_km
    ),
    "north_shift": Bounds(
        physical_bounds=(-500.0, 500.0), default_bounds=(-10.0, 10.0), unit=u_km
    ),
    "depth": Bounds(
        physical_bounds=(0.0, 1000.0), default_bounds=(0.0, 5.0), unit=u_km
    ),
    "strike": Bounds(
        physical_bounds=(-90.0, 420.0), default_bounds=(0, 180.0), unit=u_deg
    ),
    "strike1": Bounds(
        physical_bounds=(-90.0, 420.0), default_bounds=(0, 180.0), unit=u_deg
    ),
    "strike2": Bounds(
        physical_bounds=(-90.0, 420.0), default_bounds=(0, 180.0), unit=u_deg
    ),
    "dip": Bounds(
        physical_bounds=(-45.0, 135.0), default_bounds=(45.0, 90.0), unit=u_deg
    ),
    "dip1": Bounds(
        physical_bounds=(-45.0, 135.0), default_bounds=(45.0, 90.0), unit=u_deg
    ),
    "dip2": Bounds(
        physical_bounds=(-45.0, 135.0), default_bounds=(45.0, 90.0), unit=u_deg
    ),
    "rake": Bounds(
        physical_bounds=(-180.0, 270.0),
        default_bounds=(-90.0, 90.0),
        unit=u_deg,
    ),
    "rake1": Bounds(
        physical_bounds=(-180.0, 270.0),
        default_bounds=(-90.0, 90.0),
        unit=u_deg,
    ),
    "rake2": Bounds(
        physical_bounds=(-180.0, 270.0),
        default_bounds=(-90.0, 90.0),
        unit=u_deg,
    ),
    "mix": Bounds(physical_bounds=(0, 1), default_bounds=(0, 1), unit=u_hyp),
    "volume_change": Bounds(
        physical_bounds=(-1e12, 1e12), default_bounds=(1e8, 1e10), unit=u_v
    ),
    "diameter": Bounds(
        physical_bounds=(0.0, 100.0), default_bounds=(5.0, 10.0), unit=u_km
    ),
    "slip": Bounds(physical_bounds=(0.0, 150.0), default_bounds=(0.1, 8.0), unit=u_m),
    "opening_fraction": Bounds(
        physical_bounds=moffdiag, default_bounds=(0.0, 0.0), unit=u_hyp
    ),
    "azimuth": Bounds(physical_bounds=(0, 360), default_bounds=(0, 180), unit=u_deg),
    "amplitude": Bounds(
        physical_bounds=(1.0, 10e25), default_bounds=(1e10, 1e20), unit=u_nm
    ),
    "locking_depth": Bounds(
        physical_bounds=(0.1, 100.0), default_bounds=(1.0, 10.0), unit=u_km
    ),
    "nucleation_dip": Bounds(
        physical_bounds=(0.0, num.inf), default_bounds=(0.0, 7.0), unit=u_km
    ),
    "nucleation_strike": Bounds(
        physical_bounds=(0.0, num.inf), default_bounds=(0.0, 10.0), unit=u_km
    ),
    "nucleation_x": Bounds(
        physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_hyp
    ),
    "nucleation_y": Bounds(
        physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_hyp
    ),
    "time_shift": Bounds(
        physical_bounds=(-20.0, 20.0), default_bounds=(-5.0, 5.0), unit=u_s
    ),
    "coupling": Bounds(physical_bounds=(0, 100), default_bounds=(0, 1), unit=u_percent),
    "uperp": Bounds(
        physical_bounds=(-150.0, 150.0), default_bounds=(-0.3, 4.0), unit=u_m
    ),
    "uparr": Bounds(
        physical_bounds=(-1.0, 150.0), default_bounds=(-0.05, 6.0), unit=u_m
    ),
    "utens": Bounds(
        physical_bounds=(-150.0, 150.0), default_bounds=(0.0, 0.0), unit=u_m
    ),
    "durations": Bounds(
        physical_bounds=(0.0, 600.0), default_bounds=(0.5, 29.5), unit=u_s
    ),
    "velocities": Bounds(
        physical_bounds=(0.0, 20.0), default_bounds=(0.5, 4.2), unit=u_km_s
    ),
    "fn": Bounds(physical_bounds=(-1e20, 1e20), default_bounds=(-1e20, 1e20), unit=u_n),
    "fe": Bounds(physical_bounds=(-1e20, 1e20), default_bounds=(-1e20, 1e20), unit=u_n),
    "fd": Bounds(physical_bounds=(-1e20, 1e20), default_bounds=(-1e20, 1e20), unit=u_n),
    "mnn": Bounds(
        physical_bounds=(-SQRT2, SQRT2), default_bounds=(-SQRT2, SQRT2), unit=u_nm
    ),
    "mee": Bounds(
        physical_bounds=(-SQRT2, SQRT2), default_bounds=(-SQRT2, SQRT2), unit=u_nm
    ),
    "mdd": Bounds(
        physical_bounds=(-SQRT2, SQRT2), default_bounds=(-SQRT2, SQRT2), unit=u_nm
    ),
    "mne": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_nm),
    "mnd": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_nm),
    "med": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_nm),
    "magnitude": Bounds(
        physical_bounds=(-5.0, 10.0), default_bounds=(4.0, 7.0), unit=u_hyp
    ),
    "exx": Bounds(
        physical_bounds=(-num.inf, num.inf),
        default_bounds=(-200.0, 200.0),
        unit=u_nanostrain,
    ),
    "eyy": Bounds(
        physical_bounds=(-num.inf, num.inf),
        default_bounds=(-200.0, 200.0),
        unit=u_nanostrain,
    ),
    "exy": Bounds(
        physical_bounds=(-num.inf, num.inf),
        default_bounds=(-200.0, 200.0),
        unit=u_nanostrain,
    ),
    "rotation": Bounds(
        physical_bounds=(-num.inf, num.inf),
        default_bounds=(-200.0, 200.0),
        unit=u_nanostrain,
    ),
    "lat": Bounds(
        physical_bounds=(-90.0, 90.0), default_bounds=(30.0, 30.5), unit=u_deg
    ),
    "lon": Bounds(
        physical_bounds=(-180.0, 180.0), default_bounds=(30.0, 30.5), unit=u_deg
    ),
    "omega": Bounds(
        physical_bounds=(-10.0, 10.0), default_bounds=(0.5, 0.6), unit=u_deg_myr
    ),
    "w": Bounds(
        physical_bounds=(-3.0 / 8.0 * num.pi, 3.0 / 8.0 * num.pi),
        default_bounds=(-3.0 / 8.0 * num.pi, 3.0 / 8.0 * num.pi),
        unit=u_rad,
    ),
    "v": Bounds(
        physical_bounds=(-1.0 / 3, 1.0 / 3),
        default_bounds=(-1.0 / 3, 1.0 / 3),
        unit=u_rad,
    ),
    "kappa": Bounds(
        physical_bounds=(0.0, 2 * num.pi),
        default_bounds=(0.0, 2 * num.pi),
        unit=u_deg,
    ),
    "sigma": Bounds(
        physical_bounds=(-num.pi / 2.0, num.pi / 2.0),
        default_bounds=(-num.pi / 2.0, num.pi / 2.0),
        unit=u_deg,
    ),
    "h": Bounds(physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=u_deg),
    "length": Bounds(
        physical_bounds=(0.0, 7000.0), default_bounds=(5.0, 30.0), unit=u_km
    ),
    "width": Bounds(
        physical_bounds=(0.0, 500.0), default_bounds=(5.0, 20.0), unit=u_km
    ),
    "time": Bounds(
        physical_bounds=(-200.0, 200.0), default_bounds=(-5.0, 5.0), unit=u_s
    ),
    "delta_time": Bounds(
        physical_bounds=(0.0, 100.0), default_bounds=(0.0, 10.0), unit=u_s
    ),
    "depth_bottom": Bounds(
        physical_bounds=(0.0, 300.0), default_bounds=(0.0, 10.0), unit=u_km
    ),
    "distance": Bounds(
        physical_bounds=(0.0, 300.0), default_bounds=(0.0, 10.0), unit=u_km
    ),
    "duration": Bounds(
        physical_bounds=(0.0, 600.0), default_bounds=(1.0, 30.0), unit=u_s
    ),
    "peak_ratio": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=u_hyp
    ),
    "hypers": Bounds(
        physical_bounds=(-10.0, 10.0), default_bounds=(-2.0, 6.0), unit=u_hyp
    ),
    "ramp": Bounds(
        physical_bounds=(-0.005, 0.005), default_bounds=(-0.005, 0.005), unit=u_rad
    ),
    "offset": Bounds(
        physical_bounds=(-0.05, 0.05), default_bounds=(-0.05, 0.05), unit=u_m
    ),
    "traction": Bounds(physical_bounds=(0, 1000), default_bounds=(0, 50), unit=u_pa),
    "strike_traction": Bounds(
        physical_bounds=(-15000, 15000), default_bounds=(-50, 50), unit=u_pa
    ),
    "dip_traction": Bounds(
        physical_bounds=(-15000, 15000), default_bounds=(-50, 50), unit=u_pa
    ),
    "normal_traction": Bounds(
        physical_bounds=(-15000, 15000), default_bounds=(-50, 50), unit=u_pa
    ),
    "a_half_axis": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit=u_km
    ),
    "b_half_axis": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit=u_km
    ),
    "a_half_axis_bottom": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit=u_km
    ),
    "b_half_axis_bottom": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit=u_km
    ),
    "plunge": Bounds(physical_bounds=(0, 90), default_bounds=(0, 20), unit=u_deg),
    "delta_east_shift_bottom": Bounds(
        physical_bounds=(-500, 500), default_bounds=(-10, 10), unit=u_km
    ),
    "delta_north_shift_bottom": Bounds(
        physical_bounds=(-500, 500), default_bounds=(-10, 10), unit=u_km
    ),
    "curv_amplitude_bottom": Bounds(
        physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_hyp
    ),
    "curv_location_bottom": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=u_hyp
    ),
    "bend_location": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=u_hyp
    ),
    "bend_amplitude": Bounds(
        physical_bounds=moffdiag, default_bounds=moffdiag, unit=u_hyp
    ),
    "like": Bounds(
        physical_bounds=(-num.inf, num.inf), default_bounds=(0, 1), unit=u_hyp
    ),
}


def hypername(varname):
    return varname if varname in parameter_info.keys() else "hypers"


def make_path_tmpl(name="defaults"):
    return os.path.join(beat_dir_tmpl, f"{name}.pf")


def init_parameter_defaults():
    defaults = ParameterDefaults()
    for parameter_name, bounds in parameter_info.items():
        defaults.parameters[parameter_name] = bounds
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


defaults = get_defaults(force=False)
