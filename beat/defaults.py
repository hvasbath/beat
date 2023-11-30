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


# Bounds and Units for all parameters
parameter_info = {
    "east_shift": Bounds(
        physical_bounds=(-500.0, 500.0), default_bounds=(-10.0, 10.0), unit="$[km]$"
    ),
    "north_shift": Bounds(
        physical_bounds=(-500.0, 500.0), default_bounds=(-10.0, 10.0), unit="$[km]$"
    ),
    "depth": Bounds(
        physical_bounds=(0.0, 1000.0), default_bounds=(0.0, 5.0), unit="$[km]$"
    ),
    "strike": Bounds(
        physical_bounds=(-90.0, 420.0), default_bounds=(0, 180.0), unit="$[^\circ]$"
    ),
    "strike1": Bounds(
        physical_bounds=(-90.0, 420.0), default_bounds=(0, 180.0), unit="$[^\circ]$"
    ),
    "strike2": Bounds(
        physical_bounds=(-90.0, 420.0), default_bounds=(0, 180.0), unit="$[^\circ]$"
    ),
    "dip": Bounds(
        physical_bounds=(-45.0, 135.0), default_bounds=(45.0, 90.0), unit="$[^\circ]$"
    ),
    "dip1": Bounds(
        physical_bounds=(-45.0, 135.0), default_bounds=(45.0, 90.0), unit="$[^\circ]$"
    ),
    "dip2": Bounds(
        physical_bounds=(-45.0, 135.0), default_bounds=(45.0, 90.0), unit="$[^\circ]$"
    ),
    "rake": Bounds(
        physical_bounds=(-180.0, 270.0), default_bounds=(-90.0, 90.0), unit="$[^\circ]$"
    ),
    "rake1": Bounds(
        physical_bounds=(-180.0, 270.0), default_bounds=(-90.0, 90.0), unit="$[^\circ]$"
    ),
    "rake2": Bounds(
        physical_bounds=(-180.0, 270.0), default_bounds=(-90.0, 90.0), unit="$[^\circ]$"
    ),
    "mix": Bounds(physical_bounds=(0, 1), default_bounds=(0, 1), unit=""),
    "volume_change": Bounds(
        physical_bounds=(-1e12, 1e12), default_bounds=(1e8, 1e10), unit="$[m^3]$"
    ),
    "diameter": Bounds(
        physical_bounds=(0.0, 100.0), default_bounds=(5.0, 10.0), unit="$[km]$"
    ),
    "slip": Bounds(
        physical_bounds=(0.0, 150.0), default_bounds=(0.1, 8.0), unit="$[m]$"
    ),
    "opening_fraction": Bounds(
        physical_bounds=moffdiag, default_bounds=(0.0, 0.0), unit=""
    ),
    "azimuth": Bounds(
        physical_bounds=(0, 360), default_bounds=(0, 180), unit="$[^\circ]$"
    ),
    "amplitude": Bounds(
        physical_bounds=(1.0, 10e25), default_bounds=(1e10, 1e20), unit="$[Nm]$"
    ),
    "locking_depth": Bounds(
        physical_bounds=(0.1, 100.0), default_bounds=(1.0, 10.0), unit="$[km]$"
    ),
    "nucleation_dip": Bounds(
        physical_bounds=(0.0, num.inf), default_bounds=(0.0, 7.0), unit="$[km]$"
    ),
    "nucleation_strike": Bounds(
        physical_bounds=(0.0, num.inf), default_bounds=(0.0, 10.0), unit="$[km]$"
    ),
    "nucleation_x": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit=""),
    "nucleation_y": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit=""),
    "time_shift": Bounds(
        physical_bounds=(-20.0, 20.0), default_bounds=(-5.0, 5.0), unit="$[s]$"
    ),
    "coupling": Bounds(physical_bounds=(0, 100), default_bounds=(0, 1), unit="[$\%$]"),
    "uperp": Bounds(
        physical_bounds=(-150.0, 150.0), default_bounds=(-0.3, 4.0), unit="$[m]$"
    ),
    "uparr": Bounds(
        physical_bounds=(-1.0, 150.0), default_bounds=(-0.05, 6.0), unit="$[m]$"
    ),
    "utens": Bounds(
        physical_bounds=(-150.0, 150.0), default_bounds=(0.0, 0.0), unit="$[m]$"
    ),
    "durations": Bounds(
        physical_bounds=(0.0, 600.0), default_bounds=(0.5, 29.5), unit="$[s]$"
    ),
    "velocities": Bounds(
        physical_bounds=(0.0, 20.0), default_bounds=(0.5, 4.2), unit="$[km/s]$"
    ),
    "fn": Bounds(
        physical_bounds=(-1e20, 1e20), default_bounds=(-1e20, 1e20), unit="$[N]$"
    ),
    "fe": Bounds(
        physical_bounds=(-1e20, 1e20), default_bounds=(-1e20, 1e20), unit="$[N]$"
    ),
    "fd": Bounds(
        physical_bounds=(-1e20, 1e20), default_bounds=(-1e20, 1e20), unit="$[N]$"
    ),
    "mnn": Bounds(
        physical_bounds=(-SQRT2, SQRT2), default_bounds=(-SQRT2, SQRT2), unit="$[Nm]$"
    ),
    "mee": Bounds(
        physical_bounds=(-SQRT2, SQRT2), default_bounds=(-SQRT2, SQRT2), unit="$[Nm]$"
    ),
    "mdd": Bounds(
        physical_bounds=(-SQRT2, SQRT2), default_bounds=(-SQRT2, SQRT2), unit="$[Nm]$"
    ),
    "mne": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit="$[Nm]$"),
    "mnd": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit="$[Nm]$"),
    "med": Bounds(physical_bounds=moffdiag, default_bounds=moffdiag, unit="$[Nm]$"),
    "magnitude": Bounds(
        physical_bounds=(-5.0, 10.0), default_bounds=(4.0, 7.0), unit=""
    ),
    "eps_xx": Bounds(
        physical_bounds=(-num.inf, num.inf), default_bounds=(0, 1), unit=""
    ),
    "eps_yy": Bounds(
        physical_bounds=(-num.inf, num.inf), default_bounds=(0, 1), unit=""
    ),
    "eps_xy": Bounds(
        physical_bounds=(-num.inf, num.inf), default_bounds=(0, 1), unit=""
    ),
    "rotation": Bounds(
        physical_bounds=(-num.inf, num.inf),
        default_bounds=(-200.0, 200.0),
        unit="$[rad]$",
    ),
    "pole_lat": Bounds(
        physical_bounds=(-90.0, 90.0), default_bounds=(0, 1), unit="$[^\circ]$"
    ),
    "pole_lon": Bounds(
        physical_bounds=(-180.0, 180.0), default_bounds=(0, 1), unit="$[^\circ]$"
    ),
    "omega": Bounds(
        physical_bounds=(-10.0, 10.0), default_bounds=(0.5, 0.6), unit="$[^\circ/myr]$"
    ),
    "w": Bounds(
        physical_bounds=(-3.0 / 8.0 * num.pi, 3.0 / 8.0 * num.pi),
        default_bounds=(-0.005, 0.005),
        unit="$[rad]$",
    ),
    "v": Bounds(
        physical_bounds=(-1.0 / 3, 1.0 / 3),
        default_bounds=(-1.0 / 3, 1.0 / 3),
        unit="$[rad]$",
    ),
    "kappa": Bounds(
        physical_bounds=(0.0, 2 * num.pi),
        default_bounds=(0.0, 2 * num.pi),
        unit="$[^\circ]$",
    ),
    "sigma": Bounds(
        physical_bounds=(-num.pi / 2.0, num.pi / 2.0),
        default_bounds=(-num.pi / 2.0, num.pi / 2.0),
        unit="$[^\circ]$",
    ),
    "h": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit="$[^\circ]$"
    ),
    "length": Bounds(
        physical_bounds=(0.0, 7000.0), default_bounds=(5.0, 30.0), unit="$[km]$"
    ),
    "width": Bounds(
        physical_bounds=(0.0, 500.0), default_bounds=(5.0, 20.0), unit="$[km]$"
    ),
    "time": Bounds(
        physical_bounds=(-200.0, 200.0), default_bounds=(-5.0, 5.0), unit="$[s]$"
    ),
    "delta_time": Bounds(
        physical_bounds=(0.0, 100.0), default_bounds=(0.0, 10.0), unit="$[s]$"
    ),
    "depth_bottom": Bounds(
        physical_bounds=(0.0, 300.0), default_bounds=(0.0, 10.0), unit="$[km]$"
    ),
    "distance": Bounds(
        physical_bounds=(0.0, 300.0), default_bounds=(0.0, 10.0), unit="$[km]$"
    ),
    "duration": Bounds(
        physical_bounds=(0.0, 600.0), default_bounds=(1.0, 30.0), unit="$[s]$"
    ),
    "peak_ratio": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=""
    ),
    "hypers": Bounds(physical_bounds=(-4.0, 10.0), default_bounds=(-2.0, 6.0), unit=""),
    "ramp": Bounds(
        physical_bounds=(-0.005, 0.005), default_bounds=(-0.005, 0.005), unit="$[rad]$"
    ),
    "offset": Bounds(
        physical_bounds=(-0.05, 0.05), default_bounds=(-0.05, 0.05), unit="$[m]$"
    ),
    "lat": Bounds(
        physical_bounds=(30.0, 30.5), default_bounds=(30.0, 30.5), unit="$[^\circ]$"
    ),
    "lon": Bounds(
        physical_bounds=(30.0, 30.5), default_bounds=(30.0, 30.5), unit="$[^\circ]$"
    ),
    "traction": Bounds(
        physical_bounds=(0, 10000), default_bounds=(0, 10000), unit="$[MPa]$"
    ),
    "strike_traction": Bounds(
        physical_bounds=(-15000, 15000), default_bounds=(-15000, 15000), unit="$[MPa]$"
    ),
    "dip_traction": Bounds(
        physical_bounds=(-15000, 15000), default_bounds=(-15000, 15000), unit="$[MPa]$"
    ),
    "tensile_traction": Bounds(
        physical_bounds=(-15000, 15000), default_bounds=(-15000, 15000), unit="$[MPa]$"
    ),
    "major_axis": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit="$[km]$"
    ),
    "minor_axis": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit="$[km]$"
    ),
    "major_axis_bottom": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit="$[km]$"
    ),
    "minor_axis_bottom": Bounds(
        physical_bounds=(0.01, 100), default_bounds=(0.01, 10), unit="$[km]$"
    ),
    "plunge": Bounds(
        physical_bounds=(0, 90), default_bounds=(0, 20), unit="$[^\circ]$"
    ),
    "delta_east_shift_bottom": Bounds(
        physical_bounds=(-500, 500), default_bounds=(-10, 10), unit="$[km]$"
    ),
    "delta_north_shift_bottom": Bounds(
        physical_bounds=(-500, 500), default_bounds=(-10, 10), unit="$[km]$"
    ),
    "curv_amplitude_bottom": Bounds(
        physical_bounds=moffdiag, default_bounds=moffdiag, unit=""
    ),
    "curv_location_bottom": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=""
    ),
    "bend_location": Bounds(
        physical_bounds=(0.0, 1.0), default_bounds=(0.0, 1.0), unit=""
    ),
    "bend_amplitude": Bounds(
        physical_bounds=moffdiag, default_bounds=moffdiag, unit=""
    ),
    "like": Bounds(physical_bounds=(-num.inf, num.inf), default_bounds=(0, 1), unit=""),
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


defaults = get_defaults()
defaults = get_defaults()
