"""
The config module contains the classes to build the configuration files that
are being read by the beat executable.

So far there are configuration files for the three main optimization problems
implemented. Solving the fault geometry, the static distributed slip and the
kinematic distributed slip.
"""
import logging
import os
from collections import OrderedDict

import numpy as num
from pyrocko import gf, model, trace, util
from pyrocko.cake import load_model
from pyrocko.gf import RectangularSource as PyrockoRS
from pyrocko.gf.seismosizer import Cloneable, LocalEngine, stf_classes
from pyrocko.guts import (
    ArgumentError,
    Bool,
    Dict,
    Float,
    Int,
    List,
    Object,
    String,
    StringChoice,
    Tuple,
    dump,
    load,
)
from theano import config as tconfig

from beat import utility
from beat.covariance import available_noise_structures, available_noise_structures_2d
from beat.heart import (
    ArrivalTaper,
    Filter,
    FilterBase,
    Parameter,
    ReferenceLocation,
    _domain_choices,
)
from beat.sources import MTQTSource, MTSourceWithMagnitude, RectangularSource
from beat.utility import check_point_keys, list2string


guts_prefix = "beat"

logger = logging.getLogger("config")

ffi_mode_str = "ffi"
geometry_mode_str = "geometry"


block_vars = ["bl_azimuth", "bl_amplitude"]
seis_vars = ["time", "duration"]

source_names = """
    ExplosionSource
    RectangularExplosionSource
    DCSource
    CLVDSource
    MTSource
    MTQTSource
    RectangularSource
    DoubleDCSource
    RingfaultSource
    """.split()

source_classes = [
    gf.ExplosionSource,
    gf.RectangularExplosionSource,
    gf.DCSource,
    gf.CLVDSource,
    MTSourceWithMagnitude,
    MTQTSource,
    PyrockoRS,
    gf.DoubleDCSource,
    gf.RingfaultSource,
]

stf_names = """
    Boxcar
    Triangular
    HalfSinusoid
    """.split()

source_catalog = {
    name: source_class for name, source_class in zip(source_names, source_classes)
}

stf_catalog = {name: stf_class for name, stf_class in zip(stf_names, stf_classes[1:4])}

interseismic_vars = [
    "east_shift",
    "north_shift",
    "strike",
    "dip",
    "length",
    "locking_depth",
] + block_vars

static_dist_vars = ["uparr", "uperp", "utens"]
derived_dist_vars = ["coupling"]

hypo_vars = ["nucleation_strike", "nucleation_dip", "time"]
partial_kinematic_vars = ["durations", "velocities"]
voronoi_locations = ["voronoi_strike", "voronoi_dip"]

mt_components = ["mnn", "mee", "mdd", "mne", "mnd", "med"]
dc_components = ["strike1", "dip1", "rake1", "strike2", "dip2", "rake2"]

kinematic_dist_vars = static_dist_vars + partial_kinematic_vars + hypo_vars
transd_vars_dist = partial_kinematic_vars + static_dist_vars + voronoi_locations
dist_vars = static_dist_vars + partial_kinematic_vars + derived_dist_vars

interseismic_catalog = {"geodetic": interseismic_vars}

geometry_catalog = {
    "polarity": source_catalog,
    "geodetic": source_catalog,
    "seismic": source_catalog,
}

ffi_catalog = {"geodetic": static_dist_vars, "seismic": kinematic_dist_vars}

modes_catalog = OrderedDict(
    [
        [geometry_mode_str, geometry_catalog],
        [ffi_mode_str, ffi_catalog],
        ["interseismic", interseismic_catalog],
    ]
)

derived_variables_mapping = {
    "MTQTSource": mt_components + dc_components,
    "MTSource": dc_components,
    "RectangularSource": ["magnitude"],
    "RectangularSourcePole": ["magnitude", "coupling"],
}


hyper_name_laplacian = "h_laplacian"

moffdiag = (-1.0, 1.0)
mdiag = (-num.sqrt(2), num.sqrt(2))

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
    exx=(-200.0, 200.0),
    eyy=(-200.0, 200.0),
    exy=(-200.0, 200.0),
    rotation=(-200.0, 200.0),
    w=(-3.0 / 8.0 * num.pi, 3.0 / 8.0 * num.pi),
    v=(-1.0 / 3, 1.0 / 3.0),
    kappa=(0.0, 2 * num.pi),
    sigma=(-num.pi / 2.0, num.pi / 2.0),
    h=(0.0, 1.0),
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
    bl_azimuth=(0, 180),
    bl_amplitude=(0.0, 0.1),
    locking_depth=(1.0, 10.0),
    hypers=(-5.0, 8.0),
    ramp=(-0.005, 0.005),
    offset=(-0.05, 0.05),
    lat=(30.0, 30.5),
    lon=(30.0, 30.5),
    omega=(0.5, 0.6),
)

default_seis_std = 1.0e-6
default_geo_std = 1.0e-3

default_decimation_factors = {"polarity": 1, "geodetic": 4, "seismic": 2}

response_file_name = "responses.pkl"
geodetic_data_name = "geodetic_data.pkl"
seismic_data_name = "seismic_data.pkl"
stations_name = "stations.txt"


def multi_event_seismic_data_name(nevent=0):
    if nevent == 0:
        return seismic_data_name
    else:
        return "seismic_data_subevent_{}.pkl".format(nevent)


def multi_event_stations_name(nevent=0):
    if nevent == 0:
        return stations_name
    else:
        return "stations_subevent_{}.txt".format(nevent)


linear_gf_dir_name = "linear_gfs"
discretization_dir_name = os.path.join(linear_gf_dir_name, "discretization")
results_dir_name = "results"
fault_geometry_name = "fault_geometry.pkl"
phase_markers_name = "phase_markers.txt"
geodetic_linear_gf_name = "linear_geodetic_gfs.pkl"

sample_p_outname = "sample.params"

summary_name = "summary.txt"

km = 1000.0


_quantity_choices = ["displacement", "velocity", "acceleration"]
_interpolation_choices = ["nearest_neighbor", "multilinear"]
_structure_choices = available_noise_structures()
_structure_choices_2d = available_noise_structures_2d()
_mode_choices = [geometry_mode_str, ffi_mode_str]
_regularization_choices = ["laplacian", "none"]
_correlation_function_choices = ["nearest_neighbor", "gaussian", "exponential"]
_discretization_choices = ["uniform", "resolution"]
_initialization_choices = ["random", "lsq"]
_backend_choices = ["csv", "bin"]
_datatype_choices = ["geodetic", "seismic", "polarity"]
_sampler_choices = ["PT", "SMC", "Metropolis"]


class InconsistentParameterNaming(Exception):
    context = (
        '"{}" Parameter name is inconsistent with its key "{}"!\n'
        + " Please edit the config_{}.yaml accordingly!"
    )

    def __init__(self, keyname="", parameter="", mode=""):
        self.parameter = parameter
        self.mode = mode
        self.keyname = keyname

    def __str__(self):
        return self.context.format(self.parameter, self.keyname, self.mode)


class ConfigNeedsUpdatingError(Exception):
    context = (
        "Configuration file has to be updated! \n"
        + ' Please run "beat update <project_dir --parameters=structure>'
    )

    def __init__(self, errmess=""):
        self.errmess = errmess

    def __str__(self):
        return "\n%s\n%s" % (self.errmess, self.context)


class GFConfig(Object):
    """
    Base config for GreensFunction calculation parameters.
    """

    store_superdir = String.T(
        default="./",
        help="Absolute path to the directory where Greens Function"
        " stores are located",
    )
    reference_model_idx = Int.T(
        default=0,
        help="Index to velocity model to use for the optimization."
        " 0 - reference, 1..n - model of variations",
    )
    n_variations = Tuple.T(
        2,
        Int.T(),
        default=(0, 1),
        help="Start and end index to vary input velocity model. "
        "Important for the calculation of the model prediction covariance"
        " matrix with respect to uncertainties in the velocity model.",
    )
    earth_model_name = String.T(
        default="ak135-f-continental.f",
        help="Name of the reference earthmodel, see "
        "pyrocko.cake.builtin_models() for alternatives.",
    )
    nworkers = Int.T(
        default=1, help="Number of processors to use for calculating the GFs"
    )


class NonlinearGFConfig(GFConfig):
    """
    Config for non-linear GreensFunction calculation parameters.
    Defines how the grid of Green's Functions in the respective store is
    created.
    """

    use_crust2 = Bool.T(
        default=True,
        help="Flag, for replacing the crust from the earthmodel"
        "with crust from the crust2 model.",
    )
    replace_water = Bool.T(
        default=True, help="Flag, for replacing water layers in the crust2 model."
    )
    custom_velocity_model = gf.Earthmodel1D.T(
        default=None,
        optional=True,
        help="Custom Earthmodel, in case crust2 and standard model not"
        " wanted. Needs to be a :py::class:cake.LayeredModel",
    )
    source_depth_min = Float.T(
        default=0.0, help="Minimum depth [km] for GF function grid."
    )
    source_depth_max = Float.T(
        default=10.0, help="Maximum depth [km] for GF function grid."
    )
    source_depth_spacing = Float.T(
        default=1.0, help="Depth spacing [km] for GF function grid."
    )
    source_distance_radius = Float.T(
        default=20.0,
        help="Radius of distance grid [km] for GF function grid around "
        "reference event.",
    )
    source_distance_spacing = Float.T(
        default=1.0,
        help="Distance spacing [km] for GF function grid w.r.t" " reference_location.",
    )
    error_depth = Float.T(
        default=0.1,
        help="3sigma [%/100] error in velocity model layer depth, "
        "translates to interval for varying the velocity model",
    )
    error_velocities = Float.T(
        default=0.1,
        help="3sigma [%/100] in velocity model layer wave-velocities, "
        "translates to interval for varying the velocity model",
    )
    depth_limit_variation = Float.T(
        default=600.0,
        help="Depth limit [km] for varying the velocity model. Below that "
        "depth the velocity model is not varied based on the errors "
        "defined above!",
    )
    version = String.T(
        default="",
        help="Version number of the backend codes. If not defined, default versions will be used.",
    )


class SeismicGFConfig(NonlinearGFConfig):
    """
    Seismic GF parameters for Layered Halfspace.
    """

    reference_location = ReferenceLocation.T(
        default=None,
        help="Reference location for the midpoint of the Green's Function " "grid.",
        optional=True,
    )
    code = String.T(
        default="qssp",
        help="Modeling code to use. (qssp, qseis, coming soon: " "qseis2d)",
    )
    sample_rate = Float.T(default=2.0, help="Sample rate for the Greens Functions.")
    rm_gfs = Bool.T(
        default=True,
        help="Flag for removing modeling module GF files after" " completion.",
    )


class GeodeticGFConfig(NonlinearGFConfig):
    """
    Geodetic GF parameters for Layered Halfspace.
    """

    code = String.T(
        default="psgrn",
        help="Modeling code to use. (psgrn, ... others need to be" "implemented!)",
    )
    sample_rate = Float.T(
        default=1.0 / (3600.0 * 24.0),
        help="Sample rate for the Greens Functions. Mainly relevant for"
        " viscoelastic modeling. Default: coseismic-one day",
    )
    sampling_interval = Float.T(
        default=1.0,
        help="Distance dependent sampling spacing coefficient." "1. - equidistant",
    )
    medium_depth_spacing = Float.T(
        default=1.0, help="Depth spacing [km] for GF medium grid."
    )
    medium_distance_spacing = Float.T(
        default=10.0, help="Distance spacing [km] for GF medium grid."
    )


class DiscretizationConfig(Object):
    """
    Config to determine the discretization of the finite fault(s)
    """

    extension_widths = List.T(
        Float.T(),
        default=[0.1],
        help="Extend reference sources by this factor in each"
        " dip-direction. 0.1 means extension of the fault by 10% in each"
        " direction, i.e. 20% in total. If patches would intersect with"
        " the free surface they are constrained to end at the surface."
        " Each value is applied following the list-order to the"
        " respective reference source.",
    )
    extension_lengths = List.T(
        Float.T(),
        default=[0.1],
        help="Extend reference sources by this factor in each"
        " strike-direction. 0.1 means extension of the fault by 10% in"
        " each direction, i.e. 20% in total. Each value is applied "
        " following the list-order to the respective reference source.",
    )


class UniformDiscretizationConfig(DiscretizationConfig):

    patch_widths = List.T(
        Float.T(),
        default=[5.0],
        help="List of Patch width [km] to divide reference sources. Each value"
        " is applied following the list-order to the respective reference"
        " source",
    )
    patch_lengths = List.T(
        Float.T(),
        default=[5.0],
        help="Patch length [km] to divide reference sources Each value"
        " is applied following the list-order to the respective reference"
        " source",
    )

    def get_patch_dimensions(self):
        return self.patch_widths, self.patch_lengths


class ResolutionDiscretizationConfig(DiscretizationConfig):
    """
    Parameters that control the resolution based source discretization.

    References
    ----------
    .. [Atzori2011] Atzori & Antonioli (2011).
        Optimal fault resolution in geodetic inversion of coseismic data.
        Geophysical Journal International, 185(1):529-538
    """

    epsilon = Float.T(
        default=4.0e-3,
        help="Damping constant for Laplacian of Greens Functions. "
        "Usually reasonable between: [0.1 to 0.005]",
    )
    epsilon_search_runs = Int.T(
        default=1,
        help="If above 1, the algorithm is iteratively run, starting with "
        "epsilon as lower bound on equal logspace up to epsilon * 100. "
        '"epsilon_search_runs" determines times of repetition and the '
        "spacing between epsilons. If this is 1, only the model for "
        '"epsilon" is created! The epsilon that minimises Resolution '
        "spreading (Atzori et al. 2019) is chosen!",
    )
    resolution_thresh = Float.T(
        default=0.999,
        help="Resolution threshold discretization continues until all patches "
        "are below this threshold. The lower the finer the "
        "discretization. Reasonable between: [0.95, 0.999]",
    )
    depth_penalty = Float.T(
        default=3.5,
        help="The higher the number the more penalty on the deeper "
        "patches-ergo larger patches.",
    )
    alpha = Float.T(
        default=0.3,
        help="Decimal percentage of largest patches that are subdivided "
        "further. Reasonable: [0.1, 0.3]",
    )
    patch_widths_min = List.T(
        Float.T(),
        default=[1.0],
        help="Patch width [km] for min final discretization of patches.",
    )
    patch_widths_max = List.T(
        Float.T(),
        default=[5.0],
        help="Patch width [km] for max initial discretization of patches.",
    )
    patch_lengths_min = List.T(
        Float.T(),
        default=[1.0],
        help="Patch length [km] for min final discretization of" " patches.",
    )
    patch_lengths_max = List.T(
        Float.T(),
        default=[5.0],
        help="Patch length [km] for max initial discretization of" " patches.",
    )

    def get_patch_dimensions(self):
        """
        Returns
        -------
        List of patch_widths, List of patch_lengths
        """
        return self.patch_widths_max, self.patch_lengths_max


discretization_catalog = {
    "uniform": UniformDiscretizationConfig,
    "resolution": ResolutionDiscretizationConfig,
}


class LinearGFConfig(GFConfig):
    """
    Config for linear GreensFunction calculation parameters.
    """

    reference_sources = List.T(
        RectangularSource.T(), help="Geometry of the reference source(s) to fix"
    )
    sample_rate = Float.T(default=2.0, help="Sample rate for the Greens Functions.")
    discretization = StringChoice.T(
        default="uniform",
        choices=_discretization_choices,
        help="Flag for discretization of finite sources into patches."
        " Choices: %s" % utility.list2string(_discretization_choices),
    )
    discretization_config = DiscretizationConfig.T(
        default=UniformDiscretizationConfig.D(),
        help="Discretization configuration that allows customization.",
    )

    def __init__(self, **kwargs):

        kwargs = _init_kwargs(
            method_config_name="discretization_config",
            method_name="discretization",
            method_catalog=discretization_catalog,
            kwargs=kwargs,
        )

        Object.__init__(self, **kwargs)


class SeismicLinearGFConfig(LinearGFConfig):
    """
    Config for seismic linear GreensFunction calculation parameters.
    """

    reference_location = ReferenceLocation.T(
        default=None,
        help="Reference location for the midpoint of the Green's Function " "grid.",
        optional=True,
    )
    duration_sampling = Float.T(
        default=1.0,
        help="Calculate Green's Functions for varying Source Time Function"
        " durations determined by prior bounds. Discretization between"
        " is determined by duration sampling.",
    )
    starttime_sampling = Float.T(
        default=1.0,
        help="Calculate Green's Functions for varying rupture onset times."
        "These are determined by the (rupture) velocity prior bounds "
        "and the hypocenter location.",
    )

    def __init__(self, **kwargs):

        Object.__init__(self, **kwargs)

        if self.discretization == "resolution":
            raise ValueError(
                "Resolution based discretization only available " "for geodetic data!"
            )


class GeodeticLinearGFConfig(LinearGFConfig):
    pass


class WaveformFitConfig(Object):
    """
    Config for specific parameters that are applied to post-process
    a specific type of waveform and calculate the misfit.
    """

    include = Bool.T(default=True, help="Flag to include waveform into optimization.")
    preprocess_data = Bool.T(default=True, help="Flag to filter input data.")
    name = String.T("any_P")
    arrivals_marker_path = String.T(
        default=os.path.join("./", phase_markers_name),
        help='Path to table of "PhaseMarker" containing arrival times of '
        "waveforms at station(s) dumped by "
        "pyrocko.gui.marker.save_markers.",
    )
    blacklist = List.T(
        String.T(),
        default=[],
        help="Network.Station codes for stations to be thrown out.",
    )
    quantity = StringChoice.T(
        choices=_quantity_choices,
        default="displacement",
        help="Quantity of synthetics to be computed.",
    )
    channels = List.T(String.T(), default=["Z"])
    filterer = List.T(
        FilterBase.T(default=Filter.D()),
        help="List of Filters that are applied in the order of the list.",
    )
    distances = Tuple.T(2, Float.T(), default=(30.0, 90.0))
    interpolation = StringChoice.T(
        choices=_interpolation_choices,
        default="multilinear",
        help="GF interpolation scheme. Choices: %s"
        % utility.list2string(_interpolation_choices),
    )
    arrival_taper = trace.Taper.T(
        default=ArrivalTaper.D(),
        help="Taper a,b/c,d time [s] before/after wave arrival",
    )
    event_idx = Int.T(
        default=0,
        optional=True,
        help="Index to event from events list for reference time and data "
        "extraction. Default is 0 - always use the reference event.",
    )
    domain = StringChoice.T(
        choices=_domain_choices, default="time", help="type of trace"
    )


class SeismicNoiseAnalyserConfig(Object):

    structure = StringChoice.T(
        choices=_structure_choices,
        default="variance",
        help="Determines data-covariance matrix structure."
        " Choices: %s" % utility.list2string(_structure_choices),
    )
    pre_arrival_time = Float.T(
        default=5.0,
        help="Time [s] before synthetic P-wave arrival until " "variance is estimated",
    )


class GeodeticNoiseAnalyserConfig(Object):

    structure = StringChoice.T(
        choices=_structure_choices_2d,
        default="import",
        help="Determines data-covariance matrix structure."
        " Choices: %s" % utility.list2string(_structure_choices_2d),
    )
    max_dist_perc = Float.T(
        default=0.2,
        help="Distance in decimal percent from maximum distance in scene to use for "
        "non-Toeplitz noise estimation",
    )


class SeismicConfig(Object):
    """
    Config for seismic data optimization related parameters.
    """

    datadir = String.T(default="./")
    noise_estimator = SeismicNoiseAnalyserConfig.T(
        default=SeismicNoiseAnalyserConfig.D(),
        help="Determines the structure of the data-covariance matrix.",
    )
    responses_path = String.T(default=None, optional=True, help="Path to response file")
    pre_stack_cut = Bool.T(
        default=True,
        help="Cut the GF traces before stacking around the specified arrival" " taper",
    )
    station_corrections = Bool.T(
        default=False, help="If set, optimize for time shift for each station."
    )
    waveforms = List.T(WaveformFitConfig.T(default=WaveformFitConfig.D()))
    dataset_specific_residual_noise_estimation = Bool.T(
        default=False,
        help="If set, for EACH DATASET specific hyperparameter estimation."
        "n_hypers = nstations * nchannels."
        "If false one hyperparameter for each DATATYPE and "
        "displacement COMPONENT.",
    )
    gf_config = GFConfig.T(default=SeismicGFConfig.D())

    def __init__(self, **kwargs):

        waveforms = "waveforms"
        wavenames = kwargs.pop("wavenames", ["any_P"])
        wavemaps = []
        if waveforms not in kwargs:
            for wavename in wavenames:
                wavemaps.append(WaveformFitConfig(name=wavename))

            kwargs[waveforms] = wavemaps

        mode = kwargs.pop("mode", geometry_mode_str)

        if mode == geometry_mode_str:
            gf_config = SeismicGFConfig()
        elif mode == ffi_mode_str:
            gf_config = SeismicLinearGFConfig()

        if "gf_config" not in kwargs:
            kwargs["gf_config"] = gf_config

        Object.__init__(self, **kwargs)

    def get_waveform_names(self):
        return [wc.name for wc in self.waveforms]

    def get_unique_channels(self):
        cl = [wc.channels for wc in self.waveforms]
        uc = []
        for c in cl:
            uc.extend(c)
        return list(set(uc))

    def get_hypernames(self):
        hids = []
        for i, wc in enumerate(self.waveforms):
            if wc.include:
                for c in wc.channels:
                    hypername = "_".join(("h", wc.name, str(i), c))
                    hids.append(hypername)

        return hids

    def get_station_blacklist(self):
        blacklist = []
        for wc in self.waveforms:
            blacklist.extend(wc.blacklist)

        return list(set(blacklist))

    def get_hierarchical_names(self):
        if self.station_corrections:
            return ["time_shift"]
        else:
            return []

    def init_waveforms(self, wavenames=["any_P"]):
        """
        Initialise waveform configurations.
        """
        for wavename in wavenames:
            self.waveforms.append(WaveformFitConfig(name=wavename))


class PolarityGFConfig(NonlinearGFConfig):

    code = String.T(
        default="cake", help="Raytracing code to use for takeoff-angle computations."
    )
    always_raytrace = Bool.T(
        default=True, help="Set to true for ignoring the interpolation table."
    )
    reference_location = ReferenceLocation.T(
        default=None,
        help="Reference location for the midpoint of the " "Green's Function 'grid.",
        optional=True,
    )
    sample_rate = Float.T(
        default=1.0,
        optional=True,
        help="Sample rate for the polarity Greens Functions.",
    )


class PolarityFitConfig(Object):

    name = String.T(default="any_P", help="Seismic phase name for picked polarities")
    include = Bool.T(
        default=True, help="Whether to include this FitConfig to the estimation."
    )
    polarities_marker_path = String.T(
        default=os.path.join("./", phase_markers_name),
        help='Path to table of "PhaseMarker" containing polarity of waveforms '
        "at station(s) dumped by pyrocko.gui.marker.save_markers.",
    )
    blacklist = List.T(
        String.T(),
        default=[""],
        help="List of Network.Station name(s) for stations to be thrown out.",
    )
    event_idx = Int.T(
        default=0,
        optional=True,
        help="Index to event from events list for reference time and data "
        "extraction. Default is 0 - always use the reference event.",
    )


class PolarityConfig(Object):

    datadir = String.T(default="./")
    waveforms = List.T(
        PolarityFitConfig.T(default=PolarityFitConfig.D()),
        help="Polarity mapping for potentially fitting several phases.",
    )
    gf_config = GFConfig.T(default=PolarityGFConfig.D())

    def __init__(self, **kwargs):

        waveforms = "waveforms"
        wavenames = kwargs.pop("wavenames", ["any_P"])
        wavemaps = []
        if waveforms not in kwargs:
            for wavename in wavenames:
                wavemaps.append(PolarityFitConfig(name=wavename))

            kwargs[waveforms] = wavemaps

        mode = kwargs.pop("mode", geometry_mode_str)

        if mode == geometry_mode_str:
            gf_config = PolarityGFConfig()
        else:
            raise NotImplementedError(
                'Polarity composite is only implemented for "geometry" mode!'
            )

        if "gf_config" not in kwargs:
            kwargs["gf_config"] = gf_config

        Object.__init__(self, **kwargs)

    def get_waveform_names(self):
        return [pc.name for pc in self.waveforms]

    def get_unique_channels(self):
        cl = [pc.channels for pc in self.waveforms]
        uc = []
        for c in cl:
            uc.extend(c)
        return list(set(uc))

    def get_hypernames(self):
        hids = []
        for i, pmap_config in enumerate(self.waveforms):
            if pmap_config.include:
                hypername = "_".join(("h", pmap_config.name, "pol", str(i)))
                hids.append(hypername)

        return hids

    def init_waveforms(self, wavenames=["any_P"]):
        """
        Initialise waveform configurations.
        """
        for wavename in wavenames:
            self.waveforms.append(PolarityFitConfig(name=wavename))


class CorrectionConfig(Object):

    dataset_names = List.T(
        String.T(), default=[], help="Datasets to include in the correction."
    )
    enabled = Bool.T(default=False, help="Flag to enable Correction.")

    def get_suffixes(self):
        return self._suffixes

    @property
    def _suffixes(self):
        return [""]

    def get_hierarchical_names(self):
        raise NotImplementedError("Needs to be implemented in the subclass!")

    def check_consistency(self):
        if len(self.dataset_names) == 0 and self.enabled:
            raise AttributeError(
                "%s correction is enabled, but "
                "dataset_names are empty! Either the correction needs to be "
                'disabled or the field "dataset_names" needs to be '
                "filled!" % self.feature
            )


class GNSSCorrectionConfig(CorrectionConfig):

    station_blacklist = List.T(
        String.T(), default=[], help="GNSS station names to apply no correction."
    )
    station_whitelist = List.T(
        String.T(), default=[], help="GNSS station names to apply the correction."
    )

    def get_hierarchical_names(self, name=None, number=0):
        return ["{}_{}".format(number, suffix) for suffix in self.get_suffixes()]


class EulerPoleConfig(GNSSCorrectionConfig):
    @property
    def _suffixes(self):
        return ["pole_lat", "pole_lon", "omega"]

    @property
    def feature(self):
        return "Euler Pole"

    def init_correction(self):
        from beat.models.corrections import EulerPoleCorrection

        self.check_consistency()
        return EulerPoleCorrection(self)


class StrainRateConfig(GNSSCorrectionConfig):
    @property
    def _suffixes(self):
        return ["exx", "eyy", "exy", "rotation"]

    @property
    def feature(self):
        return "Strain Rate"

    def init_correction(self):
        from beat.models.corrections import StrainRateCorrection

        self.check_consistency()
        return StrainRateCorrection(self)


class RampConfig(CorrectionConfig):
    @property
    def _suffixes(self):
        return ["azimuth_ramp", "range_ramp", "offset"]

    @property
    def feature(self):
        return "Ramps"

    def get_hierarchical_names(self, name, number=0):
        return [
            "{}_{}".format(name, suffix)
            for suffix in self.get_suffixes()
            if name in self.dataset_names
        ]

    def init_correction(self):
        from beat.models.corrections import RampCorrection

        self.check_consistency()
        return RampCorrection(self)


class GeodeticCorrectionsConfig(Object):
    """
    Config for corrections to geodetic datasets.
    """

    euler_poles = List.T(EulerPoleConfig.T(), default=[EulerPoleConfig.D()])
    ramp = RampConfig.T(default=RampConfig.D())
    strain_rates = List.T(StrainRateConfig.T(), default=[StrainRateConfig.D()])

    def iter_corrections(self):
        out_corr = [self.ramp]

        for euler_pole_conf in self.euler_poles:
            out_corr.append(euler_pole_conf)

        for strain_conf in self.strain_rates:
            out_corr.append(strain_conf)

        return out_corr

    @property
    def has_enabled_corrections(self):
        return any([corr.enabled for corr in self.iter_corrections()])


class DatasetConfig(Object):
    """
    Base config for datasets.
    """

    datadir = String.T(default="./", help="Path to directory of the data")
    names = List.T(String.T(), default=["Data prefix filenames here ..."])

    def load_data(self):
        raise NotImplementedError("Needs implementation in the subclass!")


class SARDatasetConfig(DatasetConfig):
    def load_data(self):
        from beat.inputf import load_kite_scenes

        return load_kite_scenes(self.datadir, self.names)


class GNSSDatasetConfig(DatasetConfig):

    components = List.T(String.T(), default=["north", "east", "up"])
    blacklist = List.T(
        String.T(),
        default=["put blacklisted station names here or delete"],
        help="GNSS station to be thrown out.",
    )

    def load_data(self, campaign=False):
        from beat.inputf import load_and_blacklist_gnss

        all_targets = []
        for filename in self.names:
            logger.info("Loading file %s ..." % filename)
            try:
                targets = load_and_blacklist_gnss(
                    self.datadir,
                    filename,
                    self.blacklist,
                    campaign=campaign,
                    components=self.components,
                )
                if targets:
                    logger.info("Successfully loaded GNSS data from file %s" % filename)
                    if campaign:
                        all_targets.append(targets)
                    else:
                        all_targets.extend(targets)
            except OSError:
                logger.warning(
                    "GNSS of file %s not conform with ascii format!" % filename
                )

            return all_targets


class GeodeticConfig(Object):
    """
    Config for geodetic data optimization related parameters.
    """

    types = Dict.T(
        String.T(),
        DatasetConfig.T(),
        default={"SAR": SARDatasetConfig.D(), "GNSS": GNSSDatasetConfig.D()},
        help="Types of geodetic data, i.e. SAR, GNSS, with their configs",
    )
    noise_estimator = GeodeticNoiseAnalyserConfig.T(
        default=GeodeticNoiseAnalyserConfig.D(),
        help="Determines the structure of the data-covariance matrix.",
    )
    interpolation = StringChoice.T(
        choices=_interpolation_choices,
        default="multilinear",
        help="GF interpolation scheme during synthetics generation."
        " Choices: %s" % utility.list2string(_interpolation_choices),
    )
    corrections_config = GeodeticCorrectionsConfig.T(
        default=GeodeticCorrectionsConfig.D(),
        help="Config for additional corrections to apply to geodetic datasets.",
    )
    dataset_specific_residual_noise_estimation = Bool.T(
        default=False,
        help="If set, for EACH DATASET specific hyperparameter estimation."
        "For geodetic data: n_hypers = nimages (SAR) or "
        "nstations * ncomponents (GNSS)."
        "If false one hyperparameter for each DATATYPE and "
        "displacement COMPONENT.",
    )
    gf_config = GFConfig.T(default=GeodeticGFConfig.D())

    def __init__(self, **kwargs):

        mode = kwargs.pop("mode", geometry_mode_str)

        if mode == geometry_mode_str:
            gf_config = GeodeticGFConfig()
        elif mode == ffi_mode_str:
            gf_config = GeodeticLinearGFConfig()

        if "gf_config" not in kwargs:
            kwargs["gf_config"] = gf_config

        Object.__init__(self, **kwargs)

    def get_hypernames(self):
        return ["_".join(("h", typ)) for typ in self.types]

    def get_hierarchical_names(self, datasets=None):

        out_names = []
        for number, corr_conf in enumerate(self.corrections_config.iter_corrections()):
            if corr_conf.enabled:
                for dataset in datasets:
                    if dataset.name in corr_conf.dataset_names:
                        hiernames = corr_conf.get_hierarchical_names(
                            name=dataset.name, number=number
                        )

                        out_names.extend(hiernames)
                    else:
                        logger.info(
                            "Did not find dataset name %s in the corrections list.",
                            dataset.name,
                        )

        return list(set(out_names))


class ModeConfig(Object):
    """
    BaseConfig for optimization mode specific configuration.
    """

    pass


class RegularizationConfig(Object):
    pass


class NoneRegularizationConfig(Object):
    """
    Dummy class to return None.
    """

    def __new__(self):
        return None


class LaplacianRegularizationConfig(RegularizationConfig):
    """
    Determines the structure of the Laplacian.
    """

    correlation_function = StringChoice.T(
        default="nearest_neighbor",
        choices=_correlation_function_choices,
        help="Determines the correlation function for smoothing across "
        "patches. Choices: %s" % utility.list2string(_correlation_function_choices),
    )

    def get_hypernames(self):
        return [hyper_name_laplacian]


regularization_catalog = {
    "laplacian": LaplacianRegularizationConfig,
    "none": NoneRegularizationConfig,
}


def _init_kwargs(method_config_name, method_name, method_catalog, kwargs):
    """
    Fiddle with input arguments for method_config and initialise sub method
    config according to requested method name.
    """
    method_config = kwargs.pop(method_config_name, None)
    method = kwargs.pop(method_name, None)

    if method and not method_config:
        kwargs[method_config_name] = method_catalog[method]()
    elif method and method_config:
        wanted_config = method_catalog[method]
        if not isinstance(method_config, wanted_config):
            logger.info(
                "%s method changed!" " Initializing new config..." % method_name
            )
            kwargs[method_config_name] = wanted_config()
        else:
            kwargs[method_config_name] = method_config

    if method:
        kwargs[method_name] = method

    return kwargs


class FFIConfig(ModeConfig):

    regularization = StringChoice.T(
        default="none",
        choices=_regularization_choices,
        help="Flag for regularization in distributed slip-optimization."
        " Choices: %s" % utility.list2string(_regularization_choices),
    )
    regularization_config = RegularizationConfig.T(
        optional=True,
        default=None,
        help="Additional configuration parameters for regularization",
    )
    initialization = StringChoice.T(
        default="random",
        choices=_initialization_choices,
        help="Initialization of chain starting points, default: random."
        " Choices: %s" % utility.list2string(_initialization_choices),
    )
    npatches = Int.T(
        default=None,
        optional=True,
        help="Number of patches on full fault. Must not be edited manually!"
        " Please edit indirectly through patch_widths and patch_lengths"
        " parameters!",
    )
    subfault_npatches = List.T(
        Int.T(),
        default=[],
        optional=True,
        help="Number of patches on each sub-fault."
        " Must not be edited manually!"
        " Please edit indirectly through patch_widths and patch_lengths"
        " parameters!",
    )

    def __init__(self, **kwargs):

        kwargs = _init_kwargs(
            method_config_name="regularization_config",
            method_name="regularization",
            method_catalog=regularization_catalog,
            kwargs=kwargs,
        )

        Object.__init__(self, **kwargs)


def get_parameter(variable, nvars=1, lower=1, upper=2):
    return Parameter(
        name=variable,
        lower=num.full(shape=(nvars,), fill_value=lower, dtype=tconfig.floatX),
        upper=num.full(shape=(nvars,), fill_value=upper, dtype=tconfig.floatX),
        testvalue=num.full(
            shape=(nvars,), fill_value=(lower + (upper / 5.0)), dtype=tconfig.floatX
        ),
    )


class ProblemConfig(Object):
    """
    Config for optimization problem to setup.
    """

    mode = StringChoice.T(
        choices=_mode_choices,
        default=geometry_mode_str,
        help="Problem to solve. Choices: %s" % utility.list2string(_mode_choices),
    )
    mode_config = ModeConfig.T(
        optional=True, help="Global optimization mode specific parameters."
    )
    source_type = StringChoice.T(
        default="RectangularSource",
        choices=source_names,
        help="Source type to optimize for. Choices: %s"
        % (", ".join(name for name in source_names)),
    )
    stf_type = StringChoice.T(
        default="HalfSinusoid",
        choices=stf_names,
        help="Source time function type to use. Choices: %s"
        % (", ".join(name for name in stf_names)),
    )
    decimation_factors = Dict.T(
        default=None,
        optional=True,
        help="Determines the reduction of discretization of an extended" " source.",
    )
    n_sources = Int.T(default=1, help="Number of Sub-sources to solve for")
    datatypes = List.T(default=["geodetic"])
    hyperparameters = Dict.T(
        default=OrderedDict(),
        help="Hyperparameters to estimate the noise in different"
        " types of datatypes.",
    )
    priors = Dict.T(default=OrderedDict(), help="Priors of the variables in question.")
    hierarchicals = Dict.T(
        default=OrderedDict(),
        help="Hierarchical parameters that affect the posterior"
        " likelihood, but do not affect the forward problem."
        " Implemented: Temporal station corrections, orbital"
        " ramp estimation",
    )

    def __init__(self, **kwargs):

        mode = "mode"
        mode_config = "mode_config"
        if mode in kwargs:
            omode = kwargs[mode]

            if omode == ffi_mode_str:
                if mode_config not in kwargs:
                    kwargs[mode_config] = FFIConfig()

        Object.__init__(self, **kwargs)

    def init_vars(self, variables=None, nvars=None):
        """
        Initiate priors based on the problem mode and datatypes.

        Parameters
        ----------
        variables : list
            of str of variable names to initialise
        """
        if variables is None:
            variables = self.select_variables()

        self.priors = OrderedDict()

        for variable in variables:

            if nvars is None:
                if variable in block_vars:
                    nvars = 1
                else:
                    nvars = self.n_sources

            lower = default_bounds[variable][0]
            upper = default_bounds[variable][1]
            self.priors[variable] = get_parameter(variable, nvars, lower, upper)

    def set_vars(self, bounds_dict, attribute="priors", init=False):
        """
        Set variable bounds to given bounds.
        """
        for variable, bounds in bounds_dict.items():
            upd_dict = getattr(self, attribute)
            if variable in list(upd_dict.keys()) or init:
                if init:
                    logger.info(
                        'Initialising new variable "%s" in %s' % (variable, attribute)
                    )
                    param = get_parameter(variable, nvars=len(bounds[0]))
                    upd_dict[variable] = param
                else:
                    param = upd_dict[variable]

                param.lower = num.atleast_1d(bounds[0])
                param.upper = num.atleast_1d(bounds[1])
                try:
                    param.testvalue = num.atleast_1d(bounds[2])
                except IndexError:
                    param.testvalue = num.atleast_1d(num.mean(bounds, axis=0))
            else:
                logger.warning(
                    "Prior for variable %s does not exist!"
                    " Bounds not updated!" % variable
                )

        setattr(self, attribute, upd_dict)

    def select_variables(self):
        """
        Return model variables depending on problem config.
        """

        if self.mode not in modes_catalog.keys():
            raise ValueError("Problem mode %s not implemented" % self.mode)

        vars_catalog = modes_catalog[self.mode]

        variables = []
        for datatype in self.datatypes:
            if datatype in vars_catalog.keys():
                if self.mode == geometry_mode_str:
                    if self.source_type in vars_catalog[datatype].keys():
                        source = vars_catalog[datatype][self.source_type]
                        svars = set(source.keys())

                        if isinstance(source(), (PyrockoRS, gf.ExplosionSource)):
                            svars.discard("magnitude")

                        variables += utility.weed_input_rvs(svars, self.mode, datatype)
                    else:
                        raise ValueError(
                            "Source Type not supported for type"
                            " of problem, and datatype!"
                        )

                    if datatype == "seismic":
                        if self.stf_type in stf_catalog.keys():
                            stf = stf_catalog[self.stf_type]
                            variables += utility.weed_input_rvs(
                                set(stf.keys()), self.mode, datatype
                            )
                else:
                    variables += vars_catalog[datatype]
            else:
                raise ValueError(
                    "Datatype %s not supported for type of"
                    " problem! Supported datatype are: %s"
                    % (datatype, ", ".join('"%s"' % d for d in vars_catalog.keys()))
                )

        unique_variables = utility.unique_list(variables)

        if len(unique_variables) == 0:
            raise Exception(
                "Mode and datatype combination not implemented"
                " or not resolvable with given datatypes."
            )

        return unique_variables

    def get_random_variables(self):
        """
        Evaluate problem setup and return random variables dictionary.

        Returns
        -------
        rvs : dict
            variable random variables
        fixed_params : dict
            fixed random parameters
        """
        from pymc3 import Uniform

        logger.debug("Optimization for %i sources", self.n_sources)

        rvs = dict()
        fixed_params = dict()
        for param in self.priors.values():
            if not num.array_equal(param.lower, param.upper):

                shape = self.get_parameter_shape(param)

                kwargs = dict(
                    name=param.name,
                    shape=num.sum(shape),
                    lower=param.get_lower(shape),
                    upper=param.get_upper(shape),
                    testval=param.get_testvalue(shape),
                    transform=None,
                    dtype=tconfig.floatX,
                )
                try:
                    rvs[param.name] = Uniform(**kwargs)

                except TypeError:
                    kwargs.pop("name")
                    rvs[param.name] = Uniform.dist(**kwargs)

            else:
                logger.info(
                    "not solving for %s, got fixed at %s"
                    % (param.name, utility.list2string(param.lower.flatten()))
                )
                fixed_params[param.name] = param.lower

        return rvs, fixed_params

    def get_slip_variables(self):
        """
        Return a list of slip variable names defined in the ProblemConfig.
        """
        if self.mode == ffi_mode_str:
            return [var for var in static_dist_vars if var in self.priors.keys()]
        elif self.mode == geometry_mode_str:
            return [var for var in ["slip", "magnitude"] if var in self.priors.keys()]
        elif self.mode == "interseismic":
            return ["bl_amplitude"]

    def set_decimation_factor(self):
        """
        Determines the reduction of discretization of an extended source.
        Influences yet only the RectangularSource.
        """
        if self.source_type == "RectangularSource":
            self.decimation_factors = {}
            for datatype in self.datatypes:
                self.decimation_factors[datatype] = default_decimation_factors[datatype]
        else:
            self.decimation_factors = None

    def _validate_parameters(self, dict_name=None):
        """
        Check if parameters and their test values do not contradict!

        Parameters
        ----------
        dict_name : str
            of dict to be tested
        """

        d = getattr(self, dict_name)
        double_check = []
        if d is not None:
            for name, param in d.items():
                param.validate_bounds()
                if name not in double_check:
                    if name != param.name:
                        raise InconsistentParameterNaming(name, param.name, self.mode)
                    double_check.append(name)
                else:
                    raise ValueError(
                        "Parameter %s not unique in %s!".format(name, dict_name)
                    )

            logger.info("All {} ok!".format(dict_name))
        else:
            logger.info("No {} defined!".format(dict_name))

    def validate_all(self):
        """
        Validate all involved sampling parameters.
        """
        self.validate_hierarchicals()
        self.validate_hypers()
        self.validate_priors()

    def validate_priors(self):
        """
        Check if priors and their test values do not contradict!
        """
        self._validate_parameters(dict_name="priors")

    def validate_hypers(self):
        """
        Check if hyperparameters and their test values do not contradict!
        """
        self._validate_parameters(dict_name="hyperparameters")

    def validate_hierarchicals(self):
        """
        Check if hierarchicals and their test values do not contradict!
        """
        self._validate_parameters(dict_name="hierarchicals")

    def get_test_point(self):
        """
        Returns dict with test point
        """
        test_point = {}
        for varname, var in self.priors.items():
            shape = self.get_parameter_shape(var)
            test_point[varname] = var.get_testvalue(shape)

        for varname, var in self.hyperparameters.items():
            test_point[varname] = var.get_testvalue()

        for varname, var in self.hierarchicals.items():
            test_point[varname] = var.get_testvalue()

        return test_point

    def get_parameter_shape(self, param):
        if self.mode == ffi_mode_str:
            if param.name in hypo_vars:
                shape = self.n_sources
            elif param.name not in hypo_vars and self.mode_config.npatches:
                shape = self.mode_config.subfault_npatches
                if len(shape) == 0:
                    shape = self.mode_config.npatches
            else:
                shape = param.dimension

        elif self.mode == geometry_mode_str:
            shape = param.dimension
        else:
            raise TypeError("Mode not implemented: %s" % self.mode)

        return shape

    def get_derived_variables_shapes(self):

        source_type = self.source_type

        tpoint = self.get_test_point()
        has_pole, _ = check_point_keys(tpoint, phrase="*_pole_lat")

        if has_pole:
            source_type += "Pole"

        try:
            varnames = derived_variables_mapping[source_type]
            shapes = []
            for varname in varnames:
                if self.mode == geometry_mode_str:
                    shape = (self.n_sources,)
                elif self.mode == ffi_mode_str:
                    if varname == "magnitude":
                        shape = (1,)
                    else:
                        shape = (self.mode_config.npatches,)

                shapes.append(shape)

            logger.info(
                "Adding derived variables %s with shapes %s to "
                "trace." % (list2string(varnames), list2string(shapes))
            )
        except KeyError:
            logger.info("No derived variables for %s" % source_type)
            varnames = []
            shapes = []

        return varnames, shapes


class SamplerParameters(Object):

    tune_interval = Int.T(
        default=50, help="Tune interval for adaptive tuning of Metropolis step size."
    )
    proposal_dist = String.T(
        default="Normal",
        help="Normal Proposal distribution, for Metropolis steps;"
        "Alternatives: Cauchy, Laplace, Poisson, MultivariateNormal",
    )
    check_bnd = Bool.T(
        default=True,
        help="Flag for checking whether proposed step lies within" " variable bounds.",
    )

    rm_flag = Bool.T(default=False, help="Remove existing results prior to sampling.")


class ParallelTemperingConfig(SamplerParameters):

    n_samples = Int.T(
        default=int(1e5),
        help="Number of samples of the posterior distribution."
        " Only the samples of processors that sample from the posterior"
        " (beta=1) are kept.",
    )
    n_chains = Int.T(
        default=2,
        help="Number of PT chains to sample in parallel."
        " A number < 2 will raise an Error, as this is the minimum"
        " amount of chains needed. ",
    )
    swap_interval = Tuple.T(
        2,
        Int.T(),
        default=(100, 300),
        help="Interval for uniform random integer that is drawn to determine"
        " the length of MarkovChains on each worker. When chain is"
        " completed the last sample is returned for swapping state"
        " between chains. Consequently, lower number will result in"
        " more state swapping.",
    )
    beta_tune_interval = Int.T(
        default=int(5e3),
        help="Sample interval of master chain after which the chain swap"
        " acceptance is evaluated. High acceptance will result in"
        " closer spaced betas and vice versa.",
    )
    n_chains_posterior = Int.T(
        default=1, help="Number of chains that sample from the posterior at beat=1."
    )
    resample = Bool.T(
        default=False,
        help='If "true" the testvalue of the priors is taken as seed for'
        " all Markov Chains.",
    )
    thin = Int.T(
        default=3,
        help='Thinning parameter of the sampled trace. Every "thin"th sample'
        " is taken.",
    )
    burn = Float.T(
        default=0.5,
        help="Burn-in parameter between 0. and 1. to discard fraction of"
        " samples from the beginning of the chain.",
    )
    record_worker_chains = Bool.T(
        default=False,
        help="If True worker chain samples are written to disc using the"
        " specified backend trace objects (during sampler initialization)."
        " Very useful for debugging purposes. MUST be False for runs on"
        " distributed computing systems!",
    )


class MetropolisConfig(SamplerParameters):
    """
    Config for optimization parameters of the Adaptive Metropolis algorithm.
    """

    n_jobs = Int.T(
        default=1,
        help="Number of processors to use, i.e. chains to sample in parallel.",
    )
    n_steps = Int.T(default=25000, help="Number of steps for the MC chain.")
    n_chains = Int.T(default=20, help="Number of Metropolis chains for sampling.")
    thin = Int.T(
        default=2,
        help='Thinning parameter of the sampled trace. Every "thin"th sample'
        " is taken.",
    )
    burn = Float.T(
        default=0.5,
        help="Burn-in parameter between 0. and 1. to discard fraction of"
        " samples from the beginning of the chain.",
    )


class SMCConfig(SamplerParameters):
    """
    Config for optimization parameters of the SMC algorithm.
    """

    n_jobs = Int.T(
        default=1,
        help="Number of processors to use, i.e. chains to sample in parallel.",
    )
    n_steps = Int.T(default=100, help="Number of steps for the MC chain.")
    n_chains = Int.T(default=1000, help="Number of Metropolis chains for sampling.")
    coef_variation = Float.T(
        default=1.0,
        help="Coefficient of variation, determines the similarity of the"
        "intermediate stage pdfs;"
        "low - small beta steps (slow cooling),"
        "high - wide beta steps (fast cooling)",
    )
    stage = Int.T(
        default=0,
        help="Stage where to start/continue the sampling. Has to"
        " be int -1 for final stage",
    )
    proposal_dist = String.T(
        default="MultivariateNormal",
        help="Multivariate Normal Proposal distribution, for Metropolis steps"
        "alternatives need to be implemented",
    )

    update_covariances = Bool.T(
        default=False,
        help="Update model prediction covariance matrixes in transition " "stages.",
    )


sampler_catalog = {
    "PT": ParallelTemperingConfig,
    "SMC": SMCConfig,
    "Metropolis": MetropolisConfig,
}


class SamplerConfig(Object):
    """
    Config for the sampler specific parameters.
    """

    name = StringChoice.T(
        default="SMC",
        choices=_sampler_choices,
        help="Sampler to use for sampling the solution space. "
        "Choices: %s" % utility.list2string(_sampler_choices),
    )
    backend = StringChoice.T(
        default="csv",
        choices=_backend_choices,
        help="File type to store output traces. Binary is fast, "
        "csv is good for easy sample inspection. Choices: %s."
        " Default: csv" % utility.list2string(_backend_choices),
    )
    progressbar = Bool.T(default=True, help="Display progressbar(s) during sampling.")
    buffer_size = Int.T(
        default=5000,
        help="number of samples after which the result " "buffer is written to disk",
    )
    buffer_thinning = Int.T(
        default=1,
        help="Factor by which the result trace is thinned before " "writing to disc.",
    )
    parameters = SamplerParameters.T(
        default=SMCConfig.D(), help="Sampler tependend Parameters"
    )

    def __init__(self, **kwargs):

        kwargs = _init_kwargs(
            method_config_name="parameters",
            method_name="name",
            method_catalog=sampler_catalog,
            kwargs=kwargs,
        )

        Object.__init__(self, **kwargs)


class GFLibaryConfig(Object):
    """
    Baseconfig for GF Libraries
    """

    component = String.T(default="uparr")
    event = model.Event.T(default=model.Event.D())
    crust_ind = Int.T(default=0)
    reference_sources = List.T(
        RectangularSource.T(), help="Geometry of the reference source(s) to fix"
    )


class GeodeticGFLibraryConfig(GFLibaryConfig):
    """
    Config for the linear Geodetic GF Library for dumping and loading.
    """

    dimensions = Tuple.T(2, Int.T(), default=(0, 0))
    datatype = String.T(default="geodetic")


class SeismicGFLibraryConfig(GFLibaryConfig):
    """
    Config for the linear Seismic GF Library for dumping and loading.
    """

    wave_config = WaveformFitConfig.T(default=WaveformFitConfig.D())
    starttime_sampling = Float.T(default=0.5)
    duration_sampling = Float.T(default=0.5)
    starttime_min = Float.T(default=0.0)
    duration_min = Float.T(default=0.1)
    dimensions = Tuple.T(5, Int.T(), default=(0, 0, 0, 0, 0))
    datatype = String.T(default="seismic")
    mapnumber = Int.T(default=None, optional=True)

    @property
    def _mapid(self):
        if hasattr(self, "mapnumber"):
            if self.mapnumber is not None:
                return "_".join((self.wave_config.name, str(self.mapnumber)))
        else:
            return self.wave_config.name


datatype_catalog = {
    "polarity": PolarityConfig,
    "geodetic": GeodeticConfig,
    "seismic": SeismicConfig,
}


class BEATconfig(Object, Cloneable):
    """
    BEATconfig is the overarching configuration class, providing all the
    sub-configurations classes for the problem setup, Greens Function
    generation, optimization algorithm and the data being used.
    """

    name = String.T()
    date = String.T()
    event = model.Event.T(optional=True)
    subevents = List.T(
        model.Event.T(),
        default=[],
        help="Event objects of other events that are supposed to be estimated "
        "jointly with the main event. "
        "May have large temporal separation.",
    )
    project_dir = String.T(default="event/")

    problem_config = ProblemConfig.T(default=ProblemConfig.D())
    geodetic_config = GeodeticConfig.T(default=None, optional=True)
    seismic_config = SeismicConfig.T(default=None, optional=True)
    polarity_config = PolarityConfig.T(default=None, optional=True)
    sampler_config = SamplerConfig.T(default=SamplerConfig.D())
    hyper_sampler_config = SamplerConfig.T(default=SamplerConfig.D(), optional=True)

    def update_hypers(self):
        """
        Evaluate the whole config and initialise necessary hyperparameters.
        """

        hypernames = []
        for datatype in _datatype_choices:
            datatype_conf = getattr(self, "%s_config" % datatype)
            if datatype_conf is not None:
                hypernames.extend(datatype_conf.get_hypernames())

        if self.problem_config.mode == ffi_mode_str:
            if self.problem_config.mode_config.regularization == "laplacian":
                hypernames.append(hyper_name_laplacian)

        hypers = OrderedDict()
        for name in hypernames:
            logger.info("Added hyperparameter %s to config and " "model setup!" % name)

            defaultb_name = "hypers"
            hypers[name] = Parameter(
                name=name,
                lower=num.ones(1, dtype=tconfig.floatX)
                * default_bounds[defaultb_name][0],
                upper=num.ones(1, dtype=tconfig.floatX)
                * default_bounds[defaultb_name][1],
                testvalue=num.ones(1, dtype=tconfig.floatX)
                * num.mean(default_bounds[defaultb_name]),
            )

        self.problem_config.hyperparameters = hypers
        self.problem_config.validate_hypers()

        n_hypers = len(hypers)
        logger.info("Number of hyperparameters! %i" % n_hypers)
        if n_hypers == 0:
            self.hyper_sampler_config = None

    def update_hierarchicals(self):
        """
        Evaluate the whole config and initialise necessary
        hierarchical parameters.
        """

        hierarnames = []
        if self.geodetic_config is not None:
            if self.geodetic_config.corrections_config.has_enabled_corrections:
                logger.info(
                    "Loading geodetic data to resolve " "correction dependencies..."
                )
                geodetic_data_path = os.path.join(self.project_dir, geodetic_data_name)

                datasets = utility.load_objects(geodetic_data_path)
                hierarnames.extend(
                    self.geodetic_config.get_hierarchical_names(datasets)
                )
            else:
                logger.info("No corrections enabled")

        if self.seismic_config is not None:
            hierarnames.extend(self.seismic_config.get_hierarchical_names())

        hierarchicals = OrderedDict()
        shp = 1
        for name in hierarnames:
            logger.info(
                "Added hierarchical parameter %s to config and " "model setup!" % name
            )

            if name == "time_shift":
                defaultb_name = name
            else:
                correction_name = name.split("_")[-1]
                defaultb_name = correction_name

            hierarchicals[name] = Parameter(
                name=name,
                lower=num.ones(shp, dtype=tconfig.floatX)
                * default_bounds[defaultb_name][0],
                upper=num.ones(shp, dtype=tconfig.floatX)
                * default_bounds[defaultb_name][1],
                testvalue=num.ones(shp, dtype=tconfig.floatX)
                * num.mean(default_bounds[defaultb_name]),
            )

        self.problem_config.hierarchicals = hierarchicals
        self.problem_config.validate_hierarchicals()

        n_hierarchicals = len(hierarchicals)
        logger.info("Number of hierarchicals! %i" % n_hierarchicals)


def init_reference_sources(source_points, n_sources, source_type, stf_type, event=None):
    """
    Initialise sources of specified geometry

    Parameters
    ----------
    source_points : list
        of dicts or kite sources
    """
    isdict = isinstance(source_points[0], dict)

    reference_sources = []
    for i in range(n_sources):
        # rf = source_catalog[source_type](stf=stf_catalog[stf_type]())
        # maybe future if several meshtypes
        stf = stf_catalog[stf_type](anchor=-1)
        if isdict:
            rf = RectangularSource(stf=stf, anchor="top")
            utility.update_source(rf, **source_points[i])
        else:
            kwargs = {}
            kwargs["stf"] = stf
            rf = RectangularSource.from_kite_source(source_points[i], kwargs=kwargs)

        rf.nucleation_x = None
        rf.nucleation_y = None
        if event is not None:
            rf.update(time=event.time)
            if rf.lat == 0 and rf.lon == 0:
                logger.info(
                    "Reference source is configured without Latitude "
                    "and Longitude! Updating with event information! ..."
                )
                rf.update(lat=event.lat, lon=event.lon)
            reference_sources.append(rf)

    return reference_sources


def init_config(
    name,
    date=None,
    min_magnitude=6.0,
    main_path="./",
    datatypes=["geodetic"],
    mode="geometry",
    source_type="RectangularSource",
    n_sources=1,
    waveforms=["any_P"],
    sampler="SMC",
    hyper_sampler="Metropolis",
    use_custom=False,
    individual_gfs=False,
):
    """
    Initialise BEATconfig File and write it main_path/name .
    Fine parameters have to be edited in the config file .yaml manually.

    Parameters
    ----------
    name : str
        Name of the event
    date : str
        'YYYY-MM-DD', date of the event
    min_magnitude : scalar, float
        approximate minimum Mw of the event
    datatypes : List of strings
        data sets to include in the optimization: either 'geodetic' and/or
        'seismic'
    mode : str
        type of optimization problem: 'Geometry' / 'Static'/ 'Kinematic'
    n_sources : int
        number of sources to solve for / discretize depending on mode parameter
    wavenames : list
        of strings of wavenames to include into the misfit function and
        GF calculation
    sampler : str
        Optimization algorithm to use to sample the solution space
        Options: 'SMC', 'Metropolis'
    use_custom : boolean
        Flag to setup manually a custom velocity model.
    individual_gfs : boolean
        Flag to use individual Green's Functions for each specific station.
        If false a reference location will be initialised in the config file.
        If true the reference locations will be taken from the imported station
        objects.

    Returns
    -------
    :class:`BEATconfig`
    """

    def init_dataset_config(config, datatype):

        dconfig = datatype_catalog[datatype]()

        if hasattr(dconfig.gf_config, "reference_location"):
            if not individual_gfs:
                dconfig.gf_config.reference_location = ReferenceLocation(
                    lat=10.0, lon=10.0
                )
            else:
                dconfig.gf_config.reference_location = None

        if use_custom:
            logger.info(
                "use_custom flag set! The velocity model in the"
                " %s GF configuration has to be updated!" % datatype
            )
            dconfig.gf_config.custom_velocity_model = load_model().extract(
                depth_max=100.0 * km
            )
            dconfig.gf_config.use_crust2 = False
            dconfig.gf_config.replace_water = False

        config["%s_config" % datatype] = dconfig
        return config

    c = BEATconfig(name=name, date=date)
    c.project_dir = os.path.join(os.path.abspath(main_path), name)

    if mode == geometry_mode_str or mode == "interseismic":
        if date is not None and not mode == "interseismic":
            c.event = utility.search_catalog(date=date, min_magnitude=min_magnitude)

        elif mode == "interseismic":
            c.event = model.Event(lat=10.0, lon=10.0, depth=0.0)
            c.date = "dummy"
            logger.info(
                "Interseismic mode! Using event as reference for the"
                " stable block! Please update coordinates!"
            )
        else:
            logger.warn(
                "No given date! Using dummy event!"
                " Updating reference coordinates (spatial & temporal)"
                " necessary!"
            )
            c.event = model.Event(duration=1.0)
            c.date = "dummy"

        for datatype in datatypes:
            init_dataset_config(c, datatype=datatype)

    elif mode == ffi_mode_str:

        if source_type != "RectangularSource":
            raise TypeError(
                "Distributed slip is so far only supported" " for RectangularSource(s)"
            )

        try:
            gmc = load_config(c.project_dir, geometry_mode_str)
        except IOError:
            raise ImportError(
                "No geometry configuration file existing! Please initialise"
                ' a "%s" configuration ("beat init command"), update'
                " the Greens Function information and create GreensFunction"
                " stores for the non-linear problem." % geometry_mode_str
            )

        logger.info("Taking information from geometry_config ...")
        if source_type != gmc.problem_config.source_type:
            raise ValueError(
                'Specified reference source: "%s" differs from the'
                " source that has been used previously in"
                ' "geometry" mode: "%s"!'
                % (source_type, gmc.problem_config.source_type)
            )

        n_sources = gmc.problem_config.n_sources
        point = {k: v.testvalue for k, v in gmc.problem_config.priors.items()}
        point = utility.adjust_point_units(point)
        source_points = utility.split_point(point)

        reference_sources = init_reference_sources(
            source_points,
            n_sources,
            gmc.problem_config.source_type,
            gmc.problem_config.stf_type,
            event=gmc.event,
        )

        c.date = gmc.date
        c.event = gmc.event

        if "geodetic" in datatypes:
            gc = gmc.geodetic_config
            if gc is None:
                logger.warning(
                    'Asked for "geodetic" datatype but %s config '
                    'has no such datatype! Initialising default "geodetic"'
                    " linear config!" % geometry_mode_str
                )
                gc = GeodeticConfig()
                lgf_config = GeodeticLinearGFConfig()
            else:
                logger.info("Initialising geodetic config")
                lgf_config = GeodeticLinearGFConfig(
                    earth_model_name=gc.gf_config.earth_model_name,
                    store_superdir=gc.gf_config.store_superdir,
                    n_variations=gc.gf_config.n_variations,
                    reference_sources=reference_sources,
                    sample_rate=gc.gf_config.sample_rate,
                )

            c.geodetic_config = gc
            c.geodetic_config.gf_config = lgf_config

        if "seismic" in datatypes:
            sc = gmc.seismic_config
            if sc is None:
                logger.warning(
                    'Asked for "seismic" datatype but %s config '
                    'has no such datatype! Initialising default "seismic"'
                    " linear config!" % geometry_mode_str
                )
                sc = SeismicConfig(mode=mode)
                lgf_config = SeismicLinearGFConfig()
            else:
                logger.info("Initialising seismic config")
                lgf_config = SeismicLinearGFConfig(
                    earth_model_name=sc.gf_config.earth_model_name,
                    sample_rate=sc.gf_config.sample_rate,
                    reference_location=sc.gf_config.reference_location,
                    store_superdir=sc.gf_config.store_superdir,
                    n_variations=sc.gf_config.n_variations,
                    reference_sources=reference_sources,
                )
            c.seismic_config = sc
            c.seismic_config.gf_config = lgf_config

    c.problem_config = ProblemConfig(
        n_sources=n_sources, datatypes=datatypes, mode=mode, source_type=source_type
    )
    c.problem_config.init_vars()
    c.problem_config.set_decimation_factor()

    c.sampler_config = SamplerConfig(name=sampler)
    c.hyper_sampler_config = SamplerConfig(name=hyper_sampler)

    c.update_hypers()
    c.problem_config.validate_priors()

    c.regularize()
    c.validate()

    logger.info("Project_directory: %s \n" % c.project_dir)
    util.ensuredir(c.project_dir)

    dump_config(c)
    return c


def dump_config(config):
    """
    Dump configuration file.

    Parameters
    ----------
    config : :class:`BEATConfig`
    """
    config_file_name = "config_" + config.problem_config.mode + ".yaml"
    conf_out = os.path.join(config.project_dir, config_file_name)
    dump(config, filename=conf_out)


def load_config(project_dir, mode):
    """
    Load configuration file.

    Parameters
    ----------
    project_dir : str
        path to the directory of the configuration file
    mode : str
        type of optimization problem: 'geometry' / 'static'/ 'kinematic'
    update : list
        of strings to update parameters
        'hypers' or/and 'hierarchicals'

    Returns
    -------
    :class:`BEATconfig`
    """
    config_file_name = "config_" + mode + ".yaml"

    config_fn = os.path.join(project_dir, config_file_name)

    try:
        config = load(filename=config_fn)
    except IOError:
        raise IOError("Cannot load config, file %s" " does not exist!" % config_fn)
    except (ArgumentError, TypeError):
        raise ConfigNeedsUpdatingError()

    config.problem_config.validate_all()
    return config
