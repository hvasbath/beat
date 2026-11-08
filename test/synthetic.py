"""
Self-contained synthetic BEAT projects for the test-suite.

Green's function stores are calculated with the pyrocko ``ahfullgreen``
backend (homogeneous full-space), which is part of pyrocko itself, so no
external modelling codes (QSEIS, PSGRN/PSCMP) are needed and a store builds in
about a second.

The data are forward modelled with BEAT's own sources at the test point of the
problem configuration and seeded white noise of known standard deviation is
added. Hence, the test point (:attr:`SyntheticProject.point`) is the "true"
model of the synthetic data.

The full-space is not a realistic medium for surface observations (there is no
free surface), but for testing the BEAT machinery only consistency between
store, data and forward model matters.
"""

import copy
import os
import shutil
from dataclasses import dataclass, field

import numpy as num
from pyrocko import cake, gf, model, orthodrome
from pyrocko import util as putil
from pyrocko.fomosto import ahfullgreen
from pyrocko.model import gnss

from beat import config as bconfig
from beat import heart, inputf, utility
from beat.models.problems import SourceOptimizer

km = 1000.0

EVENT = model.Event(
    name="synthetic",
    lat=10.0,
    lon=20.0,
    depth=5.0 * km,
    time=putil.str_to_time("2020-01-01 00:00:00"),
    magnitude=5.0,
    duration=1.0,
)

MATERIAL = cake.Material(vp=5500.0, vs=3200.0, rho=2700.0, qp=1000.0, qs=500.0)

# GF store setup, store ids follow the BEAT naming convention (heart.get_store_id)
EARTH_MODEL_NAME = "ak135-f-continental.f"
REFERENCE_STATION = "SYN"
SEISMIC_SAMPLE_RATE = 2.0  # [Hz]
GEODETIC_SAMPLE_RATE = bconfig.GeodeticGFConfig().sample_rate  # used for naming only
STORE_DEPTH_MIN = 1.0  # [km]
STORE_DEPTH_MAX = 10.0
STORE_SPACING = 1.0
SEISMIC_DISTANCE_MAX = 60.0
GEODETIC_DISTANCE_MAX = 40.0

STATION_DISTANCES = [15.0, 25.0, 35.0, 45.0]  # [km]
STATION_AZIMUTHS = [10.0, 100.0, 190.0, 280.0, 55.0, 145.0, 235.0, 325.0]  # [deg]

SAR_HALF_EXTENT = 20.0  # [km]
SAR_SPACING = 2.0
N_GNSS_STATIONS = 10

# priors as (lower, upper, testvalue = true value)
PRIORS = {
    "MTSource": {
        "mnn": (-1.4142135623730951, 1.4142135623730951, -0.43283071),
        "mee": (-1.4142135623730951, 1.4142135623730951, 0.65741974),
        "mdd": (-1.4142135623730951, 1.4142135623730951, -0.22458903),
        "mne": (-1.0, 1.0, 0.63839719),
        "mnd": (-1.0, 1.0, 0.50698292),
        "med": (-1.0, 1.0, 0.02063122),
        "magnitude": (4.5, 5.5, 5.0),
        "north_shift": (-3.0, 3.0, 0.5),
        "east_shift": (-3.0, 3.0, -0.5),
        "depth": (3.0, 7.0, 5.0),
        "time": (-2.0, 2.0, 0.5),
        "duration": (0.5, 3.0, 1.0),
    },
    "RectangularSource": {
        "north_shift": (-3.0, 3.0, 0.5),
        "east_shift": (-3.0, 3.0, -0.5),
        "depth": (1.0, 4.0, 2.0),
        "length": (3.0, 8.0, 5.0),
        "width": (2.0, 5.0, 3.0),
        "strike": (0.0, 90.0, 30.0),
        "dip": (30.0, 70.0, 50.0),
        "rake": (45.0, 135.0, 90.0),
        "slip": (0.2, 2.0, 1.0),
        "opening_fraction": (0.0, 0.0, 0.0),
        "nucleation_x": (-1.0, 1.0, 0.0),
        "nucleation_y": (-1.0, 1.0, 0.0),
        "time": (-2.0, 2.0, 0.5),
        "duration": (0.5, 3.0, 1.5),
    },
}


def get_material(crust_ind=0, variation=0.05):
    """
    Reference material for crust_ind 0, otherwise velocities and density are
    varied randomly (seeded by crust_ind) by the given relative amount.
    """
    if crust_ind == 0:
        return MATERIAL

    factors = num.random.RandomState(crust_ind).uniform(
        1.0 - variation, 1.0 + variation, size=3
    )
    return cake.Material(
        vp=MATERIAL.vp * factors[0],
        vs=MATERIAL.vs * factors[1],
        rho=MATERIAL.rho * factors[2],
        qp=MATERIAL.qp,
        qs=MATERIAL.qs,
    )


def homogeneous_earthmodel(zbot=100.0 * km, crust_ind=0):
    mod = cake.LayeredModel()
    mod.append(
        cake.HomogeneousLayer(
            ztop=0.0, zbot=zbot, m=get_material(crust_ind), name="fullspace"
        )
    )
    return mod


def seismic_store_id(crust_ind=0):
    return heart.get_store_id(
        REFERENCE_STATION,
        heart.get_earth_model_prefix(EARTH_MODEL_NAME),
        SEISMIC_SAMPLE_RATE,
        crust_ind,
    )


def geodetic_store_id(crust_ind=0):
    return heart.get_store_id(
        "statics",
        heart.get_earth_model_prefix(EARTH_MODEL_NAME),
        GEODETIC_SAMPLE_RATE,
        crust_ind,
    )


def build_store(
    store_superdir,
    store_id,
    distance_max,
    sample_rate,
    depth_min=STORE_DEPTH_MIN,
    depth_max=STORE_DEPTH_MAX,
    spacing=STORE_SPACING,
    crust_ind=0,
    force=False,
):
    """
    Build a GF store for a homogeneous full-space with the ahfullgreen backend.

    For static stores the last sample of the traces holds the static
    displacement, as the full-space response to a step in moment is static
    after the S-wave arrival.

    Distances, depths and spacing are in [km]. For crust_ind > 0 the material
    is varied (see get_material). Existing stores are not rebuilt unless force
    is True.
    """
    store_dir = os.path.join(store_superdir, store_id)
    if os.path.exists(os.path.join(store_dir, "traces")) and not force:
        return store_dir

    config = gf.ConfigTypeA(
        id=store_id,
        ncomponents=10,
        sample_rate=sample_rate,
        receiver_depth=0.0,
        source_depth_min=depth_min * km,
        source_depth_max=depth_max * km,
        source_depth_delta=spacing * km,
        distance_min=0.0,
        distance_max=distance_max * km,
        distance_delta=spacing * km,
        earthmodel_1d=homogeneous_earthmodel(
            zbot=max(2 * depth_max, 100.0) * km, crust_ind=crust_ind
        ),
        modelling_code_id="ahfullgreen",
        tabulated_phases=[
            gf.TPDef(id="any_P", definition="p,P,p\\,P\\"),
            gf.TPDef(id="any_S", definition="s,S,s\\,S\\"),
        ],
    )
    gf.Store.create_editables(
        store_dir,
        config=config,
        extra={"ahfullgreen": ahfullgreen.AhfullgreenConfig()},
        force=True,
    )
    store = gf.Store(store_dir, "r")
    store.make_travel_time_tables(force=True)
    store.close()
    ahfullgreen.build(store_dir, nworkers=1, force=True)
    return store_dir


def build_stores(store_superdir, datatypes, crust_inds=(0,)):
    for crust_ind in crust_inds:
        if "seismic" in datatypes:
            build_store(
                store_superdir,
                seismic_store_id(crust_ind),
                SEISMIC_DISTANCE_MAX,
                SEISMIC_SAMPLE_RATE,
                crust_ind=crust_ind,
            )
        if "geodetic" in datatypes:
            build_store(
                store_superdir,
                geodetic_store_id(crust_ind),
                GEODETIC_DISTANCE_MAX,
                sample_rate=1.0,
                crust_ind=crust_ind,
            )


def ne_to_latlon(norths, easts):
    return orthodrome.ne_to_latlon(
        EVENT.lat, EVENT.lon, num.asarray(norths) * km, num.asarray(easts) * km
    )


def make_stations():
    """Seismic stations with E, N, Z channels on rings around the event."""
    stations = []
    for i, (azi, dist) in enumerate(
        zip(STATION_AZIMUTHS, STATION_DISTANCES + STATION_DISTANCES)
    ):
        north = dist * num.cos(num.deg2rad(azi))
        east = dist * num.sin(num.deg2rad(azi))
        lat, lon = ne_to_latlon(north, east)
        stations.append(
            model.Station(
                network="SY",
                station="S%02i" % i,
                lat=float(lat),
                lon=float(lon),
                channels=[
                    model.Channel("E", azimuth=90.0, dip=0.0),
                    model.Channel("N", azimuth=0.0, dip=0.0),
                    model.Channel("Z", azimuth=0.0, dip=-90.0),
                ],
            )
        )
    return stations


def make_sar_dataset():
    x = num.arange(-SAR_HALF_EXTENT, SAR_HALF_EXTENT + 0.1, SAR_SPACING)
    norths, easts = [a.ravel() for a in num.meshgrid(x, x)]
    lats, lons = ne_to_latlon(norths, easts)
    n = lats.size
    return heart.DiffIFG(
        name="S1_dsc",
        typ="SAR",
        lats=lats,
        lons=lons,
        incidence=num.full(n, 35.0),
        heading=num.full(n, -166.0),
        displacement=num.zeros(n),
        odw=num.ones(n),
    )


def make_gnss_datasets(rstate):
    norths, easts = rstate.uniform(-25.0, 25.0, size=(2, N_GNSS_STATIONS))
    lats, lons = ne_to_latlon(norths, easts)
    stations = [
        gnss.GNSSStation(
            code="G%02i" % i,
            lat=float(lat),
            lon=float(lon),
            north=gnss.GNSSComponent(shift=0.0, sigma=1.0),
            east=gnss.GNSSComponent(shift=0.0, sigma=1.0),
            up=gnss.GNSSComponent(shift=0.0, sigma=1.0),
        )
        for i, (lat, lon) in enumerate(zip(lats, lons))
    ]
    campaign = gnss.GNSSCampaign(name="synthetic_gnss", stations=stations)
    return heart.GNSSCompoundComponent.from_pyrocko_gnss_campaign(campaign)


def set_priors(config, source_type):
    pc = config.problem_config
    for name, (lower, upper, testvalue) in PRIORS[source_type].items():
        if name in pc.priors:
            param = pc.priors[name]
            param.lower = num.atleast_1d(lower).astype(param.lower.dtype)
            param.upper = num.atleast_1d(upper).astype(param.upper.dtype)
            param.testvalue = num.atleast_1d(testvalue).astype(param.testvalue.dtype)
    pc.validate_priors()


def get_true_sources(config, point):
    """
    Return sources for each datatype updated with the given point, following
    the same steps as the point2sources methods of the geometry composites.
    """
    problem = SourceOptimizer(config)
    pc = config.problem_config
    dsources = utility.transform_sources(
        problem.sources, pc.datatypes, pc.decimation_factors
    )
    mapping = pc.get_variables_mapping()

    for datatype, sources in dsources.items():
        tpoint = utility.adjust_point_units(copy.deepcopy(point))
        if datatype == "seismic" and "time" in tpoint:
            tpoint["time"] += config.event.time

        source_points = utility.split_point(
            tpoint, mapping=mapping[datatype], weed_params=True
        )
        for source, source_point in zip(sources, source_points):
            utility.update_source(source, **source_point)
            if datatype == "geodetic":
                source.time = 0.0

    return dsources


def synthesize_raw_seismic(config, sources, noise_level, rstate):
    """
    Raw (unfiltered, untapered) synthetic E, N, Z traces with white noise.

    Returns stations, :class:`pyrocko.trace.Trace` and the standard deviation
    of the added noise.
    """
    sc = config.seismic_config
    stations = make_stations()
    targets = heart.init_seismic_targets(
        stations,
        earth_model_name=sc.gf_config.earth_model_name,
        channels=["E", "N", "Z"],
        sample_rate=sc.gf_config.sample_rate,
        crust_inds=[0],
        reference_location=sc.gf_config.reference_location,
    )
    engine = gf.LocalEngine(store_superdirs=[sc.gf_config.store_superdir])
    response = engine.process(sources=sources, targets=targets, nthreads=1)

    stacked = {}
    for _, target, tr in response.iter_results():
        if target.codes in stacked:
            stacked[target.codes].add(tr)
        else:
            stacked[target.codes] = tr

    # common time window, long enough for noise analysis before the arrivals
    tmin = config.event.time - 60.0
    tmax = config.event.time + 90.0
    traces = []
    for tr in stacked.values():
        tr.extend(tmin, tmax, fillmethod="repeat")
        tr.chop(tmin, tmax, include_last=False)
        traces.append(tr)

    sigma = noise_level * max(num.abs(tr.ydata).max() for tr in traces)
    for tr in traces:
        tr.ydata = tr.ydata + rstate.normal(scale=sigma, size=tr.ydata.size)

    return stations, traces, sigma


def synthesize_seismic(config, sources, noise_level, rstate):
    """
    Raw synthetic traces rotated to R, T, Z, as from "beat import".

    Returns stations, traces and the standard deviation of the added noise.
    """
    stations, traces, sigma = synthesize_raw_seismic(
        config, sources, noise_level, rstate
    )
    # as inputf.load_data_traces and "beat import" do
    traces = [heart.SeismicDataset.from_pyrocko_trace(tr) for tr in traces]
    traces = inputf.rotate_traces_and_stations(traces, stations, config.event)
    return stations, traces, sigma


def synthesize_geodetic(config, sources, noise_level, rstate):
    """
    SAR and GNSS datasets with static displacements projected to LOS.

    Returns datasets and the standard deviation of the added noise.
    """
    gc = config.geodetic_config
    datasets = [make_sar_dataset()] + make_gnss_datasets(rstate)
    targets = heart.init_geodetic_targets(
        datasets,
        event=config.event,
        earth_model_name=gc.gf_config.earth_model_name,
        interpolation=gc.interpolation,
        crust_inds=[0],
        sample_rate=gc.gf_config.sample_rate,
    )
    engine = gf.LocalEngine(store_superdirs=[gc.gf_config.store_superdir])
    displacements = heart.geo_synthetics(
        engine, targets, sources, outmode="stacked_arrays"
    )

    los_displacements = []
    for disp, data in zip(displacements, datasets):
        data.update_los_vector()
        los_displacements.append((disp * data.los_vector).sum(axis=1))

    sigma = noise_level * max(num.abs(los).max() for los in los_displacements)
    for los, data in zip(los_displacements, datasets):
        n = los.size
        data.displacement = los + rstate.normal(scale=sigma, size=n)
        data.covariance = heart.Covariance(data=num.eye(n) * sigma**2)
        data.odw = num.ones(n)

    return datasets, sigma


def configure_seismic(config, store_superdir):
    sc = config.seismic_config
    sc.datadir = "./"
    sc.noise_estimator.pre_arrival_time = 2.0

    gfc = sc.gf_config
    gfc.store_superdir = store_superdir
    gfc.earth_model_name = EARTH_MODEL_NAME
    gfc.custom_velocity_model = homogeneous_earthmodel()
    gfc.use_crust2 = False
    gfc.replace_water = False
    gfc.sample_rate = SEISMIC_SAMPLE_RATE
    gfc.reference_location = heart.ReferenceLocation(
        lat=EVENT.lat, lon=EVENT.lon, station=REFERENCE_STATION
    )
    gfc.source_depth_min = STORE_DEPTH_MIN
    gfc.source_depth_max = STORE_DEPTH_MAX
    gfc.source_depth_spacing = STORE_SPACING
    gfc.source_distance_radius = SEISMIC_DISTANCE_MAX
    gfc.source_distance_spacing = STORE_SPACING
    gfc.nworkers = 1

    filterer = [heart.Filter(lower_corner=0.05, upper_corner=0.5, order=3)]
    sc.waveforms = [
        bconfig.WaveformFitConfig(
            name=name,
            channels=[channel],
            filterer=filterer,
            distances=(0.0, 1.0),
            interpolation="multilinear",
            arrival_taper=heart.ArrivalTaper(a=-4.0, b=-2.0, c=8.0, d=10.0),
        )
        for name, channel in (("any_P", "Z"), ("any_S", "T"))
    ]


def configure_geodetic(config, store_superdir):
    gc = config.geodetic_config
    gc.interpolation = "multilinear"
    gc.types["SAR"].names = ["S1_dsc"]
    gc.types["GNSS"].names = ["synthetic_gnss"]

    gfc = gc.gf_config
    gfc.store_superdir = store_superdir
    gfc.earth_model_name = EARTH_MODEL_NAME
    gfc.custom_velocity_model = homogeneous_earthmodel()
    gfc.use_crust2 = False
    gfc.replace_water = False
    gfc.source_depth_min = STORE_DEPTH_MIN
    gfc.source_depth_max = STORE_DEPTH_MAX
    gfc.source_depth_spacing = STORE_SPACING
    gfc.source_distance_radius = GEODETIC_DISTANCE_MAX
    gfc.source_distance_spacing = STORE_SPACING
    gfc.nworkers = 1


def enable_geodetic_corrections(config):
    """
    Enable all geodetic corrections and set the test values of the
    hierarchicals to zero correction. Needs the geodetic data in the project.
    """
    corrections = config.geodetic_config.corrections_config
    corrections.ramp.enabled = True
    corrections.ramp.dataset_names = ["S1_dsc"]
    for corr in corrections.euler_poles + corrections.strain_rates:
        corr.enabled = True
        corr.dataset_names = ["synthetic_gnss"]

    config.update_hierarchicals()
    for name, param in config.problem_config.hierarchicals.items():
        if name.endswith(("pole_lat", "pole_lon")):
            continue  # no rotation if omega is zero

        if name.endswith("omega"):
            # default bounds exclude zero rotation
            param.lower = -param.upper

        param.testvalue = num.zeros_like(param.testvalue)

    config.problem_config.validate_hierarchicals()


@dataclass
class SyntheticProject:
    project_dir: str
    store_superdir: str
    mode: str
    datatypes: list
    source_type: str
    point: dict
    noise_sigmas: dict = field(default_factory=dict)
    noise_level: float = 0.01
    seed: int = 0

    @property
    def config(self):
        return bconfig.load_config(self.project_dir, self.mode)

    def load_problem(self, build=True, hypers=False):
        from beat.models import load_model

        return load_model(self.project_dir, self.mode, hypers=hypers, build=build)

    def copy(self, main_path):
        """Copy the project (not the GF stores) to a new main path."""
        project_dir = os.path.join(main_path, os.path.basename(self.project_dir))
        shutil.copytree(self.project_dir, project_dir)
        config = bconfig.load_config(project_dir, self.mode)
        config.project_dir = project_dir
        bconfig.dump_config(config)
        return SyntheticProject(
            project_dir=project_dir,
            store_superdir=self.store_superdir,
            mode=self.mode,
            datatypes=list(self.datatypes),
            source_type=self.source_type,
            point=copy.deepcopy(self.point),
            noise_sigmas=dict(self.noise_sigmas),
            noise_level=self.noise_level,
            seed=self.seed,
        )


def make_project(
    main_path,
    name,
    datatypes,
    source_type="RectangularSource",
    store_superdir=None,
    noise_level=0.01,
    seed=0,
    corrections=False,
):
    """
    Create a BEAT project in geometry mode with synthetic data.

    Parameters
    ----------
    main_path : str
        directory in which the project directory "name" is created
    name : str
        name of the project
    datatypes : list
        of "seismic" and/or "geodetic"
    source_type : str
        "MTSource" or "RectangularSource"
    store_superdir : str
        directory where the GF stores are (or are built if not existing),
        defaults to main_path/GF
    noise_level : float
        standard deviation of the white noise relative to the maximum
        absolute amplitude of all the synthetics of one datatype
    seed : int
        seed of the random generator for the noise
    corrections : bool
        enable geodetic corrections: orbital ramp for the SAR dataset,
        euler pole and strain rate for the GNSS data, the data are
        synthesized without corrections, i.e. the true values are zero

    Returns
    -------
    :class:`SyntheticProject`
    """
    if source_type not in PRIORS:
        raise ValueError("Source type %s not supported" % source_type)

    main_path = os.path.abspath(main_path)
    if store_superdir is None:
        store_superdir = os.path.join(main_path, "GF")
    store_superdir = os.path.abspath(store_superdir)

    build_stores(store_superdir, datatypes)

    config = bconfig.init_config(
        name,
        main_path=main_path,
        datatypes=datatypes,
        mode=bconfig.geometry_mode_str,
        source_types=[source_type],
        n_sources=[1],
    )
    config.event = copy.deepcopy(EVENT)
    config.date = "dummy"

    if "seismic" in datatypes:
        configure_seismic(config, store_superdir)
    if "geodetic" in datatypes:
        configure_geodetic(config, store_superdir)

    set_priors(config, source_type)
    config.update_hypers()
    config.sampler_config.parameters.n_jobs = 1
    config.validate()

    pc = config.problem_config
    point = {name: param.testvalue.copy() for name, param in pc.priors.items()}
    dsources = get_true_sources(config, point)

    rstate = num.random.RandomState(seed)
    noise_sigmas = {}
    if "seismic" in datatypes:
        stations, traces, noise_sigmas["seismic"] = synthesize_seismic(
            config, dsources["seismic"], noise_level, rstate
        )
        utility.dump_objects(
            os.path.join(config.project_dir, bconfig.seismic_data_name),
            outlist=[stations, traces],
        )

    if "geodetic" in datatypes:
        datasets, noise_sigmas["geodetic"] = synthesize_geodetic(
            config, dsources["geodetic"], noise_level, rstate
        )
        utility.dump_objects(
            os.path.join(config.project_dir, bconfig.geodetic_data_name),
            outlist=datasets,
        )

    if corrections and "geodetic" in datatypes:
        enable_geodetic_corrections(config)

    bconfig.dump_config(config)

    return SyntheticProject(
        project_dir=config.project_dir,
        store_superdir=store_superdir,
        mode=bconfig.geometry_mode_str,
        datatypes=list(datatypes),
        source_type=source_type,
        point=point,
        noise_sigmas=noise_sigmas,
        noise_level=noise_level,
        seed=seed,
    )


# priors of the kinematic finite fault problem as (lower, upper, testvalue)
FFI_PRIORS = {
    "uparr": (-0.05, 2.0, 1.0),
    "uperp": (-0.5, 0.5, 0.0),
    "durations": (0.5, 2.0, 1.0),
    "velocities": (2.5, 3.5, 3.0),
    "time": (-1.0, 1.0, 0.5),
}


def make_ffi_project(
    geometry_project,
    main_path,
    datatypes=None,
    patch_size=1.0,
    regularization="none",
    n_variations=None,
):
    """
    Create a finite fault (mode "ffi") project from a synthetic geometry
    project, as with "beat init --mode ffi" and calculate the linear GF
    libraries with "beat build_gfs --mode ffi --execute".

    The reference fault is the true source of the geometry project, it is
    discretized uniformly into patches of patch_size [km].

    If n_variations (int) is given, GF stores and linear GFs are calculated
    for that number of velocity models (crust_inds 0 ... n_variations - 1).

    Returns
    -------
    :class:`SyntheticProject`
    """
    from beat.apps.beat import command_build_gfs

    if datatypes is None:
        datatypes = geometry_project.datatypes

    project = geometry_project.copy(main_path)
    name = os.path.basename(project.project_dir)
    main_path = os.path.dirname(project.project_dir)

    config = bconfig.init_config(
        name,
        main_path=main_path,
        datatypes=datatypes,
        mode=bconfig.ffi_mode_str,
        source_types=["RectangularSource"],
    )

    config.problem_config.mode_config.regularization = regularization
    for datatype in datatypes:
        gfc = config[datatype + "_config"].gf_config
        gfc.nworkers = 1
        dconf = gfc.discretization_config
        dconf.patch_widths = [patch_size]
        dconf.patch_lengths = [patch_size]
        dconf.extension_widths = [0.0]
        dconf.extension_lengths = [0.0]
        if datatype == "seismic":
            gfc.starttime_sampling = 0.5
            gfc.duration_sampling = 0.5
        if n_variations is not None:
            gfc.n_variations = (0, n_variations)

    if n_variations is not None:
        build_stores(project.store_superdir, datatypes, range(n_variations))

    pc = config.problem_config
    for name_, (lower, upper, testvalue) in FFI_PRIORS.items():
        if name_ in pc.priors:
            param = pc.priors[name_]
            param.lower = num.atleast_1d(lower).astype(param.lower.dtype)
            param.upper = num.atleast_1d(upper).astype(param.upper.dtype)
            param.testvalue = num.atleast_1d(testvalue).astype(param.testvalue.dtype)

    config.update_hypers()
    config.validate()
    bconfig.dump_config(config)

    command_build_gfs(
        [
            name,
            "--main_path=%s" % main_path,
            "--mode=%s" % bconfig.ffi_mode_str,
            "--datatypes=%s" % ",".join(datatypes),
            "--execute",
            "--loglevel=warning",
        ]
    )

    config = bconfig.load_config(project.project_dir, bconfig.ffi_mode_str)
    point = {
        name_: param.testvalue.copy()
        for name_, param in config.problem_config.priors.items()
    }
    return SyntheticProject(
        project_dir=project.project_dir,
        store_superdir=project.store_superdir,
        mode=bconfig.ffi_mode_str,
        datatypes=list(datatypes),
        source_type="RectangularSource",
        point=point,
        noise_sigmas={k: v for k, v in project.noise_sigmas.items() if k in datatypes},
        noise_level=project.noise_level,
        seed=project.seed,
    )


# raw data formats, as read by "beat import"

KITE_NPIXELS = 100
KITE_SPACING = 400.0  # [m]
KITE_THETA = 55.0  # elevation angle [deg]
KITE_PHI = -76.0  # [deg] from east, counter-clockwise


def write_seismic_raw(config, sources, datadir, noise_level, rstate):
    """
    Write stations.txt and E, N, Z traces as miniseed files with names
    network.station.location.channel, as read by "beat import".
    """
    from pyrocko import io

    stations, traces, _ = synthesize_raw_seismic(config, sources, noise_level, rstate)
    model.dump_stations(stations, os.path.join(datadir, "stations.txt"))
    for station in stations:
        for tr in traces:
            if tr.station == station.station:
                filename = ".".join(
                    (station.network, station.station, station.location, tr.channel)
                )
                io.save(tr, os.path.join(datadir, filename), format="mseed")


def write_gnss_ascii(datasets, filepath):
    """
    Write GNSS data in the ascii (GLOBK) format of
    :func:`beat.inputf.load_ascii_gnss_globk`, displacements in [mm].
    """
    components = {d.component: d for d in datasets if d.typ == "GNSS"}
    stations = components["north"].stations
    lines = ["# synthetic GNSS data", "# lon lat ve vn ... [mm]", "#"]
    for i, station in enumerate(stations):
        lat, lon = station.effective_latlon
        row = num.zeros(12)
        row[0], row[1] = lon, lat
        for comp, (vel_idx, std_idx) in [
            ("east", (2, 6)),
            ("north", (3, 7)),
            ("up", (9, 11)),
        ]:
            dataset = components[comp]
            row[vel_idx] = dataset.displacement[i] * km
            row[std_idx] = num.sqrt(dataset.covariance.data[i, i]) * km

        lines.append(" ".join("%.12g" % v for v in row) + " " + station.code)

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_kite_scene(config, sources, filepath, noise_level, rstate):
    """
    Write a kite scene of LOS displacements on a regular grid around the event
    with white noise. The LOS projection uses the kite convention
    (Scene.los_rotation_factors), independent of BEAT.
    """
    from kite import Scene
    from kite.scene import FrameConfig, SceneConfig

    gc = config.geodetic_config
    n = KITE_NPIXELS
    half = n * KITE_SPACING / 2.0
    lllat, lllon = orthodrome.ne_to_latlon(EVENT.lat, EVENT.lon, -half, -half)
    coords = num.arange(n) * KITE_SPACING
    norths, easts = [
        a.ravel() for a in num.meshgrid(coords - half, coords - half, indexing="ij")
    ]

    target = gf.StaticTarget(
        lats=num.full(norths.size, EVENT.lat),
        lons=num.full(norths.size, EVENT.lon),
        north_shifts=norths,
        east_shifts=easts,
        interpolation=gc.interpolation,
        quantity="displacement",
        store_id=geodetic_store_id(),
    )
    engine = gf.LocalEngine(store_superdirs=[gc.gf_config.store_superdir])
    disp = heart.geo_synthetics(engine, [target], sources, outmode="stacked_arrays")[0]

    theta = num.full((n, n), num.deg2rad(KITE_THETA))
    phi = num.full((n, n), num.deg2rad(KITE_PHI))
    scene_config = SceneConfig()
    scene_config.frame = FrameConfig(
        llLat=float(lllat),
        llLon=float(lllon),
        dE=KITE_SPACING,
        dN=KITE_SPACING,
        spacing="meter",
    )
    scene = Scene(
        displacement=num.zeros((n, n)), theta=theta, phi=phi, config=scene_config
    )
    factors = scene.los_rotation_factors
    # geo_synthetics returns north, east, up
    north, east, up = [disp[:, i].reshape(n, n) for i in range(3)]
    los = factors[:, :, 0] * up + factors[:, :, 1] * east + factors[:, :, 2] * north

    sigma = noise_level * num.abs(los).max()
    scene.displacement = los + rstate.normal(scale=sigma, size=(n, n))

    scene.quadtree.epsilon = 2 * sigma
    scene.quadtree.tile_size_min = 2 * KITE_SPACING
    scene.quadtree.tile_size_max = 20 * KITE_SPACING

    # The leaf means of white pixel noise are uncorrelated with variance
    # sigma**2 / npixel, which is the exact data covariance. Store only the
    # matrix, otherwise kite re-estimates it on load. (The kite estimate from
    # a noise patch needs a signal free region, which the scene does not have.)
    npixels = num.array([leaf.npixel for leaf in scene.quadtree.leaves])
    config = scene.covariance.config
    config.model_coefficients = None
    config.variance = None
    config.noise_coord = None
    config.covariance_matrix = num.diag(sigma**2 / npixels)
    scene.save(filepath)
    return scene


def write_raw_data(project, datadir):
    """
    Write the data of a synthetic geometry project in raw formats to datadir
    and point the datadir(s) of the project config there, such that the data
    can be imported with "beat import":

    - seismic: stations.txt and miniseed, exactly the data of the project
    - GNSS: ascii (GLOBK) file "synthetic_gnss", exactly the data of the project
    - SAR: kite scene "S1_dsc", a new gridded dataset (quadtree)
    """
    datadir = os.path.abspath(datadir)
    os.makedirs(datadir, exist_ok=True)
    config = bconfig.load_config(project.project_dir, project.mode)
    point = copy.deepcopy(project.point)
    dsources = get_true_sources(config, point)

    if "seismic" in project.datatypes:
        # same random state as in make_project, i.e. identical noise
        write_seismic_raw(
            config,
            dsources["seismic"],
            datadir,
            project.noise_level,
            num.random.RandomState(project.seed),
        )
        config.seismic_config.datadir = datadir

    if "geodetic" in project.datatypes:
        datasets = utility.load_objects(
            os.path.join(project.project_dir, bconfig.geodetic_data_name)
        )
        write_gnss_ascii(datasets, os.path.join(datadir, "synthetic_gnss"))
        write_kite_scene(
            config,
            dsources["geodetic"],
            os.path.join(datadir, "S1_dsc"),
            project.noise_level,
            num.random.RandomState(project.seed + 1),
        )
        for dataset_config in config.geodetic_config.types.values():
            dataset_config.datadir = datadir

    bconfig.dump_config(config)
