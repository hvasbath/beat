"""
Data input functions (beat.inputf) and the dataset configs reading data.
"""

import os

import numpy as num
import pytest
from numpy.testing import assert_allclose
from pyrocko import io, model, trace
from pyrocko import util as putil
from pyrocko.model import gnss

from beat import config as bconfig
from beat import heart, inputf

EVENT = model.Event(lat=10.0, lon=20.0, depth=5000.0, time=0.0)


def make_gnss_datasets(n=5, name="campaign", offset=0.0):
    rstate = num.random.RandomState(1)
    stations = [
        gnss.GNSSStation(
            code="G%02i" % i,
            lat=10.0 + 0.1 * i,
            lon=20.0 - 0.05 * i,
            north=gnss.GNSSComponent(shift=offset + 0.01 * i, sigma=0.002),
            east=gnss.GNSSComponent(shift=offset - 0.02 * i, sigma=0.003),
            up=gnss.GNSSComponent(
                shift=offset + rstate.normal(scale=0.01), sigma=0.005
            ),
        )
        for i in range(n)
    ]
    campaign = gnss.GNSSCampaign(name=name, stations=stations)
    return heart.GNSSCompoundComponent.from_pyrocko_gnss_campaign(campaign)


@pytest.fixture
def gnss_files(tmp_path):
    from synthetic import write_gnss_ascii

    datasets = {}
    for name, offset in [("campaign_a", 0.0), ("campaign_b", 0.1)]:
        datasets[name] = make_gnss_datasets(name=name, offset=offset)
        write_gnss_ascii(datasets[name], str(tmp_path / name))

    return tmp_path, datasets


def test_from_pyrocko_gnss_campaign():
    datasets = make_gnss_datasets(n=5)
    assert [d.component for d in datasets] == ["north", "east", "up"]
    for dataset in datasets:
        assert dataset.typ == "GNSS"
        assert dataset.samples == 5
        assert dataset.odw.shape == (5,)
        assert dataset.covariance.data.shape == (5, 5)

    assert_allclose(datasets[0].displacement, 0.01 * num.arange(5))
    assert_allclose(num.diag(datasets[1].covariance.data), 0.003**2)

    los = datasets[2].update_los_vector()
    assert_allclose(los, num.tile([0.0, 0.0, 1.0], (5, 1)))


def test_load_ascii_gnss_globk(gnss_files):
    datadir, datasets = gnss_files
    campaign = inputf.load_ascii_gnss_globk(
        str(datadir), "campaign_a", components=["north", "east", "up"]
    )
    assert campaign.name == "campaign_a"
    assert [s.code for s in campaign.stations] == ["G%02i" % i for i in range(5)]

    for comp, dataset in zip(["north", "east", "up"], datasets["campaign_a"]):
        shifts = [getattr(s, comp).shift for s in campaign.stations]
        sigmas = [getattr(s, comp).sigma for s in campaign.stations]
        assert_allclose(shifts, dataset.displacement, atol=1e-12)
        assert_allclose(num.square(sigmas), num.diag(dataset.covariance.data))


def test_load_ascii_gnss_globk_missing(tmp_path):
    assert inputf.load_ascii_gnss_globk(str(tmp_path), "no_extension") is None
    with pytest.raises(ImportError):
        inputf.load_ascii_gnss_globk(str(tmp_path), "missing.txt")


def test_gnss_config_loads_all_files(gnss_files):
    datadir, datasets = gnss_files
    config = bconfig.GNSSDatasetConfig(
        datadir=str(datadir), names=["campaign_a", "campaign_b"], blacklist=[]
    )
    loaded = config.load_data()
    assert [d.id for d in loaded] == [
        "%s_%s" % (name, comp)
        for name in ["campaign_a", "campaign_b"]
        for comp in ["north", "east", "up"]
    ]
    for dataset, original in zip(
        loaded, datasets["campaign_a"] + datasets["campaign_b"]
    ):
        assert_allclose(dataset.displacement, original.displacement, atol=1e-12)

    campaigns = config.load_data(campaign=True)
    assert [c.name for c in campaigns] == ["campaign_a", "campaign_b"]


def test_gnss_blacklist(gnss_files):
    datadir, _ = gnss_files
    config = bconfig.GNSSDatasetConfig(
        datadir=str(datadir), names=["campaign_a"], blacklist=["G01", "G03"]
    )
    for dataset in config.load_data():
        assert dataset.samples == 3
        assert [s.code for s in dataset.stations] == ["G00", "G02", "G04"]


def test_setup_stations():
    stations = inputf.setup_stations(
        lats=[10.5, 10.0],
        lons=[20.0, 20.5],
        names=["N", "E"],
        networks=["XX", "XX"],
        event=EVENT,
    )
    assert [s.station for s in stations] == ["N", "E"]
    for station in stations:
        assert sorted(c.name for c in station.channels) == ["R", "T", "Z"]

    assert_allclose(stations[0].azimuth, 0.0, atol=0.1)
    assert_allclose(stations[1].azimuth, 90.0, atol=0.5)


@pytest.fixture
def mseed_data(tmp_path):
    stations = [
        model.Station(
            network="XX",
            station="S%i" % i,
            lat=lat,
            lon=lon,
            channels=[
                model.Channel("E", azimuth=90.0, dip=0.0),
                model.Channel("N", azimuth=0.0, dip=0.0),
                model.Channel("Z", azimuth=0.0, dip=-90.0),
            ],
        )
        for i, (lat, lon) in enumerate([(10.3, 20.0), (10.0, 20.3), (9.8, 19.8)])
    ]
    rstate = num.random.RandomState(0)
    traces = []
    for station in stations:
        for channel in "ENZ":
            tr = trace.Trace(
                network=station.network,
                station=station.station,
                channel=channel,
                tmin=putil.str_to_time("2020-01-01 00:00:00"),
                deltat=0.5,
                ydata=rstate.normal(size=200),
            )
            traces.append(tr)
            filename = ".".join((station.network, station.station, "", channel))
            io.save(tr, str(tmp_path / filename), format="mseed")

    return tmp_path, stations, traces


def test_load_data_traces(mseed_data):
    datadir, stations, traces = mseed_data
    loaded = inputf.load_data_traces(
        str(datadir), stations, divider=".", data_format="mseed"
    )
    assert len(loaded) == len(traces)
    for tr, original in zip(loaded, traces):
        assert isinstance(tr, heart.SeismicDataset)
        assert tr.location == "0"
        assert tr.station == original.station
        assert tr.channel == original.channel
        assert_allclose(tr.ydata, original.ydata)


def test_load_data_traces_missing_files(mseed_data):
    datadir, stations, _ = mseed_data
    os.remove(datadir / "XX.S0..E")
    loaded = inputf.load_data_traces(
        str(datadir), stations, divider=".", data_format="mseed"
    )
    assert len(loaded) == 8
    assert ("XX", "S0", "0", "E") not in [tr.nslc_id for tr in loaded]


def test_rotate_traces_and_stations(mseed_data):
    datadir, stations, traces = mseed_data
    data = inputf.load_data_traces(
        str(datadir), stations, divider=".", data_format="mseed"
    )
    rotated = inputf.rotate_traces_and_stations(data, stations, EVENT)
    assert len(rotated) == len(data)

    for station in stations:
        assert {"R", "T", "Z"} <= {c.name for c in station.channels}
        components = {
            tr.channel: tr.ydata for tr in data if tr.station == station.station
        }
        rtz = {tr.channel: tr.ydata for tr in rotated if tr.station == station.station}

        # R points away from the event (at the station), T is R rotated
        # clockwise by 90 deg
        azi = num.deg2rad(station.backazimuth + 180.0)
        n, e = components["N"], components["E"]
        assert_allclose(rtz["R"], num.cos(azi) * n + num.sin(azi) * e, atol=1e-10)
        assert_allclose(rtz["T"], -num.sin(azi) * n + num.cos(azi) * e, atol=1e-10)
        assert_allclose(rtz["Z"], components["Z"], atol=1e-10)


def test_load_kite_scene_conventions(tmp_path):
    """
    BEAT incidence, heading and LOS vectors of an imported kite scene have to
    be consistent with the LOS convention of kite.
    """
    pytest.importorskip("kite")
    from kite import Scene
    from kite.scene import FrameConfig, SceneConfig

    ny, nx, spacing = 40, 50, 500.0
    theta = num.deg2rad(num.linspace(30.0, 60.0, nx))[num.newaxis, :].repeat(ny, 0)
    phi = num.full((ny, nx), num.deg2rad(-100.0))
    y, x = num.mgrid[:ny, :nx]
    displacement = 0.01 * num.sin(x / 10.0) * num.cos(y / 7.0)

    config = SceneConfig()
    config.frame = FrameConfig(
        llLat=10.0, llLon=20.0, dE=spacing, dN=spacing, spacing="meter"
    )
    scene = Scene(displacement=displacement, theta=theta, phi=phi, config=config)
    scene.quadtree.epsilon = 0.001
    scene.quadtree.tile_size_min = 2 * spacing
    scene.quadtree.tile_size_max = 10 * spacing
    npixels = num.array([leaf.npixel for leaf in scene.quadtree.leaves])
    scene.covariance.config.covariance_matrix = num.diag(1e-6 / npixels)
    scene.save(str(tmp_path / "scene"))

    datasets = inputf.load_kite_scenes(str(tmp_path), ["scene", "not_existing"])
    assert len(datasets) == 1
    dataset = datasets[0]
    quadtree = Scene.load(str(tmp_path / "scene")).quadtree

    assert dataset.name == "scene"
    assert dataset.samples == quadtree.nleaves
    assert_allclose(dataset.displacement, quadtree.leaf_means)
    assert_allclose(num.diag(dataset.covariance.data), 1e-6 / npixels)

    assert_allclose(dataset.incidence, 90.0 - num.rad2deg(quadtree.leaf_thetas))
    assert_allclose(dataset.heading, 180.0 - num.rad2deg(quadtree.leaf_phis))

    # kite: los = sin(theta) * up + cos(theta) cos(phi) * east
    #             + cos(theta) sin(phi) * north
    thetas, phis = quadtree.leaf_thetas, quadtree.leaf_phis
    kite_los = num.vstack(
        [
            num.cos(thetas) * num.sin(phis),
            num.cos(thetas) * num.cos(phis),
            num.sin(thetas),
        ]
    ).T
    assert_allclose(dataset.update_los_vector(), kite_los, atol=1e-12)

    # leaf positions, see test_kite_scene_leaf_positions for the exact check
    north, east = dataset.update_local_coords(
        model.Event(lat=scene.frame.llLat, lon=scene.frame.llLon)
    )
    assert_allclose(north, quadtree.leaf_northings, rtol=0.01, atol=1.0)
    assert_allclose(east, quadtree.leaf_eastings, rtol=0.01, atol=1.0)


@pytest.mark.xfail(
    strict=True,
    reason="pyrocko orthodrome.ne_to_latlon (used by DiffIFG.from_kite_scene) and "
    "latlon_to_ne_numpy (used by update_local_coords) are not inverse, "
    "distances differ by 0.5 % at 10 deg latitude",
)
def test_kite_scene_leaf_positions():
    from pyrocko import orthodrome

    lat0, lon0 = 10.0, 20.0
    norths = num.array([1000.0, 10000.0, 20000.0])
    easts = num.array([2000.0, 5000.0, 20000.0])
    lats, lons = orthodrome.ne_to_latlon(lat0, lon0, norths, easts)
    dataset = heart.DiffIFG(name="positions", lats=lats, lons=lons)
    north, east = dataset.update_local_coords(model.Event(lat=lat0, lon=lon0))
    assert_allclose(north, norths, atol=1.0)
    assert_allclose(east, easts, atol=1.0)
