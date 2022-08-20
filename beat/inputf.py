import copy
import logging
import os
from glob import glob

import numpy as num
import scipy.io
from pyrocko import io, model

from beat import heart, utility

logger = logging.getLogger("inputf")

km = 1000.0
m = 0.000000001


def setup_stations(lats, lons, names, networks, event, rotate=True):
    """
    Setup station objects, based on station coordinates and reference event.

    Parameters
    ----------
    lats : :class:`num.ndarray`
        of station location latitude
    lons : :class:`num.ndarray`
        of station location longitude
    names : list
        of strings of station names
    networks : list
        of strings of network names for each station
    event : :class:`pyrocko.model.Event`

    Results
    -------
    stations : list
        of :class:`pyrocko.model.Station`
    """

    stations = []
    for lat, lon, name, network in zip(lats, lons, names, networks):
        s = model.Station(lat=lat, lon=lon, station=name, network=network)
        s.set_event_relative_data(event)
        s.set_channels_by_name("E", "N", "Z")
        if rotate:
            p = s.guess_projections_to_rtu(out_channels=("R", "T", "Z"))
        s.set_channels(p[0][2])
        stations.append(s)

    return stations


def load_matfile(datapath, **kwargs):
    try:
        return scipy.io.loadmat(datapath, **kwargs)
    except IOError:
        logger.warn("File %s does not exist." % datapath)
        return None


def load_SAR_data(datadir, names):
    """
    Load SAR data in given directory and filenames.
    Returns Diff_IFG objects.
    """
    diffgs = []
    tobeloaded_names = set(copy.deepcopy(names))

    for k in names:
        # open matlab.mat files

        data = load_matfile(
            datadir + "quad_" + k + ".mat", squeeze_me=True, struct_as_record=False
        )

        covs = load_matfile(
            datadir + "CovMatrix_" + k + ".mat", squeeze_me=True, struct_as_record=False
        )

        if data is not None and covs is not None:
            utmx = data["cfoc"][:, 0]
            utmy = data["cfoc"][:, 1]
            lons, lats = utility.utm_to_lonlat(utmx, utmy, 36)
            Lv = data["lvQT"]
            covariance = heart.Covariance(data=covs["Cov"])

            diffgs.append(
                heart.DiffIFG(
                    name=k,
                    displacement=data["sqval"],
                    utme=utmx,
                    utmn=utmy,
                    lons=lons,
                    lats=lats,
                    covariance=covariance,
                    incidence=Lv.inci,
                    heading=Lv.head,
                    odw=data["ODW_sub"],
                )
            )
            tobeloaded_names.discard(k)

        else:
            logger.info("File %s was no SAR data?!" % datadir)

    names = list(tobeloaded_names)
    return diffgs


def load_kite_scenes(datadir, names):
    """
    Load SAR data from the kite format.
    """
    try:
        from kite import Scene
        from kite.scene import UserIOWarning
    except ImportError:
        raise ImportError("kite not installed! please checkout www.pyrocko.org!")

    diffgs = []
    for k in names:
        filepath = os.path.join(datadir, k)
        try:
            logger.info("Loading scene: %s" % k)
            sc = Scene.load(filepath)
            diffgs.append(heart.DiffIFG.from_kite_scene(sc))

            logger.info("Successfully imported kite scene %s" % k)
        except (ImportError, UserIOWarning):
            logger.warning("File %s not conform with kite format!" % k)

    return diffgs


def load_ascii_gnss_globk(filedir, filename, components=["east", "north", "up"]):
    """
    Load ascii file columns containing:
    station name, Lon, Lat, ve, vn, vu, sigma_ve, sigma_vn, sigma_vu
    location [decimal deg]
    measurement unit [mm/yr]

    Returns
    -------
    :class:`pyrocko.model.gnss.GNSSCampaign`
    """
    from pyrocko.model import gnss

    filepath = os.path.join(filedir, filename)
    skiprows = 3
    if os.path.exists(filepath):
        names = num.loadtxt(filepath, skiprows=skiprows, usecols=[12], dtype="str")
        d = num.loadtxt(filepath, skiprows=skiprows, usecols=range(12), dtype="float")
    elif len(os.path.splitext(filepath)[1]) == 0:
        logger.info("File %s is not an ascii text file!" % filepath)
        return
    else:
        raise ImportError("Did not find data under: %s" % filepath)

    component_to_idx_map = {"east": (2, 6), "north": (3, 7), "up": (9, 11)}

    # velocity_idxs = [2, 3, 9]
    # std_idxs = [6, 7, 11]

    if names.size != d.shape[0]:
        raise Exception("Number of stations and available data differs!")

    data = gnss.GNSSCampaign(name=filename)
    for i, name in enumerate(names):
        # code = str(name.split('_')[0])  # may produce duplicate sites
        code = str(name)
        gnss_station = gnss.GNSSStation(
            code=code, lon=float(d[i, 0]), lat=float(d[i, 1])
        )
        for j, comp in enumerate(components):
            vel_idx, std_idx = component_to_idx_map[comp]

            setattr(
                gnss_station,
                comp,
                gnss.GNSSComponent(
                    shift=float(d[i, vel_idx] / km), sigma=float(d[i, std_idx] / km)
                ),
            )
        logger.debug("Loaded station %s" % gnss_station.code)
        data.add_station(gnss_station)

    return data


def load_repsonses_from_file(projectpath):

    network = ""
    location = ""

    response_filename = os.path.join(projectpath, "responses.txt")
    logger.info("Loading responses from: %s", response_filename)

    responses = {}
    for line in open(response_filename, "r"):
        t = line.split()
        logger.info(t)

        if len(t) == 8:
            sta, cha, instrument, lat, lon, mag, damp, period = t
            # please see the file format below
            if damp == "No_damping":
                damp = 0.001

            lat, lon, mag, damp, period = [
                float(x) for x in (lat, lon, mag, damp, period)
            ]

            responses[(network, sta, location, cha)] = (mag, damp, period)
            logger.debug("%s %s %s %s %s" % (sta, cha, mag, damp, period))

    return responses


def load_and_blacklist_gnss(
    datadir, filename, blacklist, campaign=False, components=["north", "east", "up"]
):
    """
    Load ascii GNSS data from GLOBK, apply blacklist and initialise targets.

    Parameters
    ----------
    datadir: string
        of path to the directory
    filename: string
        filename to load
    blacklist: list
        of strings with station names to blacklist
    campaign : boolean
        if True return gnss.GNSSCampaign otherwise
        list of heart.GNSSCompoundComponent
    components : tuple
        of strings ('north', 'east', 'up') for displacement components to return
    """
    gnss_campaign = load_ascii_gnss_globk(datadir, filename, components)
    if gnss_campaign:
        for station_code in blacklist:
            logger.debug("Removing station %s for campaign." % station_code)
            gnss_campaign.remove_station(station_code)

        station_names = [station.code for station in gnss_campaign.stations]
        logger.info("Loaded data of %i GNSS stations" % len(station_names))
        logger.debug("Loaded GNSS stations %s" % utility.list2string(station_names))
        if not campaign:
            return heart.GNSSCompoundComponent.from_pyrocko_gnss_campaign(
                gnss_campaign, components=components
            )
        else:
            return gnss_campaign


def load_and_blacklist_stations(datadir, blacklist):
    """
    Load stations from autokiwi output and apply blacklist
    """

    stations = model.load_stations(datadir + "stations.txt")
    return utility.apply_station_blacklist(stations, blacklist)


def load_autokiwi(datadir, stations):
    return load_data_traces(
        datadir=datadir,
        stations=stations,
        divider="-",
        load_channels=["u", "r", "a"],
        name_prefix="reference",
        convert=True,
    )


channel_mappings = {"u": "Z", "r": "T", "a": "R", "BHE": "E", "BHN": "N", "BHZ": "Z"}


def load_obspy_data(datadir):
    """
    Load data from the directory through obspy and convert to pyrocko objects.

    Parameters
    ----------
    datadir : string
        absolute path to the data directory

    Returns
    -------
    data_traces, stations
    """

    import obspy
    from pyrocko import obspy_compat

    obspy_compat.plant()

    filenames = set(glob(datadir + "/*"))
    remaining_f = copy.deepcopy(filenames)
    print(filenames)
    stations = []
    for f in filenames:
        print(f)
        try:
            inv = obspy.read_inventory(f)
            stations.extend(inv.to_pyrocko_stations())
            remaining_f.discard(f)
        except TypeError:
            logger.debug("File %s not an inventory." % f)

    filenames = copy.deepcopy(remaining_f)
    print(filenames)
    data_traces = []
    for f in filenames:
        print(f)
        try:
            stream = obspy.read(f)
            pyrocko_traces = stream.to_pyrocko_traces()
            for tr in pyrocko_traces:
                data_traces.append(heart.SeismicDataset.from_pyrocko_trace(tr))

            remaining_f.discard(f)

        except TypeError:
            logger.debug("File %s not waveforms" % f)

    print(remaining_f)
    if len(remaining_f) > 0:
        logger.warning(
            "Could not import these files %s" % utility.list2string(list(filenames))
        )

    logger.info(
        "Imported %i data_traces and %i stations" % (len(stations), len(data_traces))
    )
    return stations, data_traces


def load_data_traces(
    datadir,
    stations,
    load_channels=[],
    name_prefix=None,
    name_suffix=None,
    data_format="mseed",
    divider="-",
    convert=False,
    no_network=False,
):
    """
    Load data traces for the given stations from datadir.
    """

    data_trcs = []
    # (r)ight transverse, (a)way radial, vertical (u)p
    for station in stations:
        if not load_channels:
            channels = station.channels
        else:
            channels = [model.Channel(name=cha) for cha in load_channels]

        for channel in channels:
            if no_network:
                trace_name = divider.join(
                    (station.station, station.location, channel.name)
                )
            else:
                trace_name = divider.join(
                    (station.network, station.station, station.location, channel.name)
                )

            if name_suffix:
                trace_name = divider.join((trace_name, name_suffix))

            if name_prefix:
                trace_name = divider.join((name_prefix, trace_name))

            tracepath = os.path.join(datadir, trace_name)
            try:
                with open(tracepath):
                    dt = io.load(tracepath, format=data_format)[0]
                    # [nm] convert to m
                    if convert:
                        dt.set_ydata(dt.ydata * m)

                    dt.set_channel(channel.name)
                    dt.set_station(station.station)
                    dt.set_network(station.network)
                    dt.set_location("0")
                    # convert to BEAT seismic Dataset
                    data_trcs.append(heart.SeismicDataset.from_pyrocko_trace(dt))
            except IOError:
                logger.warn("Unable to open file: " + trace_name)

    return data_trcs


supported_channels = list(channel_mappings.values())


def rotate_traces_and_stations(datatraces, stations, event):
    """
    Rotate traces and stations into RTZ with respect to the event.
    Updates channels of stations in place!

    Parameters
    ---------
    datatraces: list
        of :class:`pyrocko.trace.Trace`
    stations: list
        of :class:`pyrocko.model.Station`
    event: :class:`pyrocko.model.Event`

    Returns
    -------
    rotated traces to RTZ
    """
    from pyrocko import trace

    station2traces = utility.gather(datatraces, lambda t: t.station)

    trs_projected = []
    for station in stations:
        station.set_event_relative_data(event)
        projections = station.guess_projections_to_rtu(out_channels=("R", "T", "Z"))

        try:
            traces = station2traces[station.station]
        except (KeyError):
            logger.warning(
                'Did not find data traces for station "%s"' % stations.station
            )
            continue

        ntraces = len(traces)
        if ntraces < 3:
            logger.warn(
                "Only found %i component(s) for station %s" % (ntraces, station.station)
            )

        for matrix, in_channels, out_channels in projections:
            proc = trace.project(traces, matrix, in_channels, out_channels)
            for tr in proc:
                logger.debug("Outtrace: \n %s" % tr.__str__())
                for ch in out_channels:
                    if ch.name == tr.channel:
                        station.add_channel(ch)

            if proc:
                logger.debug("Updated station: \n %s" % station.__str__())
                trs_projected.extend(proc)

    return trs_projected
