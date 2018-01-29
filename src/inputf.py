import scipy.io
import numpy as num
import copy

from beat import heart, utility
from pyrocko import model, io

from glob import glob
import os
import logging

logger = logging.getLogger('inputf')

km = 1000.
m = 0.000000001


def load_obspy_data(datdir):
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

    filenames = set(glob(os.getcwd() + '/*'))

    stations = []
    for f in filenames:
        try:
            inv = obspy.read_inventory(f)
            stations.extend(inv.to_pyrocko_stations())
            filenames.discard(f)
        except TypeError:
            logger.debug('File %s not an inventory.' % f)

    data_traces = []
    for f in filenames:
        try:
            stream = obspy.read()
            pyrocko_traces = stream.to_pyrocko_traces()
            for tr in pyrocko_traces:
                data_traces.append(heart.SeismicDataset.from_pyrocko_trace(tr))

            filenames.discard(f)

        except TypeError:
            logger.debug('File %s not waveforms' % f)

    if len(filenames) > 0:
        logger.warning(
            'Could not import these files %s' %
            utility.list_to_str(list(filenames)))

    logger.info('Imported %i data_traces and %i stations' %
                (len(stations), len(data_traces)))
    return data_traces, stations


def setup_stations(lats, lons, names, networks, event):
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
        s = model.Station(
            lat=lat, lon=lon, station=name, network=network)
        s.set_event_relative_data(event)
        s.set_channels_by_name('E', 'N', 'Z')
        p = s.guess_projections_to_rtu(out_channels=('R', 'T', 'Z'))
        s.set_channels(p[0][2])
        stations.append(s)

    return stations


def load_matfile(datapath, **kwargs):
    try:
        return scipy.io.loadmat(datapath, **kwargs)
    except IOError:
        logger.warn('File %s does not exist.' % datapath)
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
            datadir + 'quad_' + k + '.mat',
            squeeze_me=True,
            struct_as_record=False)

        covs = load_matfile(
            datadir + 'CovMatrix_' + k + '.mat',
            squeeze_me=True,
            struct_as_record=False)

        if data is not None and covs is not None:
            utmx = data['cfoc'][:, 0]
            utmy = data['cfoc'][:, 1]
            lons, lats = utility.utm_to_lonlat(utmx, utmy, 36)
            Lv = data['lvQT']
            covariance = heart.Covariance(data=covs['Cov'])

            diffgs.append(heart.DiffIFG(
                name=k,
                displacement=data['sqval'],
                utme=utmx,
                utmn=utmy,
                lons=lons,
                lats=lats,
                covariance=covariance,
                incidence=Lv.inci,
                heading=Lv.head,
                odw=data['ODW_sub']))
            tobeloaded_names.discard(k)

        else:
            logger.info('File %s was no SAR data?!' % datadir)

    names = list(tobeloaded_names)
    return diffgs


def load_kite_scenes(datadir, names):
    """
    Load SAR data from the kite format.
    """
    try:
        from kite import Scene
    except ImportError:
        raise ImportError(
            'kite not installed! please checkout www.pyrocko.org!')

    diffgs = []
    tobeloaded_names = set(copy.deepcopy(names))
    for k in names:
        try:
            sc = Scene.load(os.path.join(datadir, k))
            diffgs.append(heart.DiffIFG.from_kite_scene(sc))
            tobeloaded_names.discard(k)
        except ImportError:
            logger.warning('File %s not conform with kite format!' % k)

    names = list(tobeloaded_names)
    return diffgs


def load_ascii_gps(filedir, filename):
    """
    Load ascii file columns containing:
    station name, Lon, Lat, ve, vn, vu, sigma_ve, sigma_vn, sigma_vu
    location [decimal deg]
    measurement unit [mm/yr]

    Returns
    -------
    :class:`heart.GPSDataset`
    """
    filepath = os.path.join(filedir, filename)
    names = num.loadtxt(filepath, usecols=[0], dtype='string')
    d = num.loadtxt(filepath, usecols=range(1, 9), dtype='float')

    if names.size != d.shape[0]:
        raise IOError('Number of stations and available data differs!')

    data = heart.GPSDataset()
    for i, name in enumerate(names):

        gps_station = heart.GPSStation(
            name=str(name), lon=float(d[i, 0]), lat=float(d[i, 1]))
        for j, comp in enumerate('ENU'):

            gps_station.add_component(
                heart.GPSComponent(
                    name=comp,
                    v=float(d[i, j + 2] / km),
                    sigma=float(d[i, j + 5] / km)))
        data.add_station(gps_station)

    return data


def load_and_blacklist_GPS(datadir, filename, blacklist):
    """
    Load ascii GPS data, apply blacklist and initialise targets.
    """
    gps_ds = load_ascii_gps(datadir, filename)
    gps_ds.remove_stations(blacklist)
    comps = gps_ds.get_component_names()

    targets = []
    for c in comps:
        targets.append(gps_ds.get_compound(c))

    return targets


def load_and_blacklist_stations(datadir, blacklist):
    '''
    Load stations from autokiwi output and apply blacklist
    '''

    stations = model.load_stations(datadir + 'stations.txt')
    return utility.apply_station_blacklist(stations, blacklist)


def load_data_traces(datadir, stations, channels):
    '''
    Load data traces for the given stations and channels.
    '''
    trc_name_divider = '-'
    data_format = 'mseed'

    ref_channels = []
    for cha in channels:
        if cha == 'Z':
            ref_channels.append('u')
        elif cha == 'T':
            ref_channels.append('r')
        elif cha == 'R':
            ref_channels.append('a')
        else:
            raise IOError('No data for this channel!')

    # load recorded data
    data_trcs = []

    # (r)ight transverse, (a)way radial, vertical (u)p
    for ref_channel in ref_channels:
        for station in stations:
            trace_name = trc_name_divider.join(
                ('reference', station.network, station.station, ref_channel))

            tracepath = datadir + trace_name + '.' + data_format

            try:
                with open(tracepath):
                    dt = io.load(tracepath, data_format)[0]
                    # [nm] convert to m
                    dt.set_ydata(dt.ydata * m)
                    dt.station = station.station
                    dt.network = station.network
                    dt.location = '0'
                    # convert to BEAT seismic Dataset
                    data_trcs.append(
                        heart.SeismicDataset.from_pyrocko_trace(dt))
            except IOError:
                logger.warn('Unable to open file: ' + trace_name)

    return data_trcs
