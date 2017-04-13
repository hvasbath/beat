import scipy.io
import numpy as num

from beat import heart, utility
from pyrocko import model, io

import os
import logging

logger = logging.getLogger('beat')

km = 1000.
m = 0.000000001


def load_SAR_data(datadir, tracks):
    """
    Load SAR data in given directory and tracks.
    Returns Diff_IFG objects.
    """
    DIFFGs = []

    for k in tracks:
        # open matlab.mat files
        data = scipy.io.loadmat(datadir + 'quad_' + k + '.mat',
                                squeeze_me=True,
                                struct_as_record=False)
        covs = scipy.io.loadmat(datadir + 'CovMatrix_' + k + '.mat',
                                squeeze_me=True,
                                struct_as_record=False)

        utmx = data['cfoc'][:, 0]
        utmy = data['cfoc'][:, 1]
        lons, lats = utility.utm_to_lonlat(utmx, utmy, 36)
        Lv = data['lvQT']
        covariance = heart.Covariance(data=covs['Cov'])

        DIFFGs.append(heart.DiffIFG(
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

    return DIFFGs


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
    names = num.loadtxt(filepath, usecols=0, dtype='string')
    d = num.loadtxt(filepath, usecols=range(1, 9), dtype='float')

    if names.size != d.shape[0]:
        raise Exception('Number of stations and available data differs!')

    data = heart.GPSDataset()
    for i, name in enumerate(names):
        gps_station = heart.GPSStation(lon=d[i, 1], lat=d[i, 2])
        for j, comp in enumerate('ENU'):
            gps_station.add_component(
                heart.GPSComponent(
                    name=comp,
                    v=d[i, j + 3] / km,
                    sigma=d[i, j + 6] / km))
        data.add_station(gps_station)

    return data


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
        else:
            raise Exception('No data for this channel!')

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
                    data_trace = io.load(tracepath, data_format)[0]
                    # [nm] convert to m
                    data_trace.set_ydata(data_trace.ydata * m)
                    data_trcs.append(data_trace)
            except IOError:
                logger.warn('Unable to open file: ' + trace_name)

    return data_trcs
