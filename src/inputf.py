import scipy.io
import heart
import utility
from pyrocko import gf, model, io
import numpy as num

km = 1000.
nm = 0.000000001

def load_SAR_data(datadir,tracks):
    '''
    Load SAR data in given directory and tracks.
    Returns Diff_IFG objects.
    '''
    DIFFGs = []
    
    for k in tracks:
        # open matlab.mat files
        data = scipy.io.loadmat(datadir + 'quad_' + k + '.mat',
                                squeeze_me = True,
                                struct_as_record=False)
        covs = scipy.io.loadmat(datadir + 'CovMatrix_' + k + '.mat',
                                squeeze_me = True,
                                struct_as_record=False)

        utmx=data['cfoc'][:,0]
        utmy=data['cfoc'][:,1]
        lons, lats = utility.utm_to_lonlat(utmx, utmy, 36)
        Lv = data['lvQT']
        covariance = heart.Covariance(data=covs['Cov'], icov=covs['InvCov'])

        DIFFGs.append(heart.Diff_IFG(displacement=data['sqval'],
                 utme=utmx,
                 utmn=utmy,
                 lons=lons,
                 lats=lats,
                 covariance=covariance,
                 incidence=Lv.inci,
                 heading=Lv.head))

    return DIFFGs

def load_seism_data(datadir):
    '''Load stations and event files in datadir and read traces from autokiwi autput.'''
    trc_name_divider = '-'
    data_format = 'mseed'
    # load stations and event
    stations = model.load_stations(datadir + 'stations.txt')
    event = model.load_one_event(datadir + 'event.txt')
    blacklist = [28, 27]  # delete stations DRLN FRB (NIL and ARU) are in list will be discarded during data loading--> reverse order!!!
    for sta in blacklist:
        popped_sta = stations.pop(sta)
        print(' remove stations from list: ' + popped_sta.station)

    nrstat = len(stations)

    # load recorded data 
    data_trcs = []
    drop_stat = []
    for sta_num in range(len(stations)):
        for ref_channel in ['r','u']:       #(r)ight transverse, (a)way radial, vertical (u)p
            trace_name = trc_name_divider.join(('reference', stations[sta_num].network, stations[sta_num].station, ref_channel))
            tracepath = datadir + trace_name + '.' + data_format
            try:
                with open(tracepath) as file:
                    data_trace = io.load(tracepath, data_format)
                    data_trace[0].set_ydata(data_trace[0].ydata*nm) #[nm] convert to m
                    data_trcs.append(data_trace[0])
            except IOError as e:
                print('Unable to open file: ' + trace_name) #Does not exist 
                drop_stat.append(sta_num)
                break

    drop_stat = num.unique(drop_stat) # remove double station indexes
    drop_stat = drop_stat[::-1]   # start with the highest (reverse matrix)
    for sta in drop_stat:
        popped_sta = stations.pop(sta)
        print(' remove stations from list: ' + popped_sta.station)

    nrstat = len(stations)
    print 'Number of stations', str(nrstat)
    targets = [
        # have to put stations here
        gf.Target(
            quantity='displacement',
            codes=(stations[sta_num].network,
                   stations[sta_num].station,
                   stations[sta_num].location, channel), #network, statio, location, channel
            lat=stations[sta_num].lat,
            lon=stations[sta_num].lon,
            azimuth=stations[sta_num].get_channel(channel).azimuth,
            dip=stations[sta_num].get_channel(channel).dip,
            store_id='crust2_%s' % stations[sta_num].station)
        # include several channel
        for sta_num in range(0,nrstat)
            for channel in ['T','Z']]       # T for SH waves, Z for P waves

    return stations, targets, event, data_trcs

