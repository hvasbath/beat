from scipy.io import loadmat
import math
import numpy as num

from os.path import join as pjoin
from beat import info
from beat.heart import DiffIFG, init_geodetic_targets
from beat.ffi import optimize_discretization, discretize_sources
from beat.sources import RectangularSource

from pyrocko import orthodrome as otd
from pyrocko import model, util
from pyrocko.gf.seismosizer import LocalEngine


from beat.config import ResolutionDiscretizationConfig

util.setup_logging('R-based subsampling', 'info')

real = True    # if use real data

km = 1000.
nworkers = 4

n_pix = 635 # 545, 193, 635

store_superdirs = '/home/vasyurhm/GF/Marmara'
varnames = ['uparr']

event = model.Event(
  lat=40.896,
  lon=28.86,
  time=util.str_to_time('2019-10-12 00:00:00'),
  depth=15000.0,
  name='marm',
  magnitude=7.0)

testdata_path = pjoin(info.project_root, 'data/test/InputData.mat')
d = loadmat(testdata_path)
source_params = d['pm']

## optimize discretization
# load data and setup
if real:
    data_xloc = d['X'][0:n_pix]
    x_shift = data_xloc.min()
    data_xloc -= x_shift

    data_yloc = d['Y'][0:n_pix]
    y_shift = data_yloc.min()
    data_yloc -= y_shift

    sab_los = d['LOS'][0:n_pix, :]
    los = num.zeros_like(sab_los)
    los[:, 0] = sab_los[:, 1]
    los[:, 1] = sab_los[:, 0]
    los[:, 2] = sab_los[:, 2]

    ## init fault geometry
    n_sources = source_params.shape[1]
    sources = []
    for sps in range(n_sources):
        Length, Width, Depth, Dip, Strike, Xloc, Yloc, strsl, dipsl, _ = source_params[
                                                                         :, sps]
        print(Xloc, Yloc)
        lat, lon = otd.ne_to_latlon(event.lat, event.lon, (Yloc - y_shift) * km,
                                    (Xloc - x_shift) * km)
        rake = math.atan2(dipsl, strsl)
        print('d,s,r', dipsl, strsl, rake)
        slip = math.sqrt(strsl ** 2 + dipsl ** 2)
        print('lat,lon', lat, lon)
        rf = RectangularSource(
            lat=lat,
            lon=lon,
            east_shift=0.,
            north_shift=0.,
            depth=Depth * km,
            length=Length * km,
            width=Width * km,
            dip=Dip + 180.,  # no negative dip!
            strike=Strike,
            rake=rake,
            slip=slip
        )
        print(rf)
        sources.append(rf)

    epsilon = 0.05  # Damping constant for SVD: sabrnas 0.005 (without squaring -here squaring)
    R_thresh = 0.95  # Resolution threshhold (patches above R_thresh will be further subdivided)
    d_par = 5  # Depth penalty, a higher number penalized deeper patches more
    alphaprcnt = 0.2

    config = ResolutionDiscretizationConfig(
        epsilon=epsilon,
        resolution_thresh=R_thresh,
        depth_penalty=d_par,
        alpha=alphaprcnt,
        patch_widths_min=[1., 1., 1.],
        patch_widths_max=[30., 30., 30.],
        patch_lengths_min=[1., 1., 1.],
        patch_lengths_max=[30., 30., 30.],
        extension_lengths=[0., 0., 0.],
        extension_widths=[0., 0., 0.],
    )
else:
    yvec = num.linspace(-15., 15., 30)
    xvec = num.linspace(-20., 20., 40)
    y_shift = x_shift = 0.
    X, Y = num.meshgrid(xvec, yvec)
    data_xloc = X.ravel()
    data_yloc = Y.ravel()
    los = num.ones((data_xloc.size, 3)) * num.array([ -0.1009988, -0.52730111, 0.84365442])

    epsilon = 0.07  # Damping constant for SVD: sabrnas 0.005 (without squaring -here squaring)
    R_thresh = 0.95  # Resolution threshhold (patches above R_thresh will be further subdivided)
    d_par = 5  # Depth penalty, a higher number penalized deeper patches more
    alphaprcnt = 0.15

    config = ResolutionDiscretizationConfig(
        epsilon=epsilon,
        resolution_thresh=R_thresh,
        depth_penalty=d_par,
        alpha=alphaprcnt,
        patch_widths_min=[1.],
        patch_widths_max=[15.],
        patch_lengths_min=[1.],
        patch_lengths_max=[30.],
        extension_lengths=[0.],
        extension_widths=[0.],
    )

    ## init fault geometry
    n_sources = 1
    sources = []
    for sps in range(n_sources):
        Length, Width, Depth, Dip, Strike, Xloc, Yloc, strsl, dipsl, _ = source_params[:, sps]
        print(Xloc, Yloc)
        lat, lon = otd.ne_to_latlon(event.lat, event.lon, 0. * km, 0. * km)
        rake = math.atan2(dipsl, strsl)
        print('d,s,r', dipsl, strsl, rake)
        slip = math.sqrt(strsl**2 + dipsl**2)
        print('lat,lon', lat, lon)
        rf = RectangularSource(
            lat=lat,
            lon=lon,
            east_shift=0.,
            north_shift=0.,
            depth=0. * km,
            length=30. * km,
            width=18. * km,
            dip=90.,             # no negative dip!
            strike=90.,
            rake=0.,
            slip=1.
        )
        print(rf)
        sources.append(rf)

lats, lons = otd.ne_to_latlon(event.lat, event.lon, data_yloc * km, data_xloc * km)

datasets = [DiffIFG(
    east_shifts=num.zeros_like(data_yloc).ravel(),
    north_shifts=num.zeros_like(data_yloc).ravel(),
    odw=num.ones_like(data_yloc).ravel(),
    lats=lats.ravel(),
    lons=lons.ravel(),
    los_vector=los,
    displacement=num.zeros_like(data_yloc).ravel())]






fault = discretize_sources(
    config, sources=sources, datatypes=['geodetic'], varnames=varnames,
    tolerance=0.5)


engine = LocalEngine(store_superdirs=[store_superdirs])

targets = init_geodetic_targets(
        datasets, earth_model_name='ak135-f-continental.m',
        interpolation='multilinear', crust_inds=[0],
        sample_rate=0.0)

print(event)
opt_fault, R = optimize_discretization(
    config, fault,
    datasets=datasets,
    varnames=varnames,
    crust_ind=0,
    engine=engine,
    targets=targets,
    event=event,
    force=True,
    nworkers=nworkers,
    plot=True,
    debug=False)
