import heart
import theanof
from pyrocko import model
import utility
import inputf
from matplotlib import pylab
import numpy as num
from theano.compile import ProfileStats
from theano import shared, function
import pscmp

profile = ProfileStats()

km = 1000.

#homedir = '/Users/vasyurhm/Aqaba1995'
#datadir = homedir + '/data/'
#storehomedir = [homedir + '/GF/']
#geo_datadir = '/Users/vasyurhm/SAR_data/Aqaba/'

homedir = '/data3TB/Teleseism'
datadir = homedir + '/autokiwi/events/Aqaba1995/kiwi/data/'
storehomedir = [homedir + '/Greensfunctions/Aqaba1995GFS/']
geo_datadir = '/data/SAR_data/Aqaba1995/subsampled/'

event = model.load_one_event(datadir + 'event.txt')

# load geodetic data
tracks = ['A_T114do', 'A_T114up', 'A_T343co', 'A_T343up', 'D_T254co', 'D_T350co']
DiffIFGs = inputf.load_SAR_data(geo_datadir, tracks)
image = 2
utmx = []
utmy = []
for ifg in DiffIFGs:
    utmx.append(ifg.utme)
    utmy.append(ifg.utmn)

lons, lats = utility.utm_to_lonlat(num.hstack(utmx), num.hstack(utmy), 36)


def plot(lons, lats, disp):
    # plot
    #colim = num.max([disp.max(), num.abs(disp.min())])
    ax = pylab.axes()
    im = ax.scatter(lons, lats, 15, disp, edgecolors='none')
    pylab.colorbar(im)
    pylab.title(tracks[image])
    pylab.show()


class Test_Pscmp(object):
    crust_ind = 0
    event = event
    store_superdir = storehomedir[0]
    lengths = 25.
    widths = 15.
    slips = 5.
    openings = 0.
    sources = [pscmp.PsCmpRectangularSource()]

    def calc_gf(self):
        heart.geo_construct_gf(self.event, self.superdir,
                     source_distance_min=0., source_distance_max=100.,
                     source_depth_min=0., source_depth_max=50.,
                     source_spacing=0.5, earth_model='ak135-f-average.m',
                     crust_ind=self.crust_ind, execute=True)

    def standard_pscmp(self):
        # source params
        o_lons = self.event.lon
        o_lats = self.event.lat
        depths = self.event.depth / km
        east_shift = 50. * km
        north_shift = 40. * km
        strikes = event.moment_tensor.strike2
        dips = self.event.moment_tensor.dip2
        rakes = self.event.moment_tensor.rake2
        lengths = self.lengths
        widths = self.widths
        slips = self.slips
        openings = self.openings

        self.sources[0].update(lon=o_lons, lat=o_lats, east_shift=east_shift,
                          north_shift=north_shift, depth=depths,
                          strike=strikes, dip=dips, rake=rakes,
                          length=lengths, width=widths, slip=slips,
                          opening=openings)

        displ = heart.geo_layer_synthetics(
                    self.store_superdir,
                    self.crust_ind,
                    lons, lats, self.sources)
        print self.sources[0].__dict__
        return displ[0]

    def sym_pscmp_op(self):
        LONS = shared(lons)
        LATS = shared(lats)

        # source params
        ess = shared(num.array([50.], dtype=num.float64))
        nss = shared(num.array([40.], dtype=num.float64))
        ds = shared(num.array([self.event.depth / km], dtype=num.float64))
        strikes = shared(num.array([self.event.moment_tensor.strike2],
                                    dtype=num.float64))
        dips = shared(num.array([self.event.moment_tensor.dip2],
                                 dtype=num.float64))
        rakes = shared(num.array([self.event.moment_tensor.rake2],
                                  dtype=num.float64))
        ls = shared(num.array([self.lengths], dtype=num.float64))
        ws = shared(num.array([self.widths], dtype=num.float64))
        slips = shared(num.array([self.slips], dtype=num.float64))
        opns = shared(num.array([self.openings], dtype=num.float64))

        var_list_free = [LONS, LATS, ess, nss, ds, strikes, dips, rakes, ls,
                    ws, slips, opns]
        var_list_static = [ess, nss, ds, strikes, dips, rakes, ls,
                    ws, slips, opns]

        get_displacements_free = theanof.GeoLayerSynthesizerFree(
             self.store_superdir, self.crust_ind, self.sources)
        get_displacements_static = theanof.GeoLayerSynthesizerStatic(
             lats=lats, lons=lons, store_superdir=self.store_superdir,
             crust_ind=self.crust_ind, sources=self.sources)

        displf = get_displacements_free(*var_list_free)
        displs = get_displacements_static(*var_list_static)

        sym_forward_op_f = function([], [displf], profile=profile)
        sym_forward_op_s = function([], [displs], profile=profile)

        return sym_forward_op_f, sym_forward_op_s

