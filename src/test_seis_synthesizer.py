import heart
import theanof
import inputf
from pyrocko import gf
import utility
import numpy as num
from theano import shared, function
from theano.compile import ProfileStats

profile = ProfileStats()

km = 1000.

homedir = '/Users/vasyurhm/Aqaba1995'
datadir = homedir + '/data/'
storehomedir = [homedir + '/GF/']


[stations, targets, event, data_traces] = inputf.load_seism_data(datadir)

engine = gf.LocalEngine(store_superdirs=storehomedir)

sources = [
    gf.RectangularSource(
        lat=29.124977942689519,
        lon=34.871469702014863,
        width=24 * km,
        length=58 * km,
        time=817013725.5571846,
        depth=12 * km,
        strike=206.3701904106799,
        dip=73.0785305323845,
        rake=-8.135103051434966,
        magnitude=7.01,
        stf=gf.HalfSinusoidSTF(duration=3., anchor=-1.))]
    
    ## gf.RectangularSource(
    ##     lat=29.0,
    ##     lon=35.0,
    ##     width=4 * km,
    ##     length=5 * km,
    ##     time=817013720.5571846,
    ##     depth=5 * km,
    ##     strike=180.0,
    ##     dip=70.,
    ##     rake=-7.,
    ##     magnitude=4.01,
    ##     stf=gf.HalfSinusoidSTF(duration=2., anchor=-1.))]


class Test_seis_synthesizer(object):

    engine = engine
    targets = heart.init_targets([stations[9]])
    sources = sources
    event = event
    arrival_taper = heart.ArrivalTaper(a=20, b=10, c=35, d=50)
    filterer = heart.Filter(lower_corner=0.001, upper_corner=0.5, order=3)
    widths = 24. * km
    lengths = 58. * km
    times = 5.0
    slips = 6.

    def get_synthetics_standard(self, plot):
        synths, tmins = heart.seis_synthetics(self.engine, self.sources,
                                     self.targets,
                                     self.arrival_taper, self.filterer,
                                     plot=plot)
        cut_data = heart.taper_filter_traces(data_traces[18:20],
                                             self.arrival_taper,
                                             self.filterer, tmins, plot=plot)
        return synths, cut_data, tmins

    def get_synthetics_symbolic(self):
        var_dict = {
        'o_lons': shared(num.array([self.event.lon], dtype=num.float64)),
        'o_lats': shared(num.array([self.event.lat], dtype=num.float64)),
        'ds': shared(num.array([self.event.depth], dtype=num.float64)),
        'strikes': shared(num.array([self.event.moment_tensor.strike2],
                                    dtype=num.float64)),
        'dips': shared(num.array([self.event.moment_tensor.dip2],
                                 dtype=num.float64)),
        'rakes': shared(num.array([self.event.moment_tensor.rake2],
                                  dtype=num.float64)),
        'ls': shared(num.array([self.lengths], dtype=num.float64)),
        'ws': shared(num.array([self.widths], dtype=num.float64)),
        'slips': shared(num.array([self.slips], dtype=num.float64)),
        'times': shared(num.array([self.times], dtype=num.float64))}
            
        o_lons = shared(num.array([self.event.lon], dtype=num.float64))
        o_lats = shared(num.array([self.event.lat], dtype=num.float64))
        ds = shared(num.array([self.event.depth], dtype=num.float64))
        strikes = shared(num.array([self.event.moment_tensor.strike2],
                                    dtype=num.float64))
        dips = shared(num.array([self.event.moment_tensor.dip2],
                                 dtype=num.float64))
        rakes = shared(num.array([self.event.moment_tensor.rake2],
                                  dtype=num.float64))
        ls = shared(num.array([self.lengths], dtype=num.float64))
        ws = shared(num.array([self.widths], dtype=num.float64))
        slips = shared(num.array([self.slips], dtype=num.float64))
        times = shared(num.array([self.times], dtype=num.float64))

        var_list = [o_lons,o_lats,ds,strikes, dips, rakes, ls,ws,slips, times]
        get_synths = theanof.SeisSynthesizer(
                        engine=self.engine, sources=self.sources,
                        targets=self.targets, event=self.event,
                        arrival_taper=self.arrival_taper,
                        filterer=self.filterer)
        #synths = get_synths(o_lons, o_lats, ds, strikes, dips, rakes,
        #                                            ls, ws, slips, times)
        synths, tmins = get_synths(*var_list)
        chop_traces = theanof.SeisDataChopper(
                            sample_rate=1./data_traces[0].deltat,
                            traces=data_traces[18:20],
                            arrival_taper=self.arrival_taper,
                            filterer=self.filterer)
        cut_trcs = chop_traces(tmins)
        sym_forward_op = function([], [synths, tmins], profile=profile)
        data_cut_op = function([], [cut_trcs], profile=profile)
        return sym_forward_op, data_cut_op


#o_lons = shared(num.array([event.lon], dtype=num.float64))
#o_lats = shared(num.array([event.lat], dtype=num.float64))
#ds = shared(num.array([event.depth / km], dtype=num.float64))
#strikes = shared(num.array([event.moment_tensor.strike2],
#                            dtype=num.float64))
#dips = shared(num.array([event.moment_tensor.dip2],
#                         dtype=num.float64))
#rakes = shared(num.array([event.moment_tensor.rake2],
#                          dtype=num.float64))
#ls = shared(num.array([lengths], dtype=num.float64))
#ws = shared(num.array([widths], dtype=num.float64))
#slips = shared(num.array([slips], dtype=num.float64))
#times = shared(num.array([times], dtype=num.float64))

