import os
from beat import heart, utility
from beat.models import GeometryOptimizer

from pyrocko.guts import load
from pyrocko import model, util

name = 'Aqaba'
year = 1995

config_file_name = 'config.yaml'
project_dir = '/data3TB/' + name + str(year) + 'wcov'
store_superdir = '/data3TB/Teleseism/Greensfunctions/Aqaba1995GFS/'
seismic_datadir = '/data3TB/Teleseism/autokiwi/events/Aqaba1995/kiwi/data/'
geodetic_datadir = '/data/SAR_data/Aqaba1995/subsampled/'

util.ensuredir(project_dir)

tracks = ['A_T114do', 'A_T114up', 'A_T343co',
          'A_T343up', 'D_T254co', 'D_T350co']

blacklist = ['DRLN', 'FRB', 'NIL', 'ARU']
distances = (26.5, 91.0)
n_variations = 20
sample_rate = 1.0
channels = ['Z', 'T']
filterer = heart.Filter()
arrival_taper = heart.ArrivalTaper()
logger = utility.setup_logging(project_dir)


def init():
    logger.info('Welcome to BEAT the Bayesian Earthqake Analysis Tool')
    logger.info('Author: Hannes Vasyura-Bathke')
    logger.info('Email: Hannes.Vasyura-Bathke@kaust.edu.sa')
    logger.info('Version: ultra pre-alpha')
    logger.info('\n')
    logger.info('... Creating config.yaml ...')
    config = heart.init_nonlin(name, year,
        project_dir=project_dir,
        store_superdir=store_superdir,
        sample_rate=sample_rate,
        n_variations=n_variations,
        channels=channels,
        distances=distances,
        geodetic_datadir=geodetic_datadir,
        seismic_datadir=seismic_datadir,
        tracks=tracks,
        blacklist=blacklist,
        arrival_taper=arrival_taper,
        filterer=filterer)
    return config


def build_geo_gfs():

    config_fn = os.path.join(project_dir, config_file_name)
    config = load(filename=config_fn)

    n_mods = len(config.crust_inds)

    logger.info('... Building geodetic Greens Functions ...\n')
    logger.info(' %i varied crustal velocity models!'
                'Building stores in: %s \n' % (n_mods, config.store_superdir))

    eventname = os.path.join(config.seismic_datadir, 'event.txt')
    event = model.load_one_event(eventname)

    for crust_ind in config.crust_inds:
        heart.geo_construct_gf(event, store_superdir,
             source_distance_min=0., source_distance_max=100.,
             source_depth_min=0., source_depth_max=50.,
             source_distance_spacing=10., source_depth_spacing=0.5, 
             earth_model='ak135-f-average.m',
             crust_ind=crust_ind, execute=True)
        logger.info('Done building model %i / %i \n' % (crust_ind + 1, n_mods))


def check_model_setup():
    config_fn = os.path.join(project_dir, config_file_name)
    config = load(filename=config_fn)

    problem = GeometryOptimizer(config)

    problem.built_model()
    test_logp = problem.model.logpt.tag.test_value
    logger.info('The likelihood of the test_model is %f' % float(test_logp))
    problem.init_atmip(n_chains=400, tune_interval=10)
    #return problem
    mtrace = problem.sample(n_steps=50)
    return mtrace

if __name__ == '__main__':
#    config = init()
#    build_geo_gfs()
    check_model_setup()
