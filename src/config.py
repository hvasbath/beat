import logging
import os

from pyrocko.guts import Object, List, String, Float, Int, Tuple, Bool
from pyrocko.guts import load, dump
from pyrocko.cake import load_model

from pyrocko import trace, model, util
from pyrocko.gf import Earthmodel1D
from beat.heart import Filter, ArrivalTaper, TeleseismicTarget, Parameter

from beat import utility

import numpy as num

guts_prefix = 'beat'

logger = logging.getLogger('config')

geo_vars_geometry = ['east_shift', 'north_shift', 'depth', 'strike', 'dip',
                         'rake', 'length', 'width', 'slip']
geo_vars_magma = geo_vars_geometry + ['opening']

seis_vars_geometry = ['time', 'duration']

joint_vars_geometry = geo_vars_geometry + seis_vars_geometry

static_dist_vars = ['Uparr', 'Uperp']
partial_kinematic_vars = ['nuc_x', 'nuc_y', 'duration', 'velocity']

kinematic_dist_vars = static_dist_vars + partial_kinematic_vars

hyper_pars = ['alpha', 'beta', 'gamma']

default_bounds = dict(
    east_shift=(-10., 10.),
    north_shift=(-10., 10.),
    depth=(0., 5.),
    strike=(0, 180.),
    dip=(45., 90.),
    rake=(-90., 90.),
    length=(5., 30.),
    width=(5., 20.),
    slip=(0.1, 8.),
    time=(-3., 3.),
    duration=(0., 20.),
    Uparr=(-0.3, 6.),
    Uperp=(-0.3, 4.),
    nuc_x=(0., 10.),
    nuc_y=(0., 7.),
    velocity=(0.5, 4.2),
    alpha=(1e-20, 1e20),
    beta=(1e-20, 1e20),
    gamma=(1e-20, 1e20))

seismic_data_name = 'seismic_data.pkl'
geodetic_data_name = 'geodetic_data.pkl'

km = 1000.


class GFConfig(Object):
    '''
    Config for GreensFunction calculation parameters.
    '''
    store_superdir = String.T(default='./')
    earth_model = String.T(default='ak135-f-average.m',
                           help='Name of the reference earthmodel, see '
                                'pyrocko.cake.builtin_models() for '
                                'alternatives.')
    use_crust2 = Bool.T(
        default=True,
        help='Flag, for replacing the crust from the earthmodel'
             'with crust from the crust2 model.')
    replace_water = Bool.T(default=True,
                        help='Flag, for replacing water layers in the crust2'
                             'model.')
    crust_inds = List.T(default=range(10),
                       help='List of indexes for different velocity models.'
                            ' 0 is reference model.')

    source_depth_min = Float.T(default=0.,
                               help='Minimum depth [km] for GF function grid.')
    source_depth_max = Float.T(default=10.,
                               help='Maximum depth [km] for GF function grid.')
    source_depth_spacing = Float.T(default=1.,
                               help='Depth spacing [km] for GF function grid.')

    source_distance_min = Float.T(
        default=0.,
        help='Minimum distance [km] for GF function grid.')
    source_distance_max = Float.T(
        default=100.,
        help='Maximum distance [km] for GF function grid.')
    source_distance_spacing = Float.T(
        default=1.,
        help='Distance spacing [km] for GF function grid.')

    execute = Bool.T(default=False,
                     help='Flag, for starting the modeling code after config'
                          'creation.')
    nworkers = Int.T(
        default=1,
        help='Number of processors to use for calculating the GFs')


class SeismicGFConfig(GFConfig):
    '''
    Seismic GF parameters for Layered Halfspace.
    '''
    code = String.T(default='qssp',
                  help='Modeling code to use. (qssp, qseis, comming soon: '
                       'qseis2d)')
    sample_rate = Float.T(default=2.,
                          help='Sample rate for the Greens Functions.')
    depth_limit_variation = Float.T(
        default=600.,
        help='Depth limit [km] for varying the velocity model.')
    rm_gfs = Bool.T(default=True,
                    help='Flag for removing modeling module GF files after'
                         ' completion.')


class GeodeticGFConfig(GFConfig):
    '''
    Geodetic GF parameters for Layered Halfspace.
    '''
    code = String.T(default='psgrn',
                    help='Modeling code to use. (psgrn, ... others need to be'
                         'implemented!)')
    sampling_interval = Float.T(\
        default=1.0,
        help='Distance dependend sampling spacing coefficient.'
             '1. - equidistant')
    custom_velocity_model = Earthmodel1D.T(
        default=None,
        optional=True,
        help='Custom Earthmodel, in case crust2 and standard model not'
             ' wanted. Needs to be a :py::class:cake.LayeredModel')


class SeismicConfig(Object):
    '''
    Config for teleseismic setup related parameters.
    '''

    datadir = String.T(default='./')
    blacklist = List.T(String.T(),
                       default=['placeholder'],
                       help='Station name for station to be thrown out.')
    distances = Tuple.T(2, Float.T(), default=(30., 90.))
    channels = List.T(String.T(), default=['Z', 'T'])

    arrival_taper = trace.Taper.T(
                default=ArrivalTaper.D(),
                help='Taper a,b/c,d time [s] before/after wave arrival')
    filterer = Filter.T(default=Filter.D())
    targets = List.T(TeleseismicTarget.T(), optional=True)
    gf_config = SeismicGFConfig.T(default=SeismicGFConfig.D())


class GeodeticConfig(Object):
    '''
    Config for geodetic setup related parameters.
    '''

    datadir = String.T(default='./')
    tracks = List.T(String.T(), default=['Data filenames here ...'])
    targets = List.T(optional=True)
    gf_config = GeodeticGFConfig.T(default=GeodeticGFConfig.D())


class ProblemConfig(Object):
    '''
    Config for inversion problem to setup.
    '''
    mode = String.T(default='geometry',
                    help='Problem to solve: "Geometry", "Static","Kinematic"')
    n_faults = Int.T(default=1,
                     help='Number of Sub-faults to solve for')
    datasets = List.T(default=['geodetic'])
    hyperparameters = List.T(
        Parameter.T(),
        optional=True,
        help='Hyperparameters to weight different types of datasets.')
    priors = List.T(Parameter.T())

    def init_vars(self):

        if self.mode == 'geometry':
            if 'geodetic' in self.datasets:
                variables = geo_vars_geometry
            if 'seismic' in self.datasets:
                variables = joint_vars_geometry

        elif self.mode == 'static_dist':
            variables = static_dist_vars

        elif self.mode == 'kinematic_dist':
            variables = kinematic_dist_vars
            if 'seismic' not in self.datasets:
                logger.error('A kinematic model cannot be resolved with'
                             'geodetic data only.')
                raise Exception('Kinematic model not resolvable with only'
                                'geodetic data!')

        for variable in variables:
            self.priors.append(
                Parameter(
                    name=variable,
                    lower=num.ones(self.n_faults, dtype=num.float) * \
                        default_bounds[variable][0],
                    upper=num.ones(self.n_faults, dtype=num.float) * \
                        default_bounds[variable][1],
                    testvalue=num.ones(self.n_faults, dtype=num.float) * \
                        num.mean(default_bounds[variable]))
                               )

        for i in range(len(self.datasets) - 1):
            self.hyperparameters.append(
                Parameter(
                    name=hyper_pars[i],
                    lower=num.ones(1, dtype=num.float) * \
                        default_bounds[hyper_pars[i]][0],
                    upper=num.ones(1, dtype=num.float) * \
                        default_bounds[hyper_pars[i]][1],
                    testvalue=num.ones(1, dtype=num.float) * \
                        num.mean(default_bounds[hyper_pars[i]]))
                                        )

    def validate_priors(self):
        '''
        Check if priors and their test values do not contradict!
        '''
        for param in self.priors:
            param()

        for hyperparam in self.hyperparameters:
            hyperparam()

        print('All parameter-priors ok!')


class SamplerParameters(Object):
    pass


class MetropolisConfig(SamplerParameters):
    '''
    Config for optimization parameters for the adaptive Metropolis algorithm.
    '''
    n_steps = Int.T(default=10000,
                    help='Number of steps for the MC chain.')
    tune_interval = Int.T(
        default=10,
        help='Tune interval for adaptive tuning of Metropolis step size.')
    proposal_dist = String.T(
        default='Normal',
        help='Normal Proposal distribution, for Metropolis steps'
             'alternatives: Cauchy, Laplace, Poisson, MultivariateNormal')


class ATMCMCConfig(SamplerParameters):
    '''
    Config for optimization parameters for the ATMCMC algorithm.
    '''
    n_chains = Int.T(default=1000,
                     help='Number of Metropolis chains for sampling.')
    n_steps = Int.T(default=100,
                    help='Number of steps for each chain per stage.')
    n_jobs = Int.T(
        default=1,
        help='Number of processors to use, i.e. chains to sample in parallel.')
    tune_interval = Int.T(
        default=10,
        help='Tune interval for adaptive tuning of Metropolis step size.')
    coef_variation = Float.T(
        default=1.,
        help='Coefficient of variation, determines the similarity of the'
             'intermediate stage pdfs;'
             'low - small beta steps (slow cooling),'
             'high - wide beta steps (fast cooling)')
    stage = Int.T(default=0,
                  help='Stage where to start/continue the sampling.')
    proposal_dist = String.T(
        default='MvNPd',
        help='Multivariate Normal Proposal distribution, for Metropolis steps'
             'alternatives need to be implemented')
    check_bnd = Bool.T(
        default=True,
        help='Flag for checking whether proposed step lies within'
             ' variable bounds.')
    update_covariances = Bool.T(
        default=False,
        help='Update model prediction covariance matrixes in transition '
             'stages.')
    rm_flag = Bool.T(default=False,
                     help='Remove existing stage results prior to sampling.')
    plot_flag = Bool.T(default=True,
                       help='Plot stage posteriors after sampling of'
                            'each stage')


class SamplerConfig(Object):
    '''
    Contains the sampler specific parameters.
    '''

    name = String.T(default='ATMCMC',
                    help='Sampler to use for sampling the solution space.'
                         'Metropolis/ ATMCMC coming soon: ADVI')
    parameters = SamplerParameters.T(
        default=ATMCMCConfig.D(),
        optional=True,
        help='Sampler dependend Parameters')

    def set_parameters(self):

        if self.name == None:
            logger.info('Sampler not defined, using default sampler: ATMCMC')
            self.name = 'ATMCMC'

        if self.name == 'Metropolis':
            self.parameters = MetropolisConfig()

        if self.name == 'ATMCMC':
            self.parameters = ATMCMCConfig()


class BEATconfig(Object):
    '''
    BEATconfig class is the overarching class, providing all the configurations
    for seismic data and geodetic data being used. Define directory structure
    here for Greens functions geodetic and seismic.
    '''

    name = String.T()
    date = String.T()
    event = model.Event.T(optional=True)
    project_dir = String.T(default='event/')

    problem_config = ProblemConfig.T(default=ProblemConfig.D())
    geodetic_config = GeodeticConfig.T(
        default=None, optional=True)
    seismic_config = SeismicConfig.T(
        default=None, optional=True)
    sampler_config = SamplerConfig.T(default=SamplerConfig.D())


def init_config(name, date, min_magnitude=6.0, main_path='./',
                datasets=['geodetic'],
                n_variations=0, mode='geometry', n_faults=1,
                sampler='ATMCMC', use_custom=False):
    '''
    Initialise BEATconfig File and write it to main_path/name+year/ .
    Fine parameters have to be edited in the config file .yaml manually.
    Input:
    name - Str - Name of the event
    date - Str - 'YYYY-MM-DD', date of the event
    min_magnitude - approximate minimum Mw of the event
    '''

    c = BEATconfig(name=name, date=date)

    c.event = utility.search_catalog(date=date, min_magnitude=min_magnitude)

    c.project_dir = os.path.join(os.path.abspath(main_path), name)

    c.problem_config = ProblemConfig(
        n_faults=n_faults, datasets=datasets, mode=mode)
    c.problem_config.init_vars()

    if 'geodetic' in datasets:
        c.geodetic_config = GeodeticConfig()
        c.geodetic_config.gf_config.crust_inds = range(1 + n_variations)
        if use_custom:
            logger.info('use_custom flag set, the velocity model in the'
                        ' configuration file has to be updated!')
            c.geodetic_config.gf_config.custom_velocity_model = \
                load_model().extract(depth_max=100. * km)
            c.geodetic_config.gf_config.use_crust2 = False
            c.geodetic_config.gf_config.replace_water = False
    else:
        c.geodetic_config = None

    if 'seismic' in datasets:
        c.seismic_config = SeismicConfig()
        c.seismic_config.gf_config.crust_inds = range(1 + n_variations)
    else:
        c.seismic_config = None

    c.sampler_config = SamplerConfig(name=sampler)
    c.sampler_config.set_parameters()

    c.validate()
    c.problem_config.validate_priors()

    logger.info('Project_directory: %s \n' % c.project_dir)
    util.ensuredir(c.project_dir)

    config_file_name = 'config_' + mode + '.yaml'
    conf_out = os.path.join(c.project_dir, config_file_name)
    dump(c, filename=conf_out)
    return c


def load_config(project_dir, mode):
    config_file_name = 'config_' + mode + '.yaml'

    config_fn = os.path.join(project_dir, config_file_name)
    config = load(filename=config_fn)

    return config
