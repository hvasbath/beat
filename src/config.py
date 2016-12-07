"""
The config module contains the classes to build the configuration files that
are being read by the beat executable.

So far there are configuration files for the three main optimization problems
implemented. Solving the fault geometry, the static distributed slip and the
kinematic distributed slip.
"""
import logging
import os

from pyrocko.guts import Object, List, String, Float, Int, Tuple, Bool, Dict
from pyrocko.guts import load, dump
from pyrocko.cake import load_model

from pyrocko import trace, model, util
from pyrocko.gf import Earthmodel1D
from beat.heart import Filter, ArrivalTaper, TeleseismicTarget, Parameter

from beat import utility

import numpy as num

guts_prefix = 'beat'

logger = logging.getLogger('config')

modes = ['geometry', 'static_dist', 'kinematic_dist']

geo_vars_geometry = ['east_shift', 'north_shift', 'depth', 'strike', 'dip',
                         'rake', 'length', 'width', 'slip']
geo_vars_magma = geo_vars_geometry + ['opening']

seis_vars_geometry = ['time', 'duration']

joint_vars_geometry = geo_vars_geometry + seis_vars_geometry

static_dist_vars = ['Uparr', 'Uperp']
partial_kinematic_vars = ['nuc_x', 'nuc_y', 'duration', 'velocity']

kinematic_dist_vars = static_dist_vars + partial_kinematic_vars

hyper_pars = {'Z': 'seis_Z', 'T': 'seis_T',
             'SAR': 'geo_S', 'GPS': 'geo_G'}

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
    seis_Z=(-20., 20.),
    seis_T=(-20., 20.),
    geo_S=(-20., 20.),
    geo_G=(-20., 20.))

seismic_data_name = 'seismic_data.pkl'
geodetic_data_name = 'geodetic_data.pkl'

sample_p_outname = 'sample.params'

km = 1000.


class GFConfig(Object):
    """
    Config for GreensFunction calculation parameters.
    """
    store_superdir = String.T(default='./')
    earth_model = String.T(default='ak135-f-average.m',
                           help='Name of the reference earthmodel, see '
                                'pyrocko.cake.builtin_models() for '
                                'alternatives.')
    n_variations = Int.T(default=0,
                         help='Times to vary input velocity model.')
    use_crust2 = Bool.T(
        default=True,
        help='Flag, for replacing the crust from the earthmodel'
             'with crust from the crust2 model.')
    replace_water = Bool.T(default=True,
                        help='Flag, for replacing water layers in the crust2'
                             'model.')
    custom_velocity_model = Earthmodel1D.T(
        default=None,
        optional=True,
        help='Custom Earthmodel, in case crust2 and standard model not'
             ' wanted. Needs to be a :py::class:cake.LayeredModel')

    source_depth_min = Float.T(default=0.,
                               help='Minimum depth [km] for GF function grid.')
    source_depth_max = Float.T(default=10.,
                               help='Maximum depth [km] for GF function grid.')
    source_depth_spacing = Float.T(default=1.,
                               help='Depth spacing [km] for GF function grid.')

    nworkers = Int.T(
        default=1,
        help='Number of processors to use for calculating the GFs')


class SeismicGFConfig(GFConfig):
    """
    Seismic GF parameters for Layered Halfspace.
    """
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
    source_distance_radius = Float.T(
        default=20.,
        help='Radius of distance grid [km] for GF function grid around '
             'reference event.')
    source_distance_spacing = Float.T(
        default=1.,
        help='Distance spacing [km] for GF function grid.')


class GeodeticGFConfig(GFConfig):
    """
    Geodetic GF parameters for Layered Halfspace.
    """
    code = String.T(default='psgrn',
                    help='Modeling code to use. (psgrn, ... others need to be'
                         'implemented!)')
    sampling_interval = Float.T(\
        default=1.0,
        help='Distance dependend sampling spacing coefficient.'
             '1. - equidistant')
    source_distance_min = Float.T(
        default=0.,
        help='Minimum distance [km] for GF function grid.')
    source_distance_max = Float.T(
        default=100.,
        help='Maximum distance [km] for GF function grid.')
    source_distance_spacing = Float.T(
        default=1.,
        help='Distance spacing [km] for GF function grid.')


class SeismicConfig(Object):
    """
    Config for seismic data optimization related parameters.
    """

    datadir = String.T(default='./')
    blacklist = List.T(String.T(),
                       default=['placeholder'],
                       help='Station name for station to be thrown out.')
    distances = Tuple.T(2, Float.T(), default=(30., 90.))
    channels = List.T(String.T(), default=['Z', 'T'])
    calc_data_cov = Bool.T(
        default=True,
        help='Flag for calculating the data covariance matrix based on the'
             ' pre P arrival data trace noise.')
    arrival_taper = trace.Taper.T(
                default=ArrivalTaper.D(),
                help='Taper a,b/c,d time [s] before/after wave arrival')
    filterer = Filter.T(default=Filter.D())
    targets = List.T(TeleseismicTarget.T(), optional=True)
    gf_config = SeismicGFConfig.T(default=SeismicGFConfig.D())


class GeodeticConfig(Object):
    """
    Config for geodetic data optimization related parameters.
    """

    datadir = String.T(default='./')
    tracks = List.T(String.T(), default=['Data prefix filenames here ...'])
    types = List.T(
        default=['SAR'],
        help='Types of geodetic data, i.e. SAR, GPS, ...')
    gf_config = GeodeticGFConfig.T(default=GeodeticGFConfig.D())


class ProblemConfig(Object):
    """
    Config for inversion problem to setup.
    """
    mode = String.T(default='geometry',
                    help='Problem to solve: "geometry", "static","kinematic"')
    n_faults = Int.T(default=1,
                     help='Number of Sub-faults to solve for')
    datasets = List.T(default=['geodetic'])
    hyperparameters = Dict.T(
        help='Hyperparameters to weight different types of datasets.')
    priors = List.T(Parameter.T())

    def init_vars(self):

        variables = self.select_variables()

        self.priors = []
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

    def select_variables(self):
        """
        Return model variables depending on problem config.
        """

        if self.mode not in modes:
            raise ValueError('Problem mode %s not implemented' % self.mode)

        if self.mode == 'geometry':
            if 'geodetic' in self.datasets:
                variables = geo_vars_geometry
            if 'seismic' in self.datasets:
                variables = joint_vars_geometry

        elif self.mode == 'static':
            variables = static_dist_vars

        elif self.mode == 'kinematic':
            variables = kinematic_dist_vars
            if 'seismic' not in self.datasets:
                logger.error('A kinematic model cannot be resolved with'
                             'geodetic data only.')
                raise Exception('Kinematic model not resolvable with only'
                                'geodetic data!')

        return variables

    def validate_priors(self):
        """
        Check if priors and their test values do not contradict!
        """
        for param in self.priors:
            param()

        logger.info('All parameter-priors ok!')

    def validate_hypers(self):
        """
        Check if hyperparameters and their test values do not contradict!
        """
        if self.hyperparameters is not None:
            for hp in self.hyperparameters.itervalues():
                hp()

            logger.info('All hyper-parameters ok!')

        else:
            logger.info('No hyper-parameters defined!')


class SamplerParameters(Object):
    pass


class MetropolisConfig(SamplerParameters):
    """
    Config for optimization parameters of the Adaptive Metropolis algorithm.
    """
    n_jobs = Int.T(
        default=1,
        help='Number of processors to use, i.e. chains to sample in parallel.')
    n_stages = Int.T(
        default=10,
        help='Number of stages to sample/ or points in solution spacce for'
             ' hyperparameter estimation')
    n_steps = Int.T(default=25000,
                    help='Number of steps for the MC chain.')
    tune_interval = Int.T(
        default=50,
        help='Tune interval for adaptive tuning of Metropolis step size.')
    proposal_dist = String.T(
        default='Normal',
        help='Normal Proposal distribution, for Metropolis steps;'
             'Alternatives: Cauchy, Laplace, Poisson, MultivariateNormal')
    update_covariances = Bool.T(
        default=False,
        help='Update model prediction covariance matrixes in transition '
             'stages.')
    thin = Int.T(
        default=2,
        help='Thinning parameter of the sampled trace. Every "thin"th sample'
             ' is taken.')
    burn = Float.T(
        default=0.5,
        help='Burn-in parameter between 0. and 1. to discard fraction of'
             ' samples from the beginning of the chain.')
    rm_flag = Bool.T(default=False,
                     help='Remove existing stage results prior to sampling.')


class ATMCMCConfig(SamplerParameters):
    """
    Config for optimization parameters of the ATMCMC algorithm.
    """
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
    stage = String.T(default='0',
                  help='Stage where to start/continue the sampling. Has to'
                       ' be int or "final"')
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


class SamplerConfig(Object):
    """
    Config for the sampler specific parameters.
    """

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
    """
    BEATconfig is the overarching configuration class, providing all the
    sub-configurations classes for the problem setup, Greens Function
    generation, optimization algorithm and the data being used.
    """

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
    hyper_sampler_config = SamplerConfig.T(default=SamplerConfig.D())

    def update_hypers(self):
        """
        Evaluate the whole config and initialise necessary hyperparameters.
        """

        hypernames = []

        if self.geodetic_config is not None:
            for ty in self.geodetic_config.types:
                hypernames.append(hyper_pars[ty])

        if self.seismic_config is not None:
            for ch in self.seismic_config.channels:
                hypernames.append(hyper_pars[ch])

        hypers = dict()
        for name in hypernames:
            hypers[name] = Parameter(
                    name=name,
                    lower=num.ones(1, dtype=num.float) * \
                        default_bounds[name][0],
                    upper=num.ones(1, dtype=num.float) * \
                        default_bounds[name][1],
                    testvalue=num.ones(1, dtype=num.float) * \
                        num.mean(default_bounds[name])
                                        )

        self.problem_config.hyperparameters = hypers
        self.problem_config.validate_hypers()


def init_config(name, date, min_magnitude=6.0, main_path='./',
                datasets=['geodetic'],
                mode='geometry', n_faults=1,
                sampler='ATMCMC', hyper_sampler='Metropolis',
                use_custom=False):
    """
    Initialise BEATconfig File and write it main_path/name .
    Fine parameters have to be edited in the config file .yaml manually.

    Parameters
    ----------
    name : str
        Name of the event
    date : str
        'YYYY-MM-DD', date of the event
    min_magnitude : scalar, float
        approximate minimum Mw of the event
    datasets : List of strings
        data sets to include in the optimization: either 'geodetic' and/or
        'seismic'
    mode : str
        type of optimization problem: 'Geometry' / 'Static'/ 'Kinematic'
    n_faults : int
        number of faults to solve for / discretize depending on mode parameter
    sampler : str
        Optimization algorithm to use to sample the solution space
        Options: 'ATMCMC', 'Metropolis'
    use_custom : boolean
        Flag to setup manually a custom velocity model.

    Returns
    -------
    :class:`BEATconfig`
    """

    c = BEATconfig(name=name, date=date)

    c.event = utility.search_catalog(date=date, min_magnitude=min_magnitude)

    c.project_dir = os.path.join(os.path.abspath(main_path), name)

    c.problem_config = ProblemConfig(
        n_faults=n_faults, datasets=datasets, mode=mode)
    c.problem_config.init_vars()

    if 'geodetic' in datasets:
        c.geodetic_config = GeodeticConfig()
        if use_custom:
            logger.info('use_custom flag set! The velocity model in the'
                        ' geodetic GF configuration has to be updated!')
            c.geodetic_config.gf_config.custom_velocity_model = \
                load_model().extract(depth_max=100. * km)
            c.geodetic_config.gf_config.use_crust2 = False
            c.geodetic_config.gf_config.replace_water = False
    else:
        c.geodetic_config = None

    if 'seismic' in datasets:
        c.seismic_config = SeismicConfig()
        if use_custom:
            logger.info('use_custom flag set! The velocity model in the'
                        ' seismic GF configuration has to be updated!')
            c.seismic_config.gf_config.custom_velocity_model = \
                load_model().extract(depth_max=100. * km)
            c.seismic_config.gf_config.use_crust2 = False
            c.seismic_config.gf_config.replace_water = False
    else:
        c.seismic_config = None

    c.sampler_config = SamplerConfig(name=sampler)
    c.sampler_config.set_parameters()

    c.sampler_config = SamplerConfig(name=hyper_sampler)
    c.sampler_config.set_parameters()

    c.update_hypers()

    c.problem_config.validate_priors()

    c.validate()

    logger.info('Project_directory: %s \n' % c.project_dir)
    util.ensuredir(c.project_dir)

    config_file_name = 'config_' + mode + '.yaml'
    conf_out = os.path.join(c.project_dir, config_file_name)
    dump(c, filename=conf_out)
    return c


def load_config(project_dir, mode):
    """
    Load configuration file.

    Parameters
    ----------
    project_dir : str
        path to the directory of the configuration file
    mode : str
        type of optimization problem: 'Geometry' / 'Static'/ 'Kinematic'

    Returns
    -------
    :class:`BEATconfig`
    """
    config_file_name = 'config_' + mode + '.yaml'

    config_fn = os.path.join(project_dir, config_file_name)
    config = load(filename=config_fn)

    if config.problem_config.hyperparameters is None or \
        len(config.problem_config.hyperparameters) == 0:
        config.update_hypers()
        logger.info('Updated hyper parameters!')
        dump(config, filename=config_fn)

    return config
