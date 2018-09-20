"""
The config module contains the classes to build the configuration files that
are being read by the beat executable.

So far there are configuration files for the three main optimization problems
implemented. Solving the fault geometry, the static distributed slip and the
kinematic distributed slip.
"""
import logging
import os

from collections import OrderedDict

from pyrocko.guts import Object, List, String, Float, Int, Tuple, Bool, Dict
from pyrocko.guts import load, dump, StringChoice
from pyrocko.cake import load_model

from pyrocko import trace, model, util, gf
from pyrocko.gf import RectangularSource as PyrockoRS
from pyrocko.gf.seismosizer import Cloneable, stf_classes

from beat.heart import Filter, ArrivalTaper, Parameter
from beat.heart import ReferenceLocation
from beat.sources import RectangularSource, MTSourceWithMagnitude, MTQTSource

from beat import utility

import numpy as num

from theano import config as tconfig


guts_prefix = 'beat'

logger = logging.getLogger('config')

block_vars = [
    'bl_azimuth', 'bl_amplitude',
    'nucleation_strike', 'nucleation_dip', 'nucleation_time']
seis_vars = ['time', 'duration']

source_names = '''
    ExplosionSource
    RectangularExplosionSource
    DCSource
    CLVDSource
    MTSource
    MTQTSource
    RectangularSource
    DoubleDCSource
    RingfaultSource
    '''.split()

source_classes = [
    gf.ExplosionSource,
    gf.RectangularExplosionSource,
    gf.DCSource,
    gf.CLVDSource,
    MTSourceWithMagnitude,
    MTQTSource,
    PyrockoRS,
    gf.DoubleDCSource,
    gf.RingfaultSource]

stf_names = '''
Boxcar
Triangular
HalfSinusoid
'''.split()

source_catalog = {name: source_class for name, source_class in zip(
    source_names, source_classes)}

stf_catalog = {name: stf_class for name, stf_class in zip(
    stf_names, stf_classes[1:4])}

interseismic_vars = [
    'east_shift', 'north_shift', 'strike', 'dip', 'length',
    'locking_depth'] + block_vars

static_dist_vars = ['uparr', 'uperp']

hypo_vars = ['nucleation_strike', 'nucleation_dip', 'nucleation_time']
partial_kinematic_vars = ['durations', 'velocities']
voronoi_locations = ['voronoi_strike', 'voronoi_dip']

kinematic_dist_vars = static_dist_vars + partial_kinematic_vars + hypo_vars
transd_vars_dist = partial_kinematic_vars + static_dist_vars + \
    voronoi_locations

interseismic_catalog = {
    'geodetic': interseismic_vars}

geometry_catalog = {
    'geodetic': source_catalog,
    'seismic': source_catalog}

ffi_catalog = {
    'geodetic': static_dist_vars,
    'seismic': kinematic_dist_vars}

modes_catalog = OrderedDict([
    ['geometry', geometry_catalog],
    ['ffi', ffi_catalog],
    ['interseismic', interseismic_catalog]])

hyper_name_laplacian = 'h_laplacian'

moffdiag = (-1., 1.)
mdiag = (-num.sqrt(2), num.sqrt(2))

default_bounds = dict(
    east_shift=(-10., 10.),
    north_shift=(-10., 10.),
    depth=(0., 5.),

    strike=(0, 180.),
    strike1=(0, 180.),
    strike2=(0, 180.),
    dip=(45., 90.),
    dip1=(45., 90.),
    dip2=(45., 90.),
    rake=(-90., 90.),
    rake1=(-90., 90.),
    rake2=(-90., 90.),

    length=(5., 30.),
    width=(5., 20.),
    slip=(0.1, 8.),

    magnitude=(4., 7.),
    mnn=mdiag,
    mee=mdiag,
    mdd=mdiag,
    mne=moffdiag,
    mnd=moffdiag,
    med=moffdiag,

    u=(0., 3. / 4. * num.pi),
    v=(-1. / 3, 1. / 3.),
    kappa=(0., 2 * num.pi),
    sigma=(-num.pi / 2., num.pi / 2.),
    h=(0., 1.),

    volume_change=(1e8, 1e10),
    diameter=(5., 10.),
    mix=(0, 1),
    time=(-3., 3.),
    time_shift=(-5., 5.),

    delta_time=(0., 10.),
    delta_depth=(0., 10.),
    distance=(0., 10.),

    duration=(1., 30.),
    peak_ratio=(0., 1.),

    durations=(0.5, 29.5),
    uparr=(-0.05, 6.),
    uperp=(-0.3, 4.),
    nucleation_strike=(0., 10.),
    nucleation_dip=(0., 7.),
    nucleation_time=(-5., 5.),
    velocities=(0.5, 4.2),

    azimuth=(0, 180),
    amplitude=(1e10, 1e20),
    bl_azimuth=(0, 180),
    bl_amplitude=(0., 0.1),
    locking_depth=(1., 10.),

    hypers=(-20., 20.),
    ramp=(-0.005, 0.005))

default_seis_std = 1.e-6
default_geo_std = 1.e-3

default_decimation_factors = {
    'geodetic': 7,
    'seismic': 20}

seismic_data_name = 'seismic_data.pkl'
geodetic_data_name = 'geodetic_data.pkl'

linear_gf_dir_name = 'linear_gfs'
results_dir_name = 'results'
fault_geometry_name = 'fault_geometry.pkl'
geodetic_linear_gf_name = 'linear_geodetic_gfs.pkl'

sample_p_outname = 'sample.params'

summary_name = 'summary.txt'

km = 1000.


class GFConfig(Object):
    """
    Base config for GreensFunction calculation parameters.
    """
    store_superdir = String.T(
        default='./',
        help='Absolute path to the directory where Greens Function'
             ' stores are located')
    reference_model_idx = Int.T(
        default=0,
        help='Index to velocity model to use for the optimization.'
             ' 0 - reference, 1..n - model of variations')
    n_variations = Tuple.T(
        2,
        Int.T(),
        default=(0, 1),
        help='Start and end index to vary input velocity model. '
             'Important for the calculation of the model prediction covariance'
             ' matrix with respect to uncertainties in the velocity model.')
    earth_model_name = String.T(
        default='ak135-f-average.m',
        help='Name of the reference earthmodel, see '
             'pyrocko.cake.builtin_models() for alternatives.')
    nworkers = Int.T(
        default=1,
        help='Number of processors to use for calculating the GFs')


class NonlinearGFConfig(GFConfig):
    """
    Config for non-linear GreensFunction calculation parameters.
    Defines how the grid of Green's Functions in the respective store is
    created.
    """

    use_crust2 = Bool.T(
        default=True,
        help='Flag, for replacing the crust from the earthmodel'
             'with crust from the crust2 model.')
    replace_water = Bool.T(
        default=True,
        help='Flag, for replacing water layers in the crust2 model.')
    custom_velocity_model = gf.Earthmodel1D.T(
        default=None,
        optional=True,
        help='Custom Earthmodel, in case crust2 and standard model not'
             ' wanted. Needs to be a :py::class:cake.LayeredModel')
    source_depth_min = Float.T(
        default=0.,
        help='Minimum depth [km] for GF function grid.')
    source_depth_max = Float.T(
        default=10.,
        help='Maximum depth [km] for GF function grid.')
    source_depth_spacing = Float.T(
        default=1.,
        help='Depth spacing [km] for GF function grid.')
    source_distance_radius = Float.T(
        default=20.,
        help='Radius of distance grid [km] for GF function grid around '
             'reference event.')
    source_distance_spacing = Float.T(
        default=1.,
        help='Distance spacing [km] for GF function grid w.r.t'
             ' reference_location.')
    error_depth = Float.T(
        default=0.1,
        help='3sigma [%/100] error in velocity model layer depth, '
             'translates to interval for varying the velocity model')
    error_velocities = Float.T(
        default=0.1,
        help='3sigma [%/100] in velocity model layer wave-velocities, '
             'translates to interval for varying the velocity model')
    depth_limit_variation = Float.T(
        default=600.,
        help='Depth limit [km] for varying the velocity model. Below that '
             'depth the velocity model is not varied based on the errors '
             'defined above!')


class SeismicGFConfig(NonlinearGFConfig):
    """
    Seismic GF parameters for Layered Halfspace.
    """
    reference_location = ReferenceLocation.T(
        default=None,
        help="Reference location for the midpoint of the Green's Function "
             "grid.",
        optional=True)
    code = String.T(
        default='qssp',
        help='Modeling code to use. (qssp, qseis, comming soon: '
             'qseis2d)')
    sample_rate = Float.T(
        default=2.,
        help='Sample rate for the Greens Functions.')
    rm_gfs = Bool.T(
        default=True,
        help='Flag for removing modeling module GF files after'
             ' completion.')


class GeodeticGFConfig(NonlinearGFConfig):
    """
    Geodetic GF parameters for Layered Halfspace.
    """
    code = String.T(
        default='psgrn',
        help='Modeling code to use. (psgrn, ... others need to be'
             'implemented!)')
    sample_rate = Float.T(
        default=1. / (3600. * 24.),
        help='Sample rate for the Greens Functions. Mainly relevant for'
             ' viscoelastic modeling. Default: coseismic-one day')
    sampling_interval = Float.T(
        default=1.0,
        help='Distance dependend sampling spacing coefficient.'
             '1. - equidistant')
    medium_depth_spacing = Float.T(
        default=1.,
        help='Depth spacing [km] for GF medium grid.')
    medium_distance_spacing = Float.T(
        default=1.,
        help='Distance spacing [km] for GF medium grid.')


class LinearGFConfig(GFConfig):
    """
    Config for linear GreensFunction calculation parameters.
    """
    reference_sources = List.T(
        RectangularSource.T(),
        help='Geometry of the reference source(s) to fix')
    patch_width = Float.T(
        default=5.,
        help='Patch width [km] to divide reference sources')
    patch_length = Float.T(
        default=5.,
        help='Patch length [km] to divide reference sources')
    extension_width = Float.T(
        default=0.1,
        help='Extend reference sources by this factor in each'
             ' dip-direction. 0.1 means extension of the fault by 10% in each'
             ' direction, i.e. 20% in total. If patches would intersect with'
             ' the free surface they are constrained to end at the surface.')
    extension_length = Float.T(
        default=0.1,
        help='Extend reference sources by this factor in each'
             ' strike-direction. 0.1 means extension of the fault by 10% in'
             ' each direction, i.e. 20% in total.')
    sample_rate = Float.T(
        default=2.,
        help='Sample rate for the Greens Functions.')


class SeismicLinearGFConfig(LinearGFConfig):
    """
    Config for seismic linear GreensFunction calculation parameters.
    """
    reference_location = ReferenceLocation.T(
        default=None,
        help="Reference location for the midpoint of the Green's Function "
             "grid.",
        optional=True)
    duration_sampling = Float.T(
        default=1.,
        help="Calculate Green's Functions for varying Source Time Function"
             " durations determined by prior bounds. Discretization between"
             " is determined by duration sampling.")
    starttime_sampling = Float.T(
        default=1.,
        help="Calculate Green's Functions for varying rupture onset times."
             "These are determined by the (rupture) velocity prior bounds "
             "and the hypocenter location.")


class GeodeticLinearGFConfig(LinearGFConfig):
    pass


class WaveformFitConfig(Object):
    """
    Config for specific parameters that are applied to post-process
    a specific type of waveform and calculate the misfit.
    """
    include = Bool.T(
        default=True,
        help='Flag to include waveform into optimization.')
    preprocess_data = Bool.T(
        default=True,
        help='Flag to filter input data.')
    name = String.T('any_P')
    blacklist = List.T(
        String.T(),
        default=[String.D()],
        help='Station name for stations to be thrown out.')
    channels = List.T(String.T(), default=['Z'])
    filterer = Filter.T(default=Filter.D())
    distances = Tuple.T(2, Float.T(), default=(30., 90.))
    interpolation = StringChoice.T(
        choices=['nearest_neighbor', 'multilinear'],
        default='multilinear',
        help='GF interpolation sceme')
    arrival_taper = trace.Taper.T(
        default=ArrivalTaper.D(),
        help='Taper a,b/c,d time [s] before/after wave arrival')


class SeismicNoiseAnalyserConfig(Object):

    structure = StringChoice.T(
        choices=['identity', 'exponential', 'import', 'non-toeplitz'],
        default='identity',
        help='Determines data-covariance matrix structure.')
    pre_arrival_time = Float.T(
        default=5.,
        help='Time [s] before synthetic P-wave arrival until '
             'variance is estimated')


class SeismicConfig(Object):
    """
    Config for seismic data optimization related parameters.
    """

    datadir = String.T(default='./')
    noise_estimator = SeismicNoiseAnalyserConfig.T(
        default=SeismicNoiseAnalyserConfig.D(),
        help='Determines the structure of the data-covariance matrix.')
    pre_stack_cut = Bool.T(
        default=True,
        help='Cut the GF traces before stacking around the specified arrival'
             ' taper')
    station_corrections = Bool.T(
        default=False,
        help='If set, optimize for time shift for each station.')
    waveforms = List.T(WaveformFitConfig.T(default=WaveformFitConfig.D()))
    gf_config = GFConfig.T(default=SeismicGFConfig.D())

    def get_waveform_names(self):
        return [wc.name for wc in self.waveforms]

    def get_unique_channels(self):
        cl = [wc.channels for wc in self.waveforms]
        uc = []
        for c in cl:
            uc.extend(c)
        return list(set(uc))

    def get_hypernames(self):
        hids = []
        for i, wc in enumerate(self.waveforms):
            if wc.include:
                for c in wc.channels:
                    hypername = '_'.join(('h', wc.name, str(i), c))
                    hids.append(hypername)

        return hids

    def get_hierarchical_names(self):
        if self.station_corrections:
            return ['time_shift']
        else:
            return []

    def init_waveforms(self, wavenames):
        """
        Initialise waveform configurations.
        """
        for wavename in wavenames:
            self.waveforms.append(WaveformFitConfig(name=wavename))


class GeodeticConfig(Object):
    """
    Config for geodetic data optimization related parameters.
    """

    datadir = String.T(default='./')
    names = List.T(String.T(), default=['Data prefix filenames here ...'])
    blacklist = List.T(
        String.T(),
        optional=True,
        default=[],
        help='GPS station name or scene name to be thrown out.')
    types = List.T(
        String.T(),
        default=['SAR'],
        help='Types of geodetic data, i.e. SAR, GPS, ...')
    calc_data_cov = Bool.T(
        default=True,
        help='Flag for calculating the data covariance matrix, '
             'outsourced to "kite"')
    interpolation = StringChoice.T(
        choices=['nearest_neighbor', 'multilinear'],
        default='multilinear',
        help='GF interpolation scheme during synthetics generation')
    fit_plane = Bool.T(
        default=False,
        help='Flag for inverting for additional plane parameters on each'
             ' SAR datatype')
    gf_config = GFConfig.T(default=GeodeticGFConfig.D())

    def get_hypernames(self):
        return ['_'.join(('h', typ)) for typ in self.types]

    def get_hierarchical_names(self):
        if self.fit_plane:
            return [name + '_ramp' for name in self.names]
        else:
            return []


class ModeConfig(Object):
    """
    BaseConfig for optimization mode specific configuration.
    """
    pass


class FFIConfig(ModeConfig):

    regularization = StringChoice.T(
        default='none',
        choices=['laplacian', 'trans-dimensional', 'none'],
        help='Flag for regularization in distributed slip-optimization.')


class ProblemConfig(Object):
    """
    Config for optimization problem to setup.
    """
    mode = StringChoice.T(
        choices=['geometry', 'ffi', 'interseismic'],
        default='geometry',
        help='Problem to solve: "geometry", "ffi",'
             ' "interseismic"',)
    mode_config = ModeConfig.T(
        optional=True,
        help='Global optimization mode specific parameters.')
    source_type = StringChoice.T(
        default='RectangularSource',
        choices=source_names,
        help='Source type to optimize for. Options: %s' % (
            ', '.join(name for name in source_names)))
    stf_type = StringChoice.T(
        default='HalfSinusoid',
        choices=stf_names,
        help='Source time function type to use. Options: %s' % (
            ', '.join(name for name in stf_names)))
    decimation_factors = Dict.T(
        default=None,
        optional=True,
        help='Determines the reduction of discretization of an extended'
             ' source.')
    n_sources = Int.T(
        default=1,
        help='Number of Sub-sources to solve for')
    datatypes = List.T(default=['geodetic'])
    dataset_specific_residual_noise_estimation = Bool.T(
        default=False,
        help='If set, for EACH DATASET specific hyperparameter estimation.'
             'For seismic data: n_hypers = nstations * nchannels.'
             'For geodetic data: n_hypers = nimages (SAR) or '
             'nstations * ncomponents (GPS).'
             'If false one hyperparameter for each DATATYPE and '
             'displacement COMPONENT.')
    hyperparameters = Dict.T(
        default=OrderedDict(),
        help='Hyperparameters to estimate the noise in different'
             ' types of datatypes.')
    priors = Dict.T(
        default=OrderedDict(),
        help='Priors of the variables in question.')
    hierarchicals = Dict.T(
        default=OrderedDict(),
        help='Hierarchical parameters that affect the posterior'
             ' likelihood, but do not affect the forward problem.'
             ' Implemented: Temporal station corrections, orbital'
             ' ramp estimation')

    def __init__(self, **kwargs):

        mode = 'mode'
        mode_config = 'mode_config'
        if mode in kwargs:
            omode = kwargs[mode]

            if omode == 'ffi':
                if mode_config not in kwargs:
                    kwargs[mode_config] = FFIConfig()

        Object.__init__(self, **kwargs)

    def init_vars(self, variables=None, nvars=None):
        """
        Initiate priors based on the problem mode and datatypes.

        Parameters
        ----------
        variables : list
            of str of variable names to initialise
        """
        if variables is None:
            variables = self.select_variables()

        self.priors = OrderedDict()

        for variable in variables:

            if nvars is None:
                if variable in block_vars:
                    nvars = 1
                else:
                    nvars = self.n_sources

            lower = default_bounds[variable][0]
            upper = default_bounds[variable][1]
            self.priors[variable] = \
                Parameter(
                    name=variable,
                    lower=num.ones(
                        nvars,
                        dtype=tconfig.floatX) * lower,
                    upper=num.ones(
                        nvars,
                        dtype=tconfig.floatX) * upper,
                    testvalue=num.ones(
                        nvars,
                        dtype=tconfig.floatX) * (lower + (upper / 5.)))

    def set_vars(self, bounds_dict):
        """
        Set variable bounds to given bounds.
        """
        for variable, bounds in bounds_dict.items():
            if variable in self.priors.keys():
                param = self.priors[variable]
                param.lower = num.atleast_1d(bounds[0])
                param.upper = num.atleast_1d(bounds[1])
                param.testvalue = num.atleast_1d(num.mean(bounds))
            else:
                logger.warning(
                    'Prior for variable %s does not exist!'
                    ' Bounds not updated!' % variable)

    def select_variables(self):
        """
        Return model variables depending on problem config.
        """

        if self.mode not in modes_catalog.keys():
            raise ValueError('Problem mode %s not implemented' % self.mode)

        vars_catalog = modes_catalog[self.mode]

        variables = []
        for datatype in self.datatypes:
            if datatype in vars_catalog.keys():
                if self.mode == 'geometry':
                    if self.source_type in vars_catalog[datatype].keys():
                        source = vars_catalog[datatype][self.source_type]
                        svars = set(source.keys())

                        if isinstance(
                                source(), (PyrockoRS, gf.ExplosionSource)):
                            svars.discard('magnitude')

                        variables += utility.weed_input_rvs(
                            svars, self.mode, datatype)
                    else:
                        raise ValueError('Source Type not supported for type'
                                         ' of problem, and datatype!')

                    if datatype == 'seismic':
                        if self.stf_type in stf_catalog.keys():
                            stf = stf_catalog[self.stf_type]
                            variables += utility.weed_input_rvs(
                                set(stf.keys()), self.mode, datatype)
                else:
                    variables += vars_catalog[datatype]
            else:
                raise ValueError(
                    'Datatype %s not supported for type of'
                    ' problem! Supported datatype are: %s' % (
                        datatype, ', '.join(
                            '"%s"' % d for d in vars_catalog.keys())))

        unique_variables = utility.unique_list(variables)

        if len(unique_variables) == 0:
            raise Exception(
                'Mode and datatype combination not implemented'
                ' or not resolvable with given datatypes.')

        return unique_variables

    def get_slip_variables(self):
        """
        Return a list of slip variable names defined in the ProblemConfig.
        """
        if self.mode == 'ffi':
            return [
                var for var in static_dist_vars if var in self.priors.keys()]
        elif self.mode == 'geometry':
            return [
                var for var in ['slip', 'magnitude']
                if var in self.priors.keys()]
        elif self.mode == 'interseismic':
            return ['bl_amplitude']

    def set_decimation_factor(self):
        """
        Determines the reduction of discretization of an extended source.
        Influences yet only the RectangularSource.
        """
        if self.source_type == 'RectangularSource':
            self.decimation_factors = {}
            for datatype in self.datatypes:
                self.decimation_factors[datatype] = \
                    default_decimation_factors[datatype]
        else:
            self.decimation_factors = None

    def validate_priors(self):
        """
        Check if priors and their test values do not contradict!
        """
        for param in self.priors.values():
            param.validate_bounds()

        logger.info('All parameter-priors ok!')

    def validate_hypers(self):
        """
        Check if hyperparameters and their test values do not contradict!
        """
        if self.hyperparameters is not None:
            for hp in self.hyperparameters.values():
                hp.validate_bounds()

            logger.info('All hyper-parameters ok!')

        else:
            logger.info('No hyper-parameters defined!')

    def validate_hierarchicals(self):
        """
        Check if hierarchicals and their test values do not contradict!
        """
        if self.hierarchicals is not None:
            for hp in self.hierarchicals.values():
                hp.validate_bounds()

            logger.info('All hierarchical-parameters ok!')

        else:
            logger.info('No hyper-parameters defined!')

    def get_test_point(self):
        """
        Returns dict with test point
        """
        test_point = {}
        for varname, var in self.priors.items():
            test_point[varname] = var.testvalue

        for varname, var in self.hyperparameters.items():
            test_point[varname] = var.testvalue

        for varname, var in self.hierarchicals.items():
            test_point[varname] = var.testvalue

        return test_point


class SamplerParameters(Object):

    tune_interval = Int.T(
        default=50,
        help='Tune interval for adaptive tuning of Metropolis step size.')
    proposal_dist = String.T(
        default='Normal',
        help='Normal Proposal distribution, for Metropolis steps;'
             'Alternatives: Cauchy, Laplace, Poisson, MultivariateNormal')
    check_bnd = Bool.T(
        default=True,
        help='Flag for checking whether proposed step lies within'
             ' variable bounds.')

    rm_flag = Bool.T(default=False,
                     help='Remove existing results prior to sampling.')


class ParallelTemperingConfig(SamplerParameters):

    n_samples = Int.T(
        default=int(1e5),
        help='Number of samples of the posterior distribution.'
             ' Only the samples of processors that sample from the posterior'
             ' (beta=1) are kept.')
    n_chains = Int.T(
        default=2,
        help='Number of PT chains to sample in parallel.'
             ' A number < 2 will raise an Error, as this is the minimum'
             ' amount of chains needed. ')
    swap_interval = Tuple.T(
        2, Int.T(),
        default=(100, 300),
        help='Interval for uniform random integer that is drawn to determine'
             ' the length of MarkovChains on each worker. When chain is'
             ' completed the last sample is returned for swapping state'
             ' between chains. Consequently, lower number will result in'
             ' more state swapping.')
    beta_tune_interval = Int.T(
        default=int(5e3),
        help='Sample interval of master chain after which the chain swap'
             ' acceptance is evaluated. High acceptance will result in'
             ' closer spaced betas and vice versa.')
    n_chains_posterior = Int.T(
        default=1,
        help='Number of chains that sample from the posterior at beat=1.')
    resample = Bool.T(
        default=False,
        help='If "true" the testvalue of the priors is taken as seed for'
             ' all Markov Chains.')
    thin = Int.T(
        default=3,
        help='Thinning parameter of the sampled trace. Every "thin"th sample'
             ' is taken.')
    burn = Float.T(
        default=0.5,
        help='Burn-in parameter between 0. and 1. to discard fraction of'
             ' samples from the beginning of the chain.')


class MetropolisConfig(SamplerParameters):
    """
    Config for optimization parameters of the Adaptive Metropolis algorithm.
    """
    n_jobs = Int.T(
        default=1,
        help='Number of processors to use, i.e. chains to sample in parallel.')
    n_steps = Int.T(default=25000,
                    help='Number of steps for the MC chain.')
    n_chains = Int.T(default=20,
                     help='Number of Metropolis chains for sampling.')
    thin = Int.T(
        default=2,
        help='Thinning parameter of the sampled trace. Every "thin"th sample'
             ' is taken.')
    burn = Float.T(
        default=0.5,
        help='Burn-in parameter between 0. and 1. to discard fraction of'
             ' samples from the beginning of the chain.')


class SMCConfig(SamplerParameters):
    """
    Config for optimization parameters of the SMC algorithm.
    """
    n_jobs = Int.T(
        default=1,
        help='Number of processors to use, i.e. chains to sample in parallel.')
    n_steps = Int.T(default=100,
                    help='Number of steps for the MC chain.')
    n_chains = Int.T(default=1000,
                     help='Number of Metropolis chains for sampling.')
    coef_variation = Float.T(
        default=1.,
        help='Coefficient of variation, determines the similarity of the'
             'intermediate stage pdfs;'
             'low - small beta steps (slow cooling),'
             'high - wide beta steps (fast cooling)')
    stage = Int.T(default=0,
                  help='Stage where to start/continue the sampling. Has to'
                       ' be int -1 for final stage')
    proposal_dist = String.T(
        default='MultivariateNormal',
        help='Multivariate Normal Proposal distribution, for Metropolis steps'
             'alternatives need to be implemented')

    update_covariances = Bool.T(
        default=True,
        optional=True,
        help='Update model prediction covariance matrixes in transition '
             'stages.')


class SamplerConfig(Object):
    """
    Config for the sampler specific parameters.
    """

    name = String.T(
        default='SMC',
        help='Sampler to use for sampling the solution space.'
             ' Metropolis/ SMC')
    progressbar = Bool.T(
        default=True,
        help='Display progressbar(s) during sampling.')
    buffer_size = Int.T(
        default=5000,
        help='number of samples after which the result '
             'buffer is written to disk')
    parameters = SamplerParameters.T(
        default=SMCConfig.D(),
        optional=True,
        help='Sampler dependend Parameters')

    def set_parameters(self, **kwargs):

        if self.name is None:
            logger.info('Sampler not defined, using default sampler: SMC')
            self.name = 'SMC'

        if self.name == 'SMC':
            self.parameters = SMCConfig(**kwargs)

        elif self.name != 'SMC':
            kwargs.pop('update_covariances', None)

            if self.name == 'Metropolis':
                self.parameters = MetropolisConfig(**kwargs)

            elif self.name == 'PT':
                self.parameters = ParallelTemperingConfig(**kwargs)

            else:
                raise TypeError('Sampler "%s" is not implemented.' % self.name)


class GFLibaryConfig(Object):
    """
    Baseconfig for GF Libraries
    """
    component = String.T(default='uparr')
    event = model.Event.T(default=model.Event.D())
    datatype = String.T(default='undefined')
    crust_ind = Int.T(default=0)


class GeodeticGFLibraryConfig(GFLibaryConfig):
    """
    Config for the linear Geodetic GF Library for dumping and loading.
    """
    dimensions = Tuple.T(2, Int.T(), default=(0, 0))


class SeismicGFLibraryConfig(GFLibaryConfig):
    """
    Config for the linear Seismic GF Library for dumping and loading.
    """
    wave_config = WaveformFitConfig.T(default=WaveformFitConfig.D())
    starttime_sampling = Float.T(default=0.5)
    duration_sampling = Float.T(default=0.5)
    starttime_min = Float.T(default=0.)
    duration_min = Float.T(default=0.1)
    dimensions = Tuple.T(5, Int.T(), default=(0, 0, 0, 0, 0))


class BEATconfig(Object, Cloneable):
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
    hyper_sampler_config = SamplerConfig.T(
        default=SamplerConfig.D(),
        optional=True)

    def update_hypers(self):
        """
        Evaluate the whole config and initialise necessary hyperparameters.
        """

        hypernames = []
        if self.geodetic_config is not None:
            hypernames.extend(self.geodetic_config.get_hypernames())

        if self.seismic_config is not None:
            hypernames.extend(self.seismic_config.get_hypernames())

        if self.problem_config.mode == 'ffi':
            if self.problem_config.mode_config.regularization == 'laplacian':
                hypernames.append(hyper_name_laplacian)

        hypers = OrderedDict()
        for name in hypernames:
            logger.info(
                'Added hyperparameter %s to config and '
                'model setup!' % name)

            defaultb_name = 'hypers'
            hypers[name] = Parameter(
                name=name,
                lower=num.ones(1, dtype=tconfig.floatX) *
                default_bounds[defaultb_name][0],
                upper=num.ones(1, dtype=tconfig.floatX) *
                default_bounds[defaultb_name][1],
                testvalue=num.ones(1, dtype=tconfig.floatX) *
                num.mean(default_bounds[defaultb_name]))

        self.problem_config.hyperparameters = hypers
        self.problem_config.validate_hypers()

        n_hypers = len(hypers)
        logger.info('Number of hyperparameters! %i' % n_hypers)
        if n_hypers == 0:
            self.hyper_sampler_config = None

    def update_hierarchicals(self):
        """
        Evaluate the whole config and initialise necessary
        hierarchical parameters.
        """

        hierarnames = []
        if self.geodetic_config is not None:
            hierarnames.extend(self.geodetic_config.get_hierarchical_names())

        if self.seismic_config is not None:
            hierarnames.extend(self.seismic_config.get_hierarchical_names())

        hierarchicals = OrderedDict()
        for name in hierarnames:
            logger.info(
                'Added hierarchical parameter %s to config and '
                'model setup!' % name)

            if name == 'time_shift':
                shp = 1
                defaultb_name = name
            else:
                shp = 2
                defaultb_name = 'ramp'

            hierarchicals[name] = Parameter(
                name=name,
                lower=num.ones(shp, dtype=tconfig.floatX) *
                default_bounds[defaultb_name][0],
                upper=num.ones(shp, dtype=tconfig.floatX) *
                default_bounds[defaultb_name][1],
                testvalue=num.ones(shp, dtype=tconfig.floatX) *
                num.mean(default_bounds[defaultb_name]))

        self.problem_config.hierarchicals = hierarchicals
        self.problem_config.validate_hierarchicals()

        n_hierarchicals = len(hierarchicals)
        logger.info('Number of hierarchicals! %i' % n_hierarchicals)


def init_reference_sources(source_points, n_sources, source_type, stf_type):
    reference_sources = []
    for i in range(n_sources):
        # rf = source_catalog[source_type](stf=stf_catalog[stf_type]())
        # maybe future if several meshtypes
        rf = RectangularSource(stf=stf_catalog[stf_type]())
        utility.update_source(rf, **source_points[i])
        reference_sources.append(rf)

    return reference_sources


def init_config(name, date=None, min_magnitude=6.0, main_path='./',
                datatypes=['geodetic'],
                mode='geometry', source_type='RectangularSource', n_sources=1,
                waveforms=['any_P'], sampler='SMC', hyper_sampler='Metropolis',
                use_custom=False, individual_gfs=False):
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
    datatypes : List of strings
        data sets to include in the optimization: either 'geodetic' and/or
        'seismic'
    mode : str
        type of optimization problem: 'Geometry' / 'Static'/ 'Kinematic'
    n_sources : int
        number of sources to solve for / discretize depending on mode parameter
    waveforms : list
        of strings of waveforms to include into the misfit function and
        GF calculation
    sampler : str
        Optimization algorithm to use to sample the solution space
        Options: 'SMC', 'Metropolis'
    use_custom : boolean
        Flag to setup manually a custom velocity model.
    individual_gfs : boolean
        Flag to use individual Green's Functions for each specific station.
        If false a reference location will be initialised in the config file.
        If true the reference locations will be taken from the imported station
        objects.

    Returns
    -------
    :class:`BEATconfig`
    """

    c = BEATconfig(name=name, date=date)
    c.project_dir = os.path.join(os.path.abspath(main_path), name)

    if mode == 'geometry' or mode == 'interseismic':
        if date is not None and not mode == 'interseismic':
            c.event = utility.search_catalog(
                date=date, min_magnitude=min_magnitude)

        elif mode == 'interseismic':
            c.event = model.Event(lat=10., lon=10., depth=0.)
            c.date = 'dummy'
            logger.info(
                'Interseismic mode! Using event as reference for the'
                ' stable block! Please update coordinates!')
        else:
            logger.warn(
                'No given date! Using dummy event!'
                ' Updating reference coordinates (spatial & temporal)'
                ' necessary!')
            c.event = model.Event(duration=1.)
            c.date = 'dummy'

        if 'geodetic' in datatypes:
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

        if 'seismic' in datatypes:
            c.seismic_config = SeismicConfig()
            c.seismic_config.init_waveforms(waveforms)

            if not individual_gfs:
                c.seismic_config.gf_config.reference_location = \
                    ReferenceLocation(lat=10.0, lon=10.0)
            else:
                c.seismic_config.gf_config.reference_location = None

            if use_custom:
                logger.info('use_custom flag set! The velocity model in the'
                            ' seismic GF configuration has to be updated!')
                c.seismic_config.gf_config.custom_velocity_model = \
                    load_model().extract(depth_max=100. * km)
                c.seismic_config.gf_config.use_crust2 = False
                c.seismic_config.gf_config.replace_water = False
        else:
            c.seismic_config = None

    elif mode == 'ffi':

        if source_type != 'RectangularSource':
            raise TypeError('Static distributed slip is so far only supported'
                            ' for RectangularSource(s)')

        gmc = load_config(c.project_dir, 'geometry')

        if gmc is not None:
            logger.info('Taking information from geometry_config ...')
            if source_type != gmc.problem_config.source_type:
                raise ValueError(
                    'Specified reference source: "%s" differs from the'
                    ' source that has been used previously in'
                    ' "geometry" mode: "%s"!' % (
                        source_type, gmc.problem_config.source_type))

            n_sources = gmc.problem_config.n_sources
            point = {k: v.testvalue
                     for k, v in gmc.problem_config.priors.items()}
            point = utility.adjust_point_units(point)
            source_points = utility.split_point(point)

            reference_sources = init_reference_sources(
                source_points, n_sources,
                gmc.problem_config.source_type, gmc.problem_config.stf_type)

            c.date = gmc.date
            c.event = gmc.event

            if 'geodetic' in datatypes:
                gc = gmc.geodetic_config
                if gc is None:
                    logger.warning(
                        'Asked for "geodetic" datatype but geometry config '
                        'has no such datatype! Initialising default "geodetic"'
                        ' linear config!')
                    gc = GeodeticConfig()
                    lgf_config = GeodeticLinearGFConfig()
                else:
                    lgf_config = GeodeticLinearGFConfig(
                        earth_model_name=gc.gf_config.earth_model_name,
                        store_superdir=gc.gf_config.store_superdir,
                        n_variations=gc.gf_config.n_variations,
                        reference_sources=reference_sources,
                        sample_rate=gc.gf_config.sample_rate)

                c.geodetic_config = gc
                c.geodetic_config.gf_config = lgf_config

            elif 'seismic' in datatypes:
                sc = gmc.seismic_config
                if sc is None:
                    logger.warning(
                        'Asked for "seismic" datatype but geometry config '
                        'has no such datatype! Initialising default "seismic"'
                        ' linear config!')
                    sc = SeismicConfig()
                    lgf_config = SeismicLinearGFConfig()
                else:
                    lgf_config = SeismicLinearGFConfig(
                        earth_model_name=sc.gf_config.earth_model_name,
                        sample_rate=sc.gf_config.sample_rate,
                        reference_location=sc.gf_config.reference_location,
                        store_superdir=sc.gf_config.store_superdir,
                        n_variations=sc.gf_config.n_variations,
                        reference_sources=reference_sources)
                c.seismic_config = sc
                c.seismic_config.gf_config = lgf_config
        else:
            logger.warning('Found no geometry setup, ...')
            raise ImportError(
                'No geometry configuration file existing! Please initialise'
                ' a "geometry" configuration ("beat init command"), update'
                ' the Greens Function information and create GreensFunction'
                ' stores for the non-linear problem.')

    c.problem_config = ProblemConfig(
        n_sources=n_sources, datatypes=datatypes, mode=mode,
        source_type=source_type)
    c.problem_config.init_vars()
    c.problem_config.set_decimation_factor()

    c.sampler_config = SamplerConfig(name=sampler)
    c.sampler_config.set_parameters(update_covariances=False)

    c.hyper_sampler_config = SamplerConfig(name=hyper_sampler)
    c.hyper_sampler_config.set_parameters(update_covariances=None)

    c.update_hypers()
    c.problem_config.validate_priors()

    c.regularize()
    c.validate()

    logger.info('Project_directory: %s \n' % c.project_dir)
    util.ensuredir(c.project_dir)

    dump_config(c)
    return c


def dump_config(config):
    """
    Load configuration file.

    Parameters
    ----------
    config : :class:`BEATConfig`
    """
    config_file_name = 'config_' + config.problem_config.mode + '.yaml'
    conf_out = os.path.join(config.project_dir, config_file_name)
    dump(config, filename=conf_out)


def load_config(project_dir, mode):
    """
    Load configuration file.

    Parameters
    ----------
    project_dir : str
        path to the directory of the configuration file
    mode : str
        type of optimization problem: 'geometry' / 'static'/ 'kinematic'
    update : list
        of strings to update parameters
        'hypers' or/and 'hierarchicals'

    Returns
    -------
    :class:`BEATconfig`
    """
    config_file_name = 'config_' + mode + '.yaml'

    config_fn = os.path.join(project_dir, config_file_name)

    try:
        config = load(filename=config_fn)
    except IOError:
        raise IOError('Cannot load config, file %s'
                      ' does not exist!' % config_fn)

    config.problem_config.validate_priors()

    return config
