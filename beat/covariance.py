from pyrocko import gf, trace

import numpy as num
from time import time
from scipy.linalg import toeplitz

import logging

from beat import heart
from beat.utility import ensure_cov_psd, running_window_rms, list2string
from theano import config as tconfig

from pymc3 import Point


logger = logging.getLogger('covariance')


__all__ = [
    'geodetic_cov_velocity_models',
    'geodetic_cov_velocity_models_pscmp',
    'seismic_cov_velocity_models',
    'SeismicNoiseAnalyser']


def exponential_data_covariance(n, dt, tzero):
    """
    Get exponential sub-covariance matrix without variance, toeplitz form.

    Parameters
    ----------
    n : int
        length of trace/ samples of quadratic Covariance matrix
    dt : float
        time step of samples, sampling interval
    tzero : float
        shortest period of waves in trace

    Returns
    -------
    :class:`numpy.ndarray`

    Notes
    -----
    Cd(i,j) = (Variance of trace)*exp(-abs(ti-tj)/
                                     (shortest period T0 of waves))

    i,j are samples of the seismic trace
    """
    return num.exp(-num.abs(
        num.arange(n)[:, num.newaxis] - num.arange(n)[num.newaxis, :]) * (
        dt / tzero))


def identity_data_covariance(n, dt=None, tzero=None):
    """
    Get identity covariance matrix.

    Parameters
    ----------
    n : int
        length of trace/ samples of quadratic Covariance matrix

    Returns
    -------
    :class:`numpy.ndarray`
    """
    return num.eye(n, dtype=tconfig.floatX)


def ones_data_covariance(n, dt=None, tzero=None):
    """
    Get ones covariance matrix. Dummy for importing.

    Parameters
    ----------
    n : int
        length of trace/ samples of quadratic Covariance matrix

    Returns
    -------
    :class:`numpy.ndarray`
    """
    return num.ones((n, n), dtype=tconfig.floatX)


NoiseStructureCatalog = {
    'variance': identity_data_covariance,
    'exponential': exponential_data_covariance,
    'import': ones_data_covariance,
    'non-toeplitz': ones_data_covariance,
}


def available_noise_structures():
    return list(NoiseStructureCatalog.keys())


def import_data_covariance(data_trace, arrival_taper, sample_rate):
    """
    Use imported covariance matrixes and check size consistency with taper.
    Cut or extend based on variance and taper size.

    Parameters
    ----------
    data_trace : :class:`heart.SeismicDataset`
        with data covariance matrix in the covariance attribute
    arrival_taper : :class: `heart.ArrivalTaper`
        determines tapering around phase Arrival
    sample_rate : float
        sampling rate of data_traces and GreensFunction stores

    Returns
    -------
    covariance matrix : :class:`numpy.ndarray`
        with size of given arrival taper
    """

    logger.info('No data-covariance estimation, using imported'
                ' covariances...\n')

    at = arrival_taper
    n_samples = at.nsamples(sample_rate)

    if data_trace.covariance is None:
        logger.warn(
            'No data covariance given/estimated! '
            'Setting default: eye')
        return num.eye(n_samples)
    else:
        data_cov = data_trace.covariance.data
        if data_cov.shape[0] != n_samples:
            logger.warn(
                'Imported covariance %i does not agree '
                ' with taper samples %i! Using Identity'
                ' matrix and mean of variance of imported'
                ' covariance matrix!' % (
                    data_cov.shape[0], n_samples))
            data_cov = num.eye(n_samples) * \
                data_cov.diagonal().mean()

        return data_cov


class SeismicNoiseAnalyser(object):
    """
    Seismic noise analyser

    Parameters
    ----------
    structure : string
        either identity, exponential, import
    pre_arrival_time : float
        in [s], time before P arrival until variance is estimated
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        processing object for synthetics calculation
    events : list
        of :class:`pyrocko.meta.Event`
        reference event(s) from catalog
    chop_bounds : list of len 2
        of taper attributes a, b, c, or d
    """
    def __init__(
            self, structure='identity', pre_arrival_time=5.,
            engine=None, events=None, sources=None, chop_bounds=['b', 'c']):

        avail = available_noise_structures()
        if structure not in avail:
            raise AttributeError(
                'Selected noise structure "%s" not supported! Implemented'
                ' noise structures: %s' % (structure, list2string(avail)))

        self.events = events
        self.engine = engine
        self.sources = sources
        self.pre_arrival_time = pre_arrival_time
        self.structure = structure
        self.chop_bounds = chop_bounds

    def get_structure(self, wmap, sample_rate, chop_bounds=None):

        if chop_bounds is None:
            chop_bounds = self.chop_bounds

        tzero = 1. / wmap.config.filterer.upper_corner
        dt = 1. / sample_rate
        ataper = wmap.config.arrival_taper
        n = ataper.nsamples(sample_rate, chop_bounds)
        return NoiseStructureCatalog[self.structure](n, dt, tzero)

    def do_import(self, wmap, sample_rate):

        scalings = []
        for tr, target in zip(wmap.datasets, wmap.targets):
            scaling = import_data_covariance(
                tr, arrival_taper=wmap.config.arrival_taper,
                sample_rate=sample_rate)
            scalings.append(scaling)

        return scalings

    def do_non_toeplitz(self, wmap, results):

        if results is None:
            ValueError(
                'Results need(s) to be given for non-toeplitz'
                ' covariance estimates!')
        else:
            scalings = []
            for result in results:
                residual = result.processed_res.get_ydata()
                scaling = non_toeplitz_covariance(
                    residual, window_size=residual.size // 5)
                scalings.append(scaling)

            return scalings

    def do_variance_estimate(self, wmap):

        filterer = wmap.config.filterer
        scalings = []

        for i, (tr, target) in enumerate(zip(wmap.datasets, wmap.targets)):
            wavename = None   # None uses first tabulated phase
            arrival_time = heart.get_phase_arrival_time(
                engine=self.engine,
                source=self.events[wmap.config.event_idx],
                target=target,
                wavename=wavename)

            if arrival_time < tr.tmin:
                logger.warning(
                    'no data for variance estimation on pre-P arrival'
                    ' in wavemap %s, for trace %s!' % (
                        wmap._mapid, list2string(tr.nslc_id)))
                logger.info(
                    'Using reference arrival "%s" instead!' % wmap.name)
                arrival_time = heart.get_phase_arrival_time(
                    engine=self.engine,
                    source=self.events[wmap.config.event_idx],
                    target=target,
                    wavename=wmap.name)

            if filterer is not None:
                ctrace = tr.copy()
                ctrace.bandpass(
                    corner_hp=filterer.lower_corner,
                    corner_lp=filterer.upper_corner,
                    order=filterer.order)

            ctrace = ctrace.chop(
                tmin=tr.tmin,
                tmax=arrival_time - self.pre_arrival_time)

            scaling = num.var(ctrace.get_ydata())
            scalings.append(scaling)

        return scalings

    def get_data_covariances(
            self, wmap, sample_rate, results=None, chop_bounds=None):
        """
        Estimated data covariances of seismic traces

        Parameters
        ----------
        wmap : :class:`eat.WaveformMapping`
        results
        sample_rate : float
            sampling rate of data_traces and GreensFunction stores

        Returns
        -------
        :class:`numpy.ndarray`
        """

        covariance_structure = self.get_structure(
            wmap, sample_rate, chop_bounds)

        if self.structure == 'import':
            scalings = self.do_import(wmap, sample_rate)
        elif self.structure == 'non-toeplitz':
            scalings = self.do_non_toeplitz(wmap, results)
        else:
            scalings = self.do_variance_estimate(wmap)

        cov_ds = []
        for scaling in scalings:
            cov_ds.append(scaling * covariance_structure)

        return cov_ds


def model_prediction_sensitivity(engine, *args, **kwargs):
    '''
    Calculate the model prediction Covariance Sensitivity Kernel.
    (numerical derivation with respect to the input source parameter(s))
    Following Duputel et al. 2014

    :Input:
    :py:class:'engine'
    source_parms = list of parameters with respect to which the kernel
                   is being calculated e.g. ['strike', 'dip', 'depth']
    !!!
    NEEDS to have seismosizer source object parameter variable name convention
    !!!
    (see seismosizer.source.keys())

    calculate_model_prediction_sensitivity(request, source_params, **kwargs)
    calculate_model_prediction_sensitivity(sources,
                                             targets, source_params, **kwargs)

    Returns traces in a list[parameter][targets] for each station and channel
    as specified in the targets. The location code of each trace is placed to
    show the respective source parameter.
    '''

    if len(args) not in (0, 1, 2, 3):
        raise gf.BadRequest('invalid arguments')

    if len(args) == 2:
        kwargs['request'] = args[0]
        kwargs['source_params'] = args[1]

    elif len(args) == 3:
        kwargs.update(gf.Request.args2kwargs(args[0:1]))
        kwargs['source_params'] = args[2]

    request = kwargs.pop('request', None)
    nprocs = kwargs.pop('nprocs', 1)
    source_params = kwargs.pop('source_params', None)
    h = kwargs.pop('h', None)

    if request is None:
        request = gf.Request(**kwargs)

    if h is None:
        h = num.ones(len(source_params)) * 1e-1

    # create results list
    sensitivity_param_list = []
    sensitivity_param_trcs = []

    for i in range(len(source_params)):
        sensitivity_param_list.append([0] * len(request.targets))
        sensitivity_param_trcs.append([0] * len(request.targets))

    for ref_source in request.sources:
        par_count = 0
        for param in source_params:
            print(param, 'with h = ', h[par_count])
            calc_source_p2h = ref_source.clone()
            calc_source_ph = ref_source.clone()
            calc_source_mh = ref_source.clone()
            calc_source_m2h = ref_source.clone()

            setattr(calc_source_p2h, param,
                    ref_source[param] + (2 * h[par_count]))
            setattr(calc_source_ph, param,
                    ref_source[param] + (h[par_count]))
            setattr(calc_source_mh, param,
                    ref_source[param] - (h[par_count]))
            setattr(calc_source_m2h, param,
                    ref_source[param] - (2 * h[par_count]))

            calc_sources = [calc_source_p2h, calc_source_ph,
                            calc_source_mh, calc_source_m2h]

            response = engine.process(sources=calc_sources,
                                      targets=request.targets,
                                      nprocs=nprocs)

            for k in range(len(request.targets)):
                # zero padding if necessary
                trc_lengths = num.array(
                    [len(response.results_list[i][k].trace.data) for i in
                     range(len(response.results_list))])
                Id = num.where(trc_lengths != trc_lengths.max())

                for l in Id[0]:
                    response.results_list[l][k].trace.data = num.concatenate(
                            (response.results_list[l][k].trace.data,
                             num.zeros(trc_lengths.max() - trc_lengths[l])))

                # calculate numerical partial derivative for
                # each source and target
                sensitivity_param_list[par_count][k] = (
                        sensitivity_param_list[par_count][k] + (
                            - response.results_list[0][k].trace.data +
                            8 * response.results_list[1][k].trace.data -
                            8 * response.results_list[2][k].trace.data +
                            response.results_list[3][k].trace.data) /
                        (12 * h[par_count])
                    )

            par_count = par_count + 1

    # form traces from sensitivities
    par_count = 0
    for param in source_params:
        for k in range(len(request.targets)):
            sensitivity_param_trcs[par_count][k] = trace.Trace(
                        network=request.targets[k].codes[0],
                        station=request.targets[k].codes[1],
                        ydata=sensitivity_param_list[par_count][k],
                        deltat=response.results_list[0][k].trace.deltat,
                        tmin=response.results_list[0][k].trace.tmin,
                        channel=request.targets[k].codes[3],
                        location=param)

        par_count = par_count + 1

    return sensitivity_param_trcs


def seismic_cov_velocity_models(
        engine, sources, targets, arrival_taper, arrival_time,
        wavename, filterer, plot=False, n_jobs=1):
    '''
    Calculate model prediction uncertainty matrix with respect to uncertainties
    in the velocity model for station and channel.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        contains synthetics generation machine
    sources : list
        of :class:`pyrocko.gf.seismosizer.Source`
    targets : list
        of :class:`pyrocko.gf.seismosizer.Targets`
    arrival_taper : :class: `heart.ArrivalTaper`
        determines tapering around phase Arrival
    arrival_time : None or :class:`numpy.NdArray` or float
        of phase to apply taper, if None theoretic arrival of ray tracing used
    filterer : :class:`heart.Filter`
        determines the bandpass-filtering corner frequencies
    plot : boolean
        open snuffler and browse traces if True
    n_jobs : int
        number of processors to be used for calculation

    Returns
    -------
    :class:`numpy.ndarray` with Covariance due to velocity model uncertainties
    '''

    arrival_times = num.ones(len(targets), dtype='float64') * arrival_time

    t0 = time()
    synths, _ = heart.seis_synthetics(
        engine=engine,
        sources=sources,
        targets=targets,
        arrival_taper=arrival_taper,
        wavename=wavename,
        filterer=filterer,
        arrival_times=arrival_times,
        pre_stack_cut=True,
        plot=plot,
        outmode='array',
        chop_bounds=['b', 'c'])

    t1 = time()
    logger.debug('Trace generation time %f' % (t1 - t0))

    return num.cov(synths, rowvar=0)


def geodetic_cov_velocity_models(
        engine, sources, targets, dataset, plot=False, event=None, n_jobs=1):
    """
    Calculate model prediction uncertainty matrix with respect to uncertainties
    in the velocity model for geodetic targets using fomosto GF stores.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        contains synthetics generation machine
    target : :class:`pyrocko.gf.targets.StaticTarget`
        dataset and observation points to calculate covariance for
    sources : list
        of :py:class:`pyrocko.gf.seismosizer.Source` determines the covariance
        matrix
    plot : boolean
        if set, a plot is produced and not covariance matrix is returned

    Returns
    -------
    :class:`numpy.ndarray` with Covariance due to velocity model uncertainties
    """
    t0 = time()
    displacements = heart.geo_synthetics(
        engine=engine,
        targets=targets,
        sources=sources,
        outmode='stacked_arrays')
    t1 = time()
    logger.debug('Synthetics generation time %f' % (t1 - t0))

    synths = num.zeros((len(targets), dataset.samples))
    for i, disp in enumerate(displacements):
        synths[i, :] = (
            disp[:, 0] * dataset.los_vector[:, 0] +
            disp[:, 1] * dataset.los_vector[:, 1] +
            disp[:, 2] * dataset.los_vector[:, 2]) * dataset.odw

    if plot:
        from matplotlib import pyplot as plt
        indexes = dataset.get_distances_to_event(event).argsort()  # noqa
        ax = plt.axes()
        im = ax.matshow(synths)  # [:, indexes])
        plt.colorbar(im)
        plt.show()

    return num.cov(synths, rowvar=0)


def geodetic_cov_velocity_models_pscmp(
        store_superdir, crust_inds, target, sources):
    """
    Calculate model prediction uncertainty matrix with respect to uncertainties
    in the velocity model for geodetic targets based on pscmp.
    Deprecated!!!

    Parameters
    ----------
    store_superdir : str
        Absolute path to the geodetic GreensFunction directory
    crust_inds : list
        of int of indices for respective GreensFunction store indexes
    target : :class:`heart.GeodeticDataset`
        dataset and observation points to calculate covariance for
    sources : list
        of :py:class:`pscmp.PsCmpRectangularSource` determines the covariance
        matrix

    Returns
    -------
    :class:`numpy.ndarray` with Covariance due to velocity model uncertainties
    """

    synths = num.zeros((len(crust_inds), target.samples))
    for crust_ind in crust_inds:
        disp = heart.geo_layer_synthetics(
            store_superdir, crust_ind,
            lons=target.lons,
            lats=target.lats,
            sources=sources)
        synths[crust_ind, :] = (
            disp[:, 0] * target.los_vector[:, 0] +
            disp[:, 1] * target.los_vector[:, 1] +
            disp[:, 2] * target.los_vector[:, 2]) * target.odw

    return num.cov(synths, rowvar=0)


def autocovariance(data):
    """
    Calculate autocovariance of data.

    Returns
    -------
    :class:`numpy.ndarray`

    Notes
    -----
    Following Dettmer et al. 2007 JASA
    """
    n = data.size
    meand = data.mean()

    autocov = num.zeros((n), tconfig.floatX)
    for j in range(n):
        for k in range(n - j):
            autocov[j] += (data[j + k] - meand) * (data[k] - meand)

    return autocov / n


def toeplitz_covariance(data, window_size):
    """
    Get Toeplitz banded matrix for given data.

    Returns
    -------
    toeplitz : :class:`numpy.ndarray` 2-d, (size data, size data)
    stds : :class:`numpy.ndarray` 1-d, size data
        of running windows
    """
    stds = running_window_rms(data, window_size=window_size, mode='same')
    coeffs = autocovariance(data / stds)
    return toeplitz(coeffs), stds


def non_toeplitz_covariance(data, window_size):
    """
    Get scaled non- Toeplitz covariance matrix, which may be able to account
    for non-stationary data-errors.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        of data to estimate non-stationary error matrix for
    window_size : int
        samples to take on running rmse estimation over data

    Returns
    -------
    :class:`numpy.ndarray` (size data, size data)
    """
    toeplitz, stds = toeplitz_covariance(data, window_size)
    return toeplitz * stds[:, num.newaxis] * stds[num.newaxis, :]


def init_proposal_covariance(bij, vars, model, pop_size=1000):
    """
    Create initial proposal covariance matrix based on random samples
    from the solution space.
    """
    population_array = num.zeros((pop_size, bij.ordering.size))
    for i in range(pop_size):
        point = Point({v.name: v.random() for v in vars}, model=model)
        population_array[i, :] = bij.map(point)

    return num.diag(population_array.var(0))


def calc_sample_covariance(buffer, lij, bij, beta):
    """
    Calculate trace covariance matrix based on given trace values.

    Parameters
    ----------
    lpoints : list
        of list points (e.g. buffer of traces)
    lij : `beat.utility.ListArrayOrdering`
        that holds orderings of RVs
    beta : float
        tempering parameter of the trace

    Returns
    -------
    cov : :class:`numpy.ndarray`
        weighted covariances (NumPy > 1.10. required)
    """
    n_points = len(buffer)

    population_array = num.zeros((n_points, bij.ordering.size))
    for i, (lpoint, _) in enumerate(buffer):
        point = lij.l2d(lpoint)
        population_array[i, :] = bij.map(point)

    like_idx = lij.ordering['like'].list_ind
    weights = num.array([lpoint[like_idx] for lpoint, _ in buffer])
    temp_weights = num.exp((weights - weights.max())).ravel()
    norm_weights = temp_weights / num.sum(temp_weights)

    cov = num.cov(
        population_array,
        aweights=norm_weights,
        bias=False,
        rowvar=0)

    cov = ensure_cov_psd(cov)
    if num.isnan(cov).any() or num.isinf(cov).any():
        logger.warn(
            'Proposal covariances contain Inf or NaN! '
            'For chain with beta: %f '
            'Buffer size maybe too small! Keeping previous proposal.' % beta)
        cov = None

    return cov
