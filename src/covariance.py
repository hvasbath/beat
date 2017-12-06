from pyrocko import gf, trace
import numpy as num
from time import time

import logging
import copy

from beat import heart


logger = logging.getLogger('covariance')


__all__ = [
    'geodetic_cov_velocity_models',
    'geodetic_cov_velocity_models_pscmp',
    'seismic_cov_velocity_models',
    'seismic_data_covariance']


def sub_data_covariance(n, dt, tzero):
    '''
    Calculate sub-covariance matrix without variance.

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
    '''
    return num.exp(- num.abs(num.arange(n)[:, num.newaxis] - \
                              num.arange(n)[num.newaxis, :]) * dt / tzero)


def seismic_data_covariance(data_traces, engine, filterer, sample_rate,
                                 arrival_taper, event, targets):
    '''
    Calculate SubCovariance Matrix of trace object following
    Duputel et al. 2012 GJI
    "Uncertainty estimations for seismic source inversions" p. 5

    Parameters
    ----------
    data_traces : list
        of :class:`pyrocko.trace.Trace` containing observed data
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        processing object for synthetics calculation
    filterer : :class:`heart.Filter`
        determines the bandpass-filtering corner frequencies
    sample_rate : float
        sampling rate of data_traces and GreensFunction stores
    arrival_taper : :class: `heart.ArrivalTaper`
        determines tapering around phase Arrival
    event : :class:`pyrocko.meta.Event`
        reference event from catalog
    targets : list
        of :class:`pyrocko.gf.seismosizer.Targets`

    Returns
    -------
    :class:`numpy.ndarray`

    Notes
    -----
    Cd(i,j) = (Variance of trace)*exp(-abs(ti-tj)/
                                     (shortest period T0 of waves))

       i,j are samples of the seismic trace
    '''
    wavename = 'any_P'   # hardcode here, want always pre P time
    tzero = 1. / filterer.upper_corner
    dt = 1. / sample_rate
    ataper = arrival_taper
    n = int(num.ceil((num.abs(ataper.a) + ataper.d) / dt))

    csub = sub_data_covariance(n, dt, tzero)

    cov_ds = []
    for tr, target in zip(data_traces, targets):
        arrival_time = heart.get_phase_arrival_time(
            engine=engine, source=event,
            target=target, wavename=wavename)

        ctrace = tr.chop(
            tmin=tr.tmin,
            tmax=arrival_time - num.abs(ataper.b),
            inplace=False)

        cov_ds.append(num.var(ctrace.ydata, ddof=1) * csub)

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

    for i in xrange(len(source_params)):
        sensitivity_param_list.append([0] * len(request.targets))
        sensitivity_param_trcs.append([0] * len(request.targets))

    for ref_source in request.sources:
        par_count = 0
        for param in source_params:
            print param, 'with h = ', h[par_count]
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

            for k in xrange(len(request.targets)):
                # zero padding if necessary
                trc_lengths = num.array(
                    [len(response.results_list[i][k].trace.data) for i in \
                                        range(len(response.results_list))])
                Id = num.where(trc_lengths != trc_lengths.max())

                for l in Id[0]:
                    response.results_list[l][k].trace.data = num.concatenate(
                            (response.results_list[l][k].trace.data,
                             num.zeros(trc_lengths.max() - trc_lengths[l])))

                # calculate numerical partial derivative for
                # each source and target
                sensitivity_param_list[par_count][k] = (
                        sensitivity_param_list[par_count][k] + (\
                            - response.results_list[0][k].trace.data + \
                            8 * response.results_list[1][k].trace.data - \
                            8 * response.results_list[2][k].trace.data + \
                                response.results_list[3][k].trace.data) / \
                            (12 * h[par_count])
                                                       )

            par_count = par_count + 1

    # form traces from sensitivities
    par_count = 0
    for param in source_params:
        for k in xrange(len(request.targets)):
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


def seismic_cov_velocity_models(engine, sources, targets,
                  arrival_taper, wavename, filterer, plot=False, n_jobs=1):
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

    ref_target = copy.deepcopy(targets[0])

    reference_taperer = heart.get_phase_taperer(
        engine,
        sources[0],
        wavename=wavename,
        target=ref_target,
        arrival_taper=arrival_taper)

    t0 = time()
    synths, _ = heart.seis_synthetics(
        engine=engine, sources=sources, targets=targets,
        arrival_taper=arrival_taper, wavename=wavename,
        filterer=filterer, nprocs=n_jobs,
        reference_taperer=reference_taperer, plot=plot,
        pre_stack_cut=True, outmode='stacked_traces')
    t1 = time()
    logger.debug('Trace generation time %f' % (t1 - t0))

    return num.cov(synths, rowvar=0)


def geodetic_cov_velocity_models(
    engine, sources, targets, dataset, plot=False, n_jobs=1):
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

    Returns
    -------
    :class:`numpy.ndarray` with Covariance due to velocity model uncertainties
    """
    t0 = time()
    displacements = heart.geo_synthetics(
        engine=engine,
        targets=targets,
        sources=sources,
        plot=plot,
        outmode='stacked_arrays')
    t1 = time()
    logger.debug('Synthetics generation time %f' % (t1 - t0))

    synths = num.zeros((len(targets), dataset.samples))
    for i, disp in enumerate(displacements):
        synths[i, :] = (
            disp[:, 0] * dataset.los_vector[:, 0] + \
            disp[:, 1] * dataset.los_vector[:, 1] + \
            disp[:, 2] * dataset.los_vector[:, 2]) * dataset.odw

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
            disp[:, 0] * target.los_vector[:, 0] + \
            disp[:, 1] * target.los_vector[:, 1] + \
            disp[:, 2] * target.los_vector[:, 2]) * \
                target.odw

    return num.cov(synths, rowvar=0)
