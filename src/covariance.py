from pyrocko import gf, trace
import numpy as num

import logging
import copy

from beat import heart


logger = logging.getLogger('covariance')


def sub_data_covariance(n, dt, tzero):
    '''
    Calculate sub-covariance matrix without variance.
    :param: n - length of trace/ samples of quadratic Covariance matrix
    :param: dt - time step of samples
    :param: tzero - shortest period of waves in trace
    '''
    return num.exp(- num.abs(num.arange(n)[:, num.newaxis] - \
                              num.arange(n)[num.newaxis, :]) * dt / tzero)


def get_seismic_data_covariances(data_traces, engine, filterer, sample_rate,
                                 arrival_taper, event, targets):
    '''
    Calculate SubCovariance Matrix of trace object following
    Duputel et al. 2012 GJI
    "Uncertainty estimations for seismic source inversions" p. 5

    Cd(i,j) = (Variance of trace)*exp(-abs(ti-tj)/
                                     (shortest period T0 of waves))

       i,j are samples of the seismic trace
    '''

    tzero = 1. / filterer.upper_corner
    dt = 1. / sample_rate
    ataper = arrival_taper
    n = int(num.ceil((num.abs(ataper.a) + ataper.d) / dt))

    csub = sub_data_covariance(n, dt, tzero)

    cov_ds = []
    for i, tr in enumerate(data_traces):
        # assure getting P-wave arrival time
        tmp_target = copy.deepcopy(targets[i])

        tmp_target.codes = (tmp_target.codes[:3] + ('Z',))

        arrival_time = heart.get_phase_arrival_time(
            engine=engine, source=event, target=tmp_target)

        ctrace = tr.chop(
            tmin=tr.tmin,
            tmax=arrival_time - num.abs(ataper.b),
            inplace=False)

        cov_ds.append(num.var(ctrace.ydata, ddof=1) * csub)

    return cov_ds


def get_model_prediction_sensitivity(engine, *args, **kwargs):
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


def get_seis_cov_velocity_models(engine, sources, targets,
                              arrival_taper, filterer, plot=False, n_jobs=1):
    '''
    Calculate model prediction uncertainty matrix with respect to uncertainties
    in the velocity model for station and channel.
    Input:
    :py:class:`gf.Engine` - contains synthetics generation machine
    :py:class:`gf.Targets` - targets to be processed
    :py:class: `heart.ArrivalTaper` - Determines Tapering around Phase Arrival
    '''

    ref_target = copy.deepcopy(targets[0])

    reference_taperer = heart.get_phase_taperer(
        engine,
        sources[0],
        ref_target,
        arrival_taper)

    synths, _ = heart.seis_synthetics(
        engine, sources, targets,
        arrival_taper,
        filterer, nprocs=n_jobs,
        reference_taperer=reference_taperer, plot=plot)

    return num.cov(synths, rowvar=0)


def get_geo_cov_velocity_models(store_superdir, crust_inds, dataset, sources):
    '''
    Calculate model prediction uncertainty matrix with respect to uncertainties
    in the velocity model for geodetic dateset.
    Input:
    store_superdir - geodetic GF directory
    crust_inds - List of indices for respective GF stores
    dataset - :py:class:`IFG`/`DiffIFG`
    sources - List of :py:class:`PsCmpRectangularSource`
    '''

    synths = num.zeros((len(crust_inds), dataset.lons.size))
    for crust_ind in crust_inds:
        disp = heart.geo_layer_synthetics(
            store_superdir, crust_ind,
            lons=dataset.lons,
            lats=dataset.lats,
            sources=sources)
        synths[crust_ind, :] = (
            disp[:, 0] * dataset.los_vector[:, 0] + \
            disp[:, 1] * dataset.los_vector[:, 1] + \
            disp[:, 2] * dataset.los_vector[:, 2]) * \
                dataset.odw

    return num.cov(synths, rowvar=0)
