import logging
from time import time

import numpy as num
from pymc3 import Point
from pyrocko import gf, trace
from scipy.linalg import toeplitz
from scipy.spatial import KDTree
from theano import config as tconfig

from beat import heart
from beat.utility import ensure_cov_psd, list2string, running_window_rms, distances


logger = logging.getLogger("covariance")


__all__ = [
    "geodetic_cov_velocity_models",
    "geodetic_cov_velocity_models_pscmp",
    "seismic_cov_velocity_models",
    "SeismicNoiseAnalyser",
]


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
    return num.exp(
        -num.abs(num.arange(n)[:, num.newaxis] - num.arange(n)[num.newaxis, :])
        * (dt / tzero)
    )


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
    "variance": identity_data_covariance,
    "exponential": exponential_data_covariance,
    "import": ones_data_covariance,
    "non-toeplitz": ones_data_covariance,
}


NoiseStructureCatalog2d = {
    "import": ones_data_covariance,
    "non-toeplitz": ones_data_covariance,
}


def available_noise_structures():
    return list(NoiseStructureCatalog.keys())


def available_noise_structures_2d():
    return list(NoiseStructureCatalog2d.keys())


def import_data_covariance(data_trace, arrival_taper, sample_rate, domain="time"):
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

    logger.info("No data-covariance estimation, using imported" " covariances...\n")

    at = arrival_taper
    n_samples = at.nsamples(sample_rate)

    if domain == "spectrum":
        n_samples = trace.nextpow2(n_samples)
        n_samples = int(n_samples // 2) + 1

    if data_trace.covariance is None:
        logger.warn("No data covariance given/estimated! " "Setting default: eye")
        return num.eye(n_samples)
    else:
        data_cov = data_trace.covariance.data
        if data_cov.shape[0] != n_samples:
            logger.warn(
                "Imported covariance %i does not agree "
                " with taper samples %i! Using Identity"
                " matrix and mean of variance of imported"
                " covariance matrix!" % (data_cov.shape[0], n_samples)
            )
            data_cov = num.eye(n_samples) * data_cov.diagonal().mean()

        return data_cov


class GeodeticNoiseAnalyser(object):
    """
    Geodetic noise analyser

    Parameters
    ----------
    structure :  string
        either, import, variance, non-toeplitz
    events : list
        of :class:`pyrocko.meta.Event` reference event(s) from catalog
    """

    def __init__(
        self,
        config,
        events=None,
    ):

        avail = available_noise_structures_2d()

        if config.structure not in avail:
            raise AttributeError(
                'Selected noise structure "%s" not supported! Implemented'
                " noise structures: %s" % (structure, list2string(avail))
            )

        self.events = events
        self.config = config

    def get_structure(self, dataset):
        return NoiseStructureCatalog2d[self.config.structure](dataset.ncoords)

    def do_import(self, dataset):
        if dataset.covariance.data is not None:
            return dataset.covariance.data
        else:
            raise ValueError(
                "Data covariance for dataset %s needs to be defined!" % dataset.name
            )

    def do_non_toeplitz(self, dataset, result):

        if dataset.typ == "SAR":
            dataset.update_local_coords(self.events[0])
            coords = num.vstack([dataset.east_shifts, dataset.north_shifts]).T

            scaling = non_toeplitz_covariance_2d(
                coords, result.processed_res, max_dist_perc=self.config.max_dist_perc
            )
        else:
            scaling = dataset.covariance.data

        if num.isnan(scaling).any():
            raise ValueError(
                "Estimated Non-Toeplitz covariance matrix for dataset %s contains Nan! "
                "Please increase 'max_dist_perc'!" % dataset.name
            )

        return scaling

    def get_data_covariance(self, dataset, result=None):
        """
        Estimated data covariances of seismic traces

        Parameters
        ----------
        datasets
        results

        Returns
        -------
        :class:`numpy.ndarray`
        """

        covariance_structure = self.get_structure(dataset)

        if self.config.structure == "import":
            scaling = self.do_import(dataset)
        elif self.config.structure == "non-toeplitz":
            scaling = self.do_non_toeplitz(dataset, result)

        return ensure_cov_psd(scaling * covariance_structure)


class SeismicNoiseAnalyser(object):
    """
    Seismic noise analyser

    Parameters
    ----------
    structure : string
        either, variance, exponential, import, non-toeplitz
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
        self,
        structure="variance",
        pre_arrival_time=5.0,
        engine=None,
        events=None,
        sources=None,
        chop_bounds=["b", "c"],
    ):

        avail = available_noise_structures()
        if structure not in avail:
            raise AttributeError(
                'Selected noise structure "%s" not supported! Implemented'
                " noise structures: %s" % (structure, list2string(avail))
            )

        self.events = events
        self.engine = engine
        self.sources = sources
        self.pre_arrival_time = pre_arrival_time
        self.structure = structure
        self.chop_bounds = chop_bounds

    def get_structure(self, wmap, chop_bounds=None):

        if chop_bounds is None:
            chop_bounds = self.chop_bounds

        _, fmax = wmap.get_taper_frequencies()
        tzero = 1.0 / fmax

        if wmap.config.domain == "spectrum":
            n = wmap.get_nsamples_spectrum(chop_bounds=chop_bounds, pad_to_pow2=True)
            dsample = wmap.get_deltaf(chop_bounds=chop_bounds, pad_to_pow2=True)
        else:
            n = wmap.get_nsamples_time(chop_bounds)
            dsample = wmap.deltat

        return NoiseStructureCatalog[self.structure](n, dsample, tzero)

    def do_import(self, wmap):

        scalings = []
        for tr, target in zip(wmap.datasets, wmap.targets):
            scaling = import_data_covariance(
                tr,
                arrival_taper=wmap.config.arrival_taper,
                sample_rate=1.0 / wmap.deltat,
                domain=wmap.config.domain,
            )
            scalings.append(scaling)

        return scalings

    def do_non_toeplitz(self, wmap, results):

        if results is None:
            ValueError(
                "Results need(s) to be given for non-toeplitz" " covariance estimates!"
            )
        else:
            scalings = []
            for result in results:
                residual = result.processed_res.get_ydata()
                window_size = residual.size // 5
                if window_size == 0:
                    raise ValueError(
                        "Length of trace too short! Please widen taper in time"
                        " domain or frequency bands in spectral domain."
                    )
                scaling = non_toeplitz_covariance(residual, window_size=window_size)
                scalings.append(scaling)

            return scalings

    def do_variance_estimate(self, wmap, chop_bounds=None):

        filterer = wmap.config.filterer
        scalings = []

        for i, (tr, target) in enumerate(zip(wmap.datasets, wmap.targets)):
            wavename = None  # None uses first tabulated phase
            arrival_time = heart.get_phase_arrival_time(
                engine=self.engine,
                source=self.events[wmap.config.event_idx],
                target=target,
                wavename=wavename,
            )

            if arrival_time < tr.tmin:
                logger.warning(
                    "no data for variance estimation on pre-P arrival"
                    " in wavemap %s, for trace %s!"
                    % (wmap._mapid, list2string(tr.nslc_id))
                )
                logger.info('Using reference arrival "%s" instead!' % wmap.name)
                arrival_time = heart.get_phase_arrival_time(
                    engine=self.engine,
                    source=self.events[wmap.config.event_idx],
                    target=target,
                    wavename=wmap.name,
                )

            if filterer:
                ctrace = tr.copy()
                # apply all the filters
                for filt in filterer:
                    filt.apply(ctrace)

            ctrace = ctrace.chop(
                tmin=tr.tmin, tmax=arrival_time - self.pre_arrival_time
            )

            nslc_id_str = list2string(ctrace.nslc_id)

            if wmap.config.domain == "spectrum":
                valid_spectrum_indices = wmap.get_valid_spectrum_indices(
                    chop_bounds=chop_bounds, pad_to_pow2=True
                )
                data = heart.fft_transforms(
                    [ctrace],
                    valid_spectrum_indices,
                    outmode="stacked_traces",
                    pad_to_pow2=True,
                )[0].get_ydata()
            else:
                data = ctrace.get_ydata()

            if data.size == 0:
                raise ValueError(
                    "Trace %s contains no pre-P arrival data! Please either "
                    "remove/blacklist or make sure data contains times before"
                    " the P arrival time!" % nslc_id_str
                )

            scaling = num.nanvar(data)
            if num.isfinite(scaling).all():
                logger.info("Variance estimate of %s = %g" % (nslc_id_str, scaling))
                scalings.append(scaling)
            else:
                raise ValueError(
                    "Pre P-trace of %s contains Inf or" " NaN!" % nslc_id_str
                )

        return scalings

    def get_data_covariances(self, wmap, sample_rate, results=None, chop_bounds=None):
        """
        Estimated data covariances of seismic traces

        Parameters
        ----------
        wmap : :class:`beat.WaveformMapping`
        results
        sample_rate : float
            sampling rate of data_traces and GreensFunction stores

        Returns
        -------
        :class:`numpy.ndarray`
        """

        covariance_structure = self.get_structure(wmap, chop_bounds)

        if self.structure == "import":
            scalings = self.do_import(wmap)
        elif self.structure == "non-toeplitz":
            scalings = self.do_non_toeplitz(wmap, results)
        else:
            scalings = self.do_variance_estimate(wmap, chop_bounds=chop_bounds)

        cov_ds = []
        for scaling in scalings:
            cov_d = ensure_cov_psd(scaling * covariance_structure)
            cov_ds.append(cov_d)

        return cov_ds


def model_prediction_sensitivity(engine, *args, **kwargs):
    """
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
    """

    if len(args) not in (0, 1, 2, 3):
        raise gf.BadRequest("invalid arguments")

    if len(args) == 2:
        kwargs["request"] = args[0]
        kwargs["source_params"] = args[1]

    elif len(args) == 3:
        kwargs.update(gf.Request.args2kwargs(args[0:1]))
        kwargs["source_params"] = args[2]

    request = kwargs.pop("request", None)
    nprocs = kwargs.pop("nprocs", 1)
    source_params = kwargs.pop("source_params", None)
    h = kwargs.pop("h", None)

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
            print(param, "with h = ", h[par_count])
            calc_source_p2h = ref_source.clone()
            calc_source_ph = ref_source.clone()
            calc_source_mh = ref_source.clone()
            calc_source_m2h = ref_source.clone()

            setattr(calc_source_p2h, param, ref_source[param] + (2 * h[par_count]))
            setattr(calc_source_ph, param, ref_source[param] + (h[par_count]))
            setattr(calc_source_mh, param, ref_source[param] - (h[par_count]))
            setattr(calc_source_m2h, param, ref_source[param] - (2 * h[par_count]))

            calc_sources = [
                calc_source_p2h,
                calc_source_ph,
                calc_source_mh,
                calc_source_m2h,
            ]

            response = engine.process(
                sources=calc_sources, targets=request.targets, nprocs=nprocs
            )

            for k in range(len(request.targets)):
                # zero padding if necessary
                trc_lengths = num.array(
                    [
                        len(response.results_list[i][k].trace.data)
                        for i in range(len(response.results_list))
                    ]
                )
                Id = num.where(trc_lengths != trc_lengths.max())

                for l in Id[0]:
                    response.results_list[l][k].trace.data = num.concatenate(
                        (
                            response.results_list[l][k].trace.data,
                            num.zeros(trc_lengths.max() - trc_lengths[l]),
                        )
                    )

                # calculate numerical partial derivative for
                # each source and target
                sensitivity_param_list[par_count][k] = sensitivity_param_list[
                    par_count
                ][k] + (
                    -response.results_list[0][k].trace.data
                    + 8 * response.results_list[1][k].trace.data
                    - 8 * response.results_list[2][k].trace.data
                    + response.results_list[3][k].trace.data
                ) / (
                    12 * h[par_count]
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
                location=param,
            )

        par_count = par_count + 1

    return sensitivity_param_trcs


def seismic_cov_velocity_models(
    engine,
    sources,
    targets,
    arrival_taper,
    arrival_time,
    wavename,
    filterer,
    plot=False,
    n_jobs=1,
    chop_bounds=["b", "c"],
):
    """
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
    filterer : list
        of :class:`heart.Filter` determining the filtering corner frequencies
        of various filters
    plot : boolean
        open snuffler and browse traces if True
    n_jobs : int
        number of processors to be used for calculation

    Returns
    -------
    :class:`numpy.ndarray` with Covariance due to velocity model uncertainties
    """

    arrival_times = num.ones(len(targets), dtype="float64") * arrival_time

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
        outmode="array",
        chop_bounds=chop_bounds,
    )

    t1 = time()
    logger.debug("Trace generation time %f" % (t1 - t0))

    return num.cov(synths, rowvar=0)


def geodetic_cov_velocity_models(
    engine, sources, targets, dataset, plot=False, event=None, n_jobs=1
):
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
        engine=engine, targets=targets, sources=sources, outmode="stacked_arrays"
    )
    t1 = time()
    logger.debug("Synthetics generation time %f" % (t1 - t0))

    synths = num.zeros((len(targets), dataset.samples))
    for i, disp in enumerate(displacements):
        synths[i, :] = (
            disp[:, 0] * dataset.los_vector[:, 0]
            + disp[:, 1] * dataset.los_vector[:, 1]
            + disp[:, 2] * dataset.los_vector[:, 2]
        ) * dataset.odw

    if plot:
        from matplotlib import pyplot as plt

        indexes = dataset.get_distances_to_event(event).argsort()  # noqa
        ax = plt.axes()
        im = ax.matshow(synths)  # [:, indexes])
        plt.colorbar(im)
        plt.show()

    return num.cov(synths, rowvar=0)


def geodetic_cov_velocity_models_pscmp(store_superdir, crust_inds, target, sources):
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
            store_superdir,
            crust_ind,
            lons=target.lons,
            lats=target.lats,
            sources=sources,
        )
        synths[crust_ind, :] = (
            disp[:, 0] * target.los_vector[:, 0]
            + disp[:, 1] * target.los_vector[:, 1]
            + disp[:, 2] * target.los_vector[:, 2]
        ) * target.odw

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
    toeplitz : :class:`numpy.ndarray` 1-d, (size data)
    stds : :class:`numpy.ndarray` 1-d, size data
        of running windows
    """
    stds = running_window_rms(data, window_size=window_size, mode="same")
    coeffs = autocovariance(data / stds)
    return toeplitz(coeffs), stds


def non_toeplitz_covariance(data, window_size):
    """
    Get scaled non- Toeplitz covariance matrix, which may be able to account
    for non-stationary data-errors. For 1-d data.

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


def k_nearest_neighbor_rms(coords, data, k=None, max_dist_perc=0.2):
    """
    Calculate running rms on irregular sampled 2d spatial data.

    Parameters
    ----------
    coords : :class:`numpy.ndarray` 2-d, (size data, n coords-dims)
        containing spatial coordinates (east_shifts, north_shifts)
    data : :class:`numpy.ndarray` 1-d, (size data)
        containing values of physical quantity
    k : int
        taking k - nearest neighbors for std estimation
    max_dist_perc : float
        max distance [decimal percent] to select as nearest neighbors
    """

    if k and max_dist_perc:
        raise ValueError("Either k or max_dist_perc should be defined!")

    kdtree = KDTree(coords, leafsize=1)

    dists = distances(coords, coords)
    r = dists.max() * max_dist_perc

    logger.debug("Nearest neighbor distance is: %f", r)

    stds = []
    for point in coords:
        if k is not None:
            _, idxs = kdtree.query(point, k=k)
        elif r is not None:
            idxs = kdtree.query_ball_point(point, r=r)
        else:
            raise ValueError()

        stds.append(num.std(data[idxs], ddof=1))

    return num.array(stds)


def toeplitz_covariance_2d(coords, data, max_dist_perc=0.2):
    """
    Get Toeplitz banded matrix for given 2d data.

    Returns
    -------
    toeplitz : :class:`numpy.ndarray` 2-d, (size data, size data)
    stds : :class:`numpy.ndarray` 1-d, size data
        of running windows
    max_dist_perc : float
        max distance [decimal percent] to select as nearest neighbors
    """
    stds = k_nearest_neighbor_rms(coords=coords, data=data, max_dist_perc=max_dist_perc)
    coeffs = autocovariance(data / stds)
    return toeplitz(coeffs), stds


def non_toeplitz_covariance_2d(coords, data, max_dist_perc):
    """
    Get scaled non- Toeplitz covariance matrix, which may be able to account
    for non-stationary data-errors. For 2-d geospatial data.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        of data to estimate non-stationary error matrix for
    max_dist_perc : float
        max distance [decimal percent] to select as nearest neighbors

    Returns
    -------
    :class:`numpy.ndarray` (size data, size data)
    """
    toeplitz, stds = toeplitz_covariance_2d(coords, data, max_dist_perc)
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

    like_idx = lij.ordering["like"].list_ind
    weights = num.array([lpoint[like_idx] for lpoint, _ in buffer])
    temp_weights = num.exp((weights - weights.max())).ravel()
    norm_weights = temp_weights / num.sum(temp_weights)

    cov = num.cov(population_array, aweights=norm_weights, bias=False, rowvar=0)

    cov = ensure_cov_psd(cov)
    if num.isnan(cov).any() or num.isinf(cov).any():
        logger.warn(
            "Proposal covariances contain Inf or NaN! "
            "For chain with beta: %f "
            "Buffer size maybe too small! Keeping previous proposal." % beta
        )
        cov = None

    return cov
