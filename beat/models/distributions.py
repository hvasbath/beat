from logging import getLogger

import numpy as num
import theano.tensor as tt
from theano import config as tconfig
from theano import shared
from theano.printing import Print

from beat.utility import Counter

logger = getLogger("distributions")


log_2pi = num.log(2 * num.pi)


__all__ = [
    "multivariate_normal",
    "multivariate_normal_chol",
    "hyper_normal",
    "get_hyper_name",
]


def get_hyper_name(dataset):
    return "_".join(("h", dataset.typ))


def multivariate_normal(datasets, weights, hyperparams, residuals):
    """
    Calculate posterior Likelihood of a Multivariate Normal distribution.
    Uses plain inverse of the covariances.
    DEPRECATED! Is currently not being used in beat.
    Can only be executed in a `with model context`.

    Parameters
    ----------
    datasets : list
        of :class:`heart.SeismicDataset` or :class:`heart.GeodeticDataset`
    weights : list
        of :class:`theano.shared`
        Square matrix of the inverse of the covariance matrix as weights
    hyperparams : dict
        of :class:`theano.`
    residual : list or array of model residuals

    Returns
    -------
    array_like
    """
    n_t = len(datasets)

    logpts = tt.zeros((n_t), tconfig.floatX)

    for l, data in enumerate(datasets):
        M = tt.cast(shared(data.samples, name="nsamples", borrow=True), "int16")
        hp_name = get_hyper_name(data)
        norm = M * (2 * hyperparams[hp_name] + log_2pi)
        logpts = tt.set_subtensor(
            logpts[l : l + 1],
            (-0.5)
            * (
                data.covariance.slog_pdet
                + norm
                + (1 / tt.exp(hyperparams[hp_name] * 2))
                * (residuals[l].dot(weights[l]).dot(residuals[l].T))
            ),
        )

    return logpts


def multivariate_normal_chol(
    datasets, weights, hyperparams, residuals, hp_specific=False, sparse=False
):
    """
    Calculate posterior Likelihood of a Multivariate Normal distribution.
    Assumes weights to be the inverse cholesky decomposed lower triangle
    of the Covariance matrix.
    Can only be executed in a `with model context`.

    Parameters
    ----------
    datasets : list
        of :class:`heart.SeismicDataset` or :class:`heart.GeodeticDataset`
    weights : list
        of :class:`theano.shared`
        Square matrix of the inverse of the lower triangular matrix of a
        cholesky decomposed covariance matrix
    hyperparams : dict
        of :class:`theano.`
    residual : list or array of model residuals
    hp_specific : boolean
        if true, the hyperparameters have to be arrays size equal to
        the number of datasets, if false size: 1.
    sparse : boolean
        if the weight matrixes are sparse, this option may be set to speed
        up the calculation, Note: the matrix need to be more than 60%
        sparse to result in a speedup, e.g. identity matrix

    Returns
    -------
    array_like

    Notes
    -----
    adapted from https://www.quora.com/What-is-the-role-of-the-Cholesky-decomposition-in-finding-multivariate-normal-PDF
    """
    if sparse:
        import theano.sparse as ts

        dot = ts.dot
    else:
        dot = tt.dot

    n_t = len(datasets)
    logpts = tt.zeros((n_t), tconfig.floatX)
    count = Counter()

    for l, data in enumerate(datasets):
        M = tt.cast(shared(data.samples, name="nsamples", borrow=True), "int16")
        hp_name = get_hyper_name(data)

        if hp_specific:
            hp = hyperparams[hp_name][count(hp_name)]
        else:
            hp = hyperparams[hp_name]

        tmp = dot(weights[l], (residuals[l]))
        norm = M * (2 * hp + log_2pi)
        logpts = tt.set_subtensor(
            logpts[l : l + 1],
            (-0.5)
            * (
                data.covariance.slog_pdet
                + norm
                + (1 / tt.exp(hp * 2)) * (tt.dot(tmp, tmp))
            ),
        )

    return logpts


def cumulative_normal(x, s=num.sqrt(2)):
    """
    Cumulative distribution function for the standard normal distribution
    """
    return 0.5 + 0.5 * tt.erf(x / s)


def polarity_llk(obs_polarities, syn_amplitudes, gamma, sigma):
    """
    Polarity likelihood based on cumulative normal distribution

    Parameters
    ----------
    obs_polarities : float or array_like
        observed polarities of first motions of seismic phase
    syn_amplitudes : float or array_like
        synthetic amplitudes of seismic phase
    gamma : probability of correctness of polarity reading, data-error
    sigma : modelling error (mostly 1-d velocity model) of amplitudes

    Notes
    -----
    Weber, Z., 2018, Probabilistic joint inversion of waveforms and polarity
        data for double-couple focal mechanisms of local earthquakes,
        GJI, eq. 6, 7
    """
    p_i = gamma + (1 - 2.0 * gamma) * cumulative_normal(syn_amplitudes / sigma)
    llks = ((1.0 + obs_polarities) / 2.0) * tt.log(p_i) + (
        (1.0 - obs_polarities) / 2.0
    ) * tt.log(1.0 - p_i)
    return llks


def hyper_normal(datasets, hyperparams, llks, hp_specific=False):
    """
    Calculate posterior Likelihood only dependent on hyperparameters.

    Parameters
    ----------
    datasets : list
        of :class:`heart.SeismicDatset` or :class:`heart.GeodeticDataset`
    hyperparams : dict
        of :class:`theano.`
    llks : posterior likelihoods
    hp_specific : boolean
        if true, the hyperparameters have to be arrays size equal to
        the number of datasets, if false size: 1.

    Returns
    -------
    array_like
    """
    n_t = len(datasets)
    logpts = tt.zeros((n_t), tconfig.floatX)
    count = Counter()

    for k, data in enumerate(datasets):
        M = data.samples
        hp_name = get_hyper_name(data)
        #        print('hypername', hp_name)
        if hp_specific:
            idx = count(hp_name)
            #            print 'idx', idx
            hp = hyperparams[hp_name][idx]
        #            Print('all')(hyperparams[hp_name])
        #            hp = Print('hyperparam %i %s' % (idx, hp_name))(hp)
        else:
            hp = hyperparams[hp_name]

        logpts = tt.set_subtensor(
            logpts[k : k + 1],
            (-0.5)
            * (
                data.covariance.slog_pdet
                + (M * 2 * hp)
                + (1 / tt.exp(hp * 2)) * llks[k]
            ),
        )

    return logpts


def cartesian_from_polar(phi, theta):
    """
    Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
        (phi-longitude, theta-latitude)
    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    x = num.sin(theta) * num.cos(phi)
    y = num.sin(theta) * num.sin(phi)
    z = num.cos(theta)
    return num.array([x, y, z])


def vonmises_fisher(lats, lons, lats0, lons0, sigma=1.0):
    """
    Von-Mises Fisher distribution function.

    Parameters
    ----------
    lats : float or array_like
        Spherical-polar latitude [deg][-pi/2 pi/2] to evaluate function at.
    lons : float or array_like
        Spherical-polar longitude [deg][-pi pi] to evaluate function at
    lats0 : float or array_like
        latitude [deg] at the center of the distribution (estimated values)
    lons0 : float or array_like
        longitude [deg] at the center of the distribution (estimated values)
    sigma : float
        Width of the distribution.

    Returns
    -------
    float or array_like
        log-probability of the VonMises-Fisher distribution.

    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
        modified from: https://github.com/williamjameshandley/spherical_kde
    """

    def logsinh(x):
        """Compute log(sinh(x)), stably for large x.<
        Parameters
        ----------
        x : float or numpy.array
            argument to evaluate at, must be positive
        Returns
        -------
        float or numpy.array
            log(sinh(x))
        """
        if num.any(x < 0):
            raise ValueError("logsinh only valid for positive arguments")
        return x + num.log(1.0 - num.exp(-2.0 * x)) - num.log(2.0)

    # transform to [0-pi, 0-2pi]
    lats_t = 90.0 - lats
    lons_t = num.mod(lons, 360.0)
    lats0_t = 90.0 - lats0
    lons0_t = num.mod(lons0, 360.0)

    x = cartesian_from_polar(phi=num.deg2rad(lons_t), theta=num.deg2rad(lats_t))
    x0 = cartesian_from_polar(phi=num.deg2rad(lons0_t), theta=num.deg2rad(lats0_t))

    norm = -num.log(4.0 * num.pi * sigma**2) - logsinh(1.0 / sigma**2)
    return norm + num.tensordot(x, x0, axes=[[0], [0]]) / sigma**2


def vonmises_std(lons, lats):
    """
    Von-Mises sample standard deviation.

    Parameters
    ----------
    phi, theta : array-like
        Spherical-polar coordinate samples to compute mean from.

    Returns
    -------
        solution for
        ..math:: 1/tanh(x) - 1/x = R,
        where
        ..math:: R = || \sum_i^N x_i || / N

    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
        but re-parameterised for sigma rather than kappa.
    modidied from: https://github.com/williamjameshandley/spherical_kde
    """
    from scipy.optimize import brentq

    x = cartesian_from_polar(phi=num.deg2rad(lons), theta=num.deg2rad(lats))
    S = num.sum(x, axis=-1)
    R = S.dot(S) ** 0.5 / x.shape[-1]

    def f(s):
        return 1.0 / num.tanh(s) - 1.0 / s - R

    logger.debug("Estimating VonMises std ...")
    kappa = brentq(f, 1e-8, 1e8)
    sigma = kappa**-0.5
    return sigma
