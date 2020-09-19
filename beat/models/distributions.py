from logging import getLogger

import numpy as num

import theano.tensor as tt
from theano import shared
from theano.printing import Print
from theano import config as tconfig

from beat.utility import Counter


logger = getLogger('distributions')


log_2pi = num.log(2 * num.pi)


__all__ = [
    'multivariate_normal',
    'multivariate_normal_chol',
    'hyper_normal',
    'get_hyper_name']


def get_hyper_name(dataset):
    return '_'.join(('h', dataset.typ))


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
        M = tt.cast(shared(
            data.samples, name='nsamples', borrow=True), 'int16')
        hp_name = get_hyper_name(data)
        norm = (M * (2 * hyperparams[hp_name] + log_2pi))
        logpts = tt.set_subtensor(
            logpts[l:l + 1],
            (-0.5) * (
                data.covariance.slog_pdet +
                norm +
                (1 / tt.exp(hyperparams[hp_name] * 2)) *
                (residuals[l].dot(weights[l]).dot(residuals[l].T))))

    return logpts


def multivariate_normal_chol(
        datasets, weights, hyperparams, residuals, hp_specific=False,
        sparse=False):
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
        M = tt.cast(shared(
            data.samples, name='nsamples', borrow=True), 'int16')
        hp_name = get_hyper_name(data)

        if hp_specific:
            hp = hyperparams[hp_name][count(hp_name)]
        else:
            hp = hyperparams[hp_name]

        tmp = dot(weights[l], (residuals[l]))
        norm = (M * (2 * hp + log_2pi))
        logpts = tt.set_subtensor(
            logpts[l:l + 1],
            (-0.5) * (
                data.covariance.slog_pdet +
                norm +
                (1 / tt.exp(hp * 2)) *
                (tt.dot(tmp, tmp))))

    return logpts


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
            logpts[k:k + 1],
            (-0.5) * (
                data.covariance.slog_pdet +
                (M * 2 * hp) +
                (1 / tt.exp(hp * 2)) *
                llks[k]))

    return logpts


def vonmises_fisher(lats, lons, lats0, lons0, sigma):
    """
    Von-Mises Fisher distribution function.

    Parameters
    ----------
    lats : float or array_like
        Spherical-polar latitude to evaluate function at.
    lons : float or array_like
        Spherical-polar longitude to evaluate function at
    lats0 : float or array_like
        latitude at the center of the distribution.
    lons0 : float or array_like
        longitude at the center of the distribution.
    sigma : float
        Width of the distribution.

    Returns
    -------
    float or array_like
        log-probability of the VonMises-Fisher distribution.

    Notes
    -----
    (phi-longitude, theta-latitude)
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
    """
    from pyrocko.orthodrome import latlon_to_xyz

    def logsinh(x):
        """ Compute log(sinh(x)), stably for large x.
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
        return x + num.log(1. - num.exp(-2. * x)) - num.log(2.)

    latlons = num.vstack([lats, lons]).T
    latlons0 = num.vstack([lats0, lons0]).T

    x = latlon_to_xyz(latlons)
    x0 = latlon_to_xyz(latlons0)
    #x = cartesian_from_polar(phi, theta)
    #x0 = cartesian_from_polar(phi0, theta0)
    norm = -num.log(4. * num.pi * sigma ** 2) - logsinh(1. / sigma ** 2)
    return norm + num.tensordot(x, x0, axes=[[0], [0]]) / sigma ** 2
