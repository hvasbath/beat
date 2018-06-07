import logging
import unittest

from beat.models import multivariate_normal_chol, log_2pi, multivariate_normal
from beat.info import project_root
from beat.heart import SeismicDataset, Covariance

from pymc3.distributions import MvNormal
from pymc3 import Model

from theano import shared, function
from theano.printing import Print
from theano import tensor as tt
from theano import sparse as ts

from pyrocko import util

import numpy as num
from numpy.testing import assert_allclose

import scipy
from time import time

logger = logging.getLogger('test_models')

n_datasets = 2
n_samples = 10


def normal_logpdf_cov(value, mu, cov):
    return scipy.stats.multivariate_normal.logpdf(value, mu, cov).sum()


def normal_logpdf_chol(value, mu, chol):
    return normal_logpdf_cov(value, mu, num.dot(chol, chol.T)).sum()


def multivariate_normal_nohypers(datasets, weights, hyperparams, residuals):
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

    logpts = tt.zeros((n_t), 'float64')

    for l, data in enumerate(datasets):
        M = tt.cast(shared(
            data.samples, name='nsamples', borrow=True), 'int16')
 #       print data.samples
  #      residuals1 = Print('residuals_theano')(residuals[l, :])
    #    hp_name = '_'.join(('h', data.typ))
        maha = residuals[l].dot(weights[l]).dot(residuals[l].T)
#        maha = Print('maha1')(maha)
        slogpdet = Print('theano logpdet')(data.covariance.slog_pdet)
        logpts = tt.set_subtensor(
            logpts[l:l + 1],
            (-0.5) * (
                M * log_2pi + slogpdet + maha
  #              (M * 2 * hyperparams[hp_name]) +
  #              (1 / tt.exp(hyperparams[hp_name] * 2)) *
                ))

    return logpts


def generate_toydata(n_datasets, n_samples):
    datasets = []
    synthetics = []
    for d in range(n_datasets):
        a = num.atleast_2d(num.random.rand(n_samples))
        C = a * a.T + num.eye(n_samples) * 0.001
        kwargs = dict(
            ydata=num.random.rand(n_samples),
            tmin=0.,
            deltat=0.5,
            channel='T')
        ds = SeismicDataset(**kwargs)
        ds.covariance = Covariance(data=C)
        ds.set_wavename('any_P')
        datasets.append(ds)
        synthetics.append(num.random.rand(n_samples))

    return datasets, synthetics


def make_weights(datasets, wtype, make_shared=False, sparse=False):
    weights = []
    for ds in datasets:
        if wtype == 'ichol':
            w = num.linalg.inv(ds.covariance.chol)
            # print ds.covariance.chol_inverse
        elif wtype == 'icov_chol':
            w = ds.covariance.chol_inverse
            # print w

        elif wtype == 'icov':
            w = ds.covariance.inverse
        else:
            raise NotImplementedError('wtype not implemented!')

        if make_shared:
            sw = shared(w)
            if sparse:
                sw = ts.csc_from_dense(sw)

            weights.append(sw)
        else:
            weights.append(w)

    return weights


def get_bulk_weights(weights):
    return tt.concatenate(
        [C.reshape((1, n_samples, n_samples)) for C in weights], axis=0)


class TestModels(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.beatpath = project_root

        self.hyperparams = {'h_any_P_T': shared(0.)}

        self.datasets, self.synthetics = generate_toydata(
            n_datasets, n_samples)
        self.residuals = num.vstack(
            [data.ydata - self.synthetics[i]
                for i, data in enumerate(self.datasets)])

    def test_reference_llk(self):

        res = tt.matrix('residuals')

        icov_weights_numpy = make_weights(
            self.datasets, 'icov', False)

        llk_normal = multivariate_normal_nohypers(
            self.datasets, icov_weights_numpy,
            self.hyperparams, res)
        fnorm = function([res], llk_normal)

        c = fnorm(self.residuals)

        d = num.zeros(n_datasets)
        for i, data in enumerate(self.datasets):
            psd = scipy.stats._multivariate._PSD(data.covariance.data)
            logpdet = data.covariance.log_pdet

            assert_allclose(logpdet, psd.log_pdet, rtol=0., atol=1e-6)
            assert_allclose(
                psd.pinv, data.covariance.inverse, rtol=0., atol=1e-6)

            d[i] = normal_logpdf_cov(
                data.ydata, self.synthetics[i], data.covariance.data)

        assert_allclose(d, c, rtol=0., atol=1e-6)

    def test_mvn_cholesky(self):
        res = tt.matrix('residuals')

        ichol_weights = make_weights(self.datasets, 'ichol', True)
        icov_chol_weights = make_weights(self.datasets, 'icov_chol', True)

        icov_weights = make_weights(self.datasets, 'icov', True)

        llk_ichol = multivariate_normal_chol(
            self.datasets, ichol_weights,
            self.hyperparams, res, hp_specific=False)
        fichol = function([res], llk_ichol)

        llk_icov_chol = multivariate_normal_chol(
            self.datasets, icov_chol_weights,
            self.hyperparams, res, hp_specific=False)
        ficov_chol = function([res], llk_icov_chol)

        llk_normal = multivariate_normal(
            self.datasets, icov_weights,
            self.hyperparams, res)
        fnorm = function([res], llk_normal)

        llk_normal_nohyp = multivariate_normal_nohypers(
            self.datasets, icov_weights,
            self.hyperparams, res)
        fnorm_nohyp = function([res], llk_normal_nohyp)

        t0 = time()
        a = fichol(self.residuals)
        t1 = time()
        b = ficov_chol(self.residuals)
        t2 = time()
        c = fnorm(self.residuals)
        t3 = time()
        d = fnorm_nohyp(self.residuals)
        t4 = time()

        logger.info('Ichol %f [s]' % (t1 - t0))
        logger.info('Icov_chol %f [s]' % (t2 - t1))
        logger.info('Icov %f [s]' % (t3 - t2))
        logger.info('Icov_nohyp %f [s]' % (t4 - t3))

        assert_allclose(a, c, rtol=0., atol=1e-6)
        assert_allclose(b, c, rtol=0., atol=1e-6)
        assert_allclose(a, b, rtol=0., atol=1e-6)
        assert_allclose(d, c, rtol=0., atol=1e-6)
        assert_allclose(d, b, rtol=0., atol=1e-6)

    def mtest_sparse(self):

        res = tt.matrix('residuals')

        ichol_weights = make_weights(self.datasets, 'ichol', True, sparse=True)
        icov_chol_weights = make_weights(
            self.datasets, 'icov_chol', True, sparse=True)

        llk_ichol = multivariate_normal_chol(
            self.datasets, ichol_weights,
            self.hyperparams, res, hp_specific=False, sparse=True)
        fichol = function([res], llk_ichol)

        llk_icov_chol = multivariate_normal_chol(
            self.datasets, icov_chol_weights,
            self.hyperparams, res, hp_specific=False, sparse=True)
        ficov_chol = function([res], llk_icov_chol)

        t0 = time()
        a = fichol(self.residuals)
        t1 = time()
        b = ficov_chol(self.residuals)
        t2 = time()

        logger.info('Sparse Ichol %f [s]' % (t1 - t0))
        logger.info('Sparse Icov_chol %f [s]' % (t2 - t1))

        assert_allclose(a, b, rtol=0., atol=1e-6)

    def mtest_bulk(self):

        def multivariate_normal_bulk_chol(
                bulk_weights, hps, slnfs, residuals, hp_specific=False):

            M = residuals.shape[1]
            tmp = tt.batched_dot(bulk_weights, residuals)
            llk = tt.power(tmp, 2).sum(1)
            return (-0.5) * (
                slnfs + (M * 2 * hps) + (1 / tt.exp(hps * 2)) * (llk))

        res = tt.matrix('residuals')
        ichol_weights = make_weights(self.datasets, 'ichol', True)
        icov_weights = make_weights(self.datasets, 'icov', True)
        icov_chol_weights = make_weights(self.datasets, 'icov_chol', True)

        ichol_bulk_weights = get_bulk_weights(ichol_weights)
        icov_chol_bulk_weights = get_bulk_weights(icov_chol_weights)

        slnfs = tt.concatenate(
            [data.covariance.slnf.reshape((1,)) for data in self.datasets])

        ichol_bulk_llk = multivariate_normal_bulk_chol(
            bulk_weights=ichol_bulk_weights, hps=self.hyperparams['h_any_P_T'],
            slnfs=slnfs, residuals=res)

        icov_chol_bulk_llk = multivariate_normal_bulk_chol(
            bulk_weights=icov_chol_bulk_weights,
            hps=self.hyperparams['h_any_P_T'],
            slnfs=slnfs, residuals=res)

        llk_normal = multivariate_normal(
            self.datasets, icov_weights,
            self.hyperparams, res)

        fnorm = function([res], llk_normal)
        f_bulk_ichol = function([res], ichol_bulk_llk)
        f_bulk_icov_chol = function([res], icov_chol_bulk_llk)

        t0 = time()
        a = f_bulk_ichol(self.residuals)
        t1 = time()
        b = f_bulk_icov_chol(self.residuals)
        t2 = time()
        c = fnorm(self.residuals)

        logger.info('Bulk Ichol %f [s]' % (t1 - t0))
        logger.info('Bulk Icov_chol %f [s]' % (t2 - t1))

        assert_allclose(a, c, rtol=0., atol=1e-6)
        assert_allclose(b, c, rtol=0., atol=1e-6)
        assert_allclose(a, b, rtol=0., atol=1e-6)

    def test_mvn_chol_loopless(self):
        pass


if __name__ == '__main__':

    util.setup_logging('test_models', 'info')
    unittest.main()
