from beat.models.base import Composite
from beat import config as bconfig
from beat.utility import load_objects
from beat.heart import log_determinant

from pymc3 import Deterministic

from theano import tensor as tt
from theano import shared
from theano import config as tconfig

import numpy as num
from logging import getLogger
import os


logger = getLogger('ffo.laplacian')

LOG_2PI = num.log(2. * num.pi)


__all__ = [
    'LaplacianDistributerComposite',
    'get_smoothing_operator']


class LaplacianDistributerComposite(Composite):

    def __init__(self, project_dir, hypers):

        super(LaplacianDistributerComposite, self).__init__()

        self._mode = 'ffo'

        # dummy for hyperparam name
        self.hyperparams[bconfig.hyper_name_laplacian] = None
        self.hierarchicals = None

        self.slip_varnames = bconfig.static_dist_vars
        self.gfpath = os.path.join(
            project_dir, self._mode, bconfig.linear_gf_dir_name)

        self.fault = self.load_fault_geometry()
        self.spatches = shared(self.fault.npatches, borrow=True)
        self._like_name = 'laplacian_like'

        # only one subfault so far, smoothing across and fast-sweep
        # not implemented for more yet
        # TODO scipy.linalg.block_diag of all smoothing operators of subfaults
        self.smoothing_op = \
            self.fault.get_subfault_smoothing_operator(0).astype(
                tconfig.floatX)

        self.sdet_shared_smoothing_op = shared(
            log_determinant(
                self.smoothing_op.T * self.smoothing_op, inverse=False),
            borrow=True)

        self.shared_smoothing_op = shared(self.smoothing_op, borrow=True)

        if hypers:
            self._llks = []
            for varname in self.slip_varnames:
                self._llks.append(shared(
                    num.array([1.]),
                    name='laplacian_llk_%s' % varname,
                    borrow=True))

    def load_fault_geometry(self):
        """
        Load fault-geometry, i.e. discretized patches.

        Returns
        -------
        :class:`ffo.fault.FaultGeometry`
        """
        return load_objects(
            os.path.join(self.gfpath, bconfig.fault_geometry_name))[0]

    def _eval_prior(self, hyperparam, exponent):
        """
        Evaluate model parameter independend part of the smoothness prior.
        """
        return (-0.5) * \
            (-self.sdet_shared_smoothing_op +
             (self.spatches * (LOG_2PI + 2 * hyperparam)) +
             (1. / tt.exp(hyperparam * 2) * exponent))

    def get_formula(self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Get smoothing likelihood formula for the model built. Has to be called
        within a with model context.
        Part of the pymc3 model.

        Parameters
        ----------
        input_rvs : dict
            of :class:`pymc3.distribution.Distribution`
        fixed_rvs : dict
            of :class:`numpy.array` here only dummy
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`
        problem_config : :class:`config.ProblemConfig`
            here it is not used

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        logger.info('Initialising Laplacian smoothing operator ...')

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        hp_name = bconfig.hyper_name_laplacian
        self.input_rvs.update(fixed_rvs)

        logpts = tt.zeros((self.n_t), tconfig.floatX)
        for l, var in enumerate(self.slip_varnames):
            Ls = self.shared_smoothing_op.dot(input_rvs[var])
            exponent = Ls.T.dot(Ls)

            logpts = tt.set_subtensor(
                logpts[l:l + 1],
                self._eval_prior(hyperparams[hp_name], exponent=exponent))

        llk = Deterministic(self._like_name, logpts)
        return llk.sum()

    def update_llks(self, point):
        """
        Update posterior likelihoods (in place) of the composite w.r.t.
        one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        for l, varname in enumerate(self.slip_varnames):
            Ls = self.smoothing_op.dot(point[varname])
            _llk = num.asarray([Ls.T.dot(Ls)])
            self._llks[l].set_value(_llk)

    def get_hyper_formula(self, hyperparams, problem_config):
        """
        Get likelihood formula for the hyper model built. Has to be called
        within a with model context.
        """

        logpts = tt.zeros((self.n_t), tconfig.floatX)
        for k in range(self.n_t):
            logpt = self._eval_prior(
                hyperparams[bconfig.hyper_name_laplacian], self._llks[k])
            logpts = tt.set_subtensor(logpts[k:k + 1], logpt)

        llk = Deterministic(self._like_name, logpts)
        return llk.sum()

    @property
    def n_t(self):
        return len(self.slip_varnames)


def _patch_locations(n_patch_strike, n_patch_dip):
    """
    Determines from patch locations the neighboring patches

    Parameters
    ----------
    n_patch_strike : int
        number of patches in strike direction
    n_patch_dip : int
        number of patches in dip direction

    Returns
    -------
    :class:`numpy.Ndarray`
        (n_patch_strike + n_patch_dip) x 4
    """
    n_patches = n_patch_dip * n_patch_strike

    zeros_strike = num.zeros(n_patch_strike)
    zeros_dip = num.zeros(n_patch_dip)

    dmat = num.ones((n_patches, 4))
    dmat[0:n_patch_strike, 0] = zeros_strike
    dmat[-n_patch_strike:, 1] = zeros_strike
    dmat[0::n_patch_strike, 2] = zeros_dip
    dmat[n_patch_strike - 1::n_patch_strike, 3] = zeros_dip
    return dmat


def get_smoothing_operator(
        n_patch_strike, n_patch_dip, patch_size_strike, patch_size_dip):
    """
    Get second order Laplacian smoothing operator.

    This is beeing used to smooth the slip-distribution in the optimization.

    Parameters
    ----------
    n_patch_strike : int
        number of patches in strike direction
    n_patch_dip : int
        number of patches in dip direction
    patch_size_strike : float
        size of patches along strike-direction [km]
    patch_size_dip : float
        size of patches along dip-direction [km]

    Returns
    -------
    :class:`numpy.Ndarray`
        (n_patch_strike + n_patch_dip) x (n_patch_strike + n_patch_dip)
    """
    n_patches = n_patch_dip * n_patch_strike

    dmat = _patch_locations(
        n_patch_strike=n_patch_strike, n_patch_dip=n_patch_dip)

    smooth_op = num.zeros((n_patches, n_patches))

    delta_l_dip = 1. / (patch_size_dip ** 2)
    delta_l_strike = 1. / (patch_size_strike ** 2)
    deltas = num.array(
        [delta_l_dip, delta_l_dip, delta_l_strike, delta_l_strike])

    for i in range(n_patches):
        flags = dmat[i, :]

        smooth_op[i, i] = -1 * flags.dot(deltas)

        if flags[0] == 1:
            smooth_op[i, i - n_patch_strike] = delta_l_dip
        if flags[1] == 1:
            smooth_op[i, i + n_patch_strike] = delta_l_dip
        if flags[2] == 1:
            smooth_op[i, i - 1] = delta_l_strike
        if flags[3] == 1:
            smooth_op[i, i + 1] = delta_l_strike

    return smooth_op
