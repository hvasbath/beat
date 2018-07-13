"""
Text file trace backend modified from pymc3 to work efficiently with
SMC

Store sampling values as CSV files.

File format
-----------

Sampling values for each chain are saved in a separate file (under a
directory specified by the `name` argument).  The rows correspond to
sampling iterations.  The column names consist of variable names and
index labels.  For example, the heading

  x,y__0_0,y__0_1,y__1_0,y__1_1,y__2_0,y__2_1

represents two variables, x and y, where x is a scalar and y has a
shape of (3, 2).
"""
from glob import glob

import itertools
import copy
import os
import pandas as pd
import logging
import shutil

from pymc3.model import modelcontext
from pymc3.backends import base, ndarray
from pymc3.backends import tracetab as ttab
from pymc3.blocking import DictToArrayBijection, ArrayOrdering

from pymc3.step_methods.arraystep import BlockedStep

from beat import utility, config
from beat.covariance import calc_sample_covariance
from pyrocko import util
from time import time

logger = logging.getLogger('backend')


class ArrayStepSharedLLK(BlockedStep):
    """
    Modified ArrayStepShared To handle returned larger point including the
    likelihood values.
    Takes additionally a list of output vars including the likelihoods.

    Parameters
    ----------

    vars : list
        variables to be sampled
    out_vars : list
        variables to be stored in the traces
    shared : dict
        theano variable -> shared variables
    blocked : boolen
        (default True)
    """

    def __init__(self, vars, out_vars, shared, blocked=True):
        self.vars = vars
        self.ordering = ArrayOrdering(vars)
        self.lordering = utility.ListArrayOrdering(out_vars, intype='tensor')
        lpoint = [var.tag.test_value for var in out_vars]
        self.shared = {var.name: shared for var, shared in shared.items()}
        self.blocked = blocked
        self.bij = DictToArrayBijection(self.ordering, self.population[0])

        blacklist = list(set(self.lordering.variables) -
                         set([var.name for var in vars]))

        self.lij = utility.ListToArrayBijection(
            self.lordering, lpoint, blacklist=blacklist)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        apoint, alist = self.astep(self.bij.map(point))

        return self.bij.rmap(apoint), alist


class BaseTrace(object):
    """Base trace object

    Parameters
    ----------

    name : str
        Name of backend
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """

    def __init__(self, name, model=None, vars=None):
        self.name = name
        model = modelcontext(model)
        self.model = model

        if vars is None:
            vars = model.unobserved_RVs

        self.vars = vars
        self.varnames = [var.name for var in vars]

        # Get variable shapes. Most backends will need this
        # information.

        self.var_shapes_list = [var.tag.test_value.shape for var in vars]
        self.var_dtypes_list = [var.tag.test_value.dtype for var in vars]

        self.var_shapes = dict(zip(self.varnames, self.var_shapes_list))
        self.var_dtypes = dict(zip(self.varnames, self.var_dtypes_list))

        self.chain = None

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)

        try:
            return self.point(int(idx))
        except (ValueError, TypeError):  # Passed variable or variable name.
            raise ValueError('Can only index with slice or integer')

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class MemoryTraceError(Exception):
    pass


class MemoryTrace(BaseTrace):
    """
    Slim memory trace object. Keeps points in a list in memory.
    """
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = None
        self.count = 0

    def setup(self, draws, chain):
        self.draws = draws
        self.chain = chain

        if self.buffer is None:
            self.buffer = []

    def write(self, lpoint, draw):
        """
        Write sampling results into buffer.
        """
        if self.buffer is not None:
            self.count += 1
            self.buffer.append(lpoint)
        else:
            raise MemoryTraceError('Trace is not setup!')

    def get_sample_covariance(self, lij, beta):
        """
        Return sample Covariance matrix from buffer.
        """
        self.count -= self.buffer_size
        if self.count < 0:
            raise ValueError('Covariance has been updated already!')

        cov = calc_sample_covariance(self.buffer, lij, beta=beta)
        # reset buffer, keep last sample
        self.buffer = [self.buffer[-1]]
        return cov


class TextChain(BaseTrace):
    """
    Text trace object

    Parameters
    ----------

    name : str
        Name of directory to store text files
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    progressbar : boolean
        flag if a progressbar is active, if not a logmessage is printed
        everytime the buffer is written to disk
    """

    def __init__(
            self, name, model=None, vars=None,
            buffer_size=5000, progressbar=False):

        if not os.path.exists(name):
            os.mkdir(name)
        super(TextChain, self).__init__(name, model, vars)

        self.flat_names = {v: ttab.create_flat_names(v, shape)
                           for v, shape in self.var_shapes.items()}
        self.filename = None
        self.df = None
        self.corrupted_flag = False
        self.progressbar = progressbar
        self.buffer_size = buffer_size
        self.stored_samples = 0
        self.buffer = []

    def setup(self, draws, chain, overwrite=True):
        """
        Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        logger.debug('SetupTrace: Chain_%i step_%i' % (chain, draws))
        self.chain = chain
        self.count = 0
        self.draws = draws
        self.filename = os.path.join(self.name, 'chain-{}.csv'.format(chain))

        cnames = [fv for v in self.varnames for fv in self.flat_names[v]]

        if os.path.exists(self.filename):
            if overwrite:
                os.remove(self.filename)
            else:
                logger.info('Found existing trace, appending!')
                return

        with open(self.filename, 'w') as fh:
            fh.write(','.join(cnames) + '\n')

    def empty_buffer(self):
        self.buffer = []
        self.count = 0

    def write(self, lpoint, draw):
        """
        Write sampling results into buffer.
        If buffer is full write it out to file.
        """
        self.buffer.append((lpoint, draw))
        self.count += 1
        if self.count == self.buffer_size:
            self.record_buffer()

    def record_buffer(self):

        n_samples = len(self.buffer)
        self.stored_samples += n_samples

        if not self.progressbar:
            if n_samples > self.buffer_size / 2:
                logger.info(
                    'Writing %i / %i samples of chain %i to disk...' %
                    (self.stored_samples, self.draws, self.chain))

        t0 = time()
        logger.debug(
            'Start Record: Chain_%i' % self.chain)
        for lpoint, draw in self.buffer:
            self.record(lpoint, draw)

        t1 = time()
        logger.debug('End Record: Chain_%i' % self.chain)
        logger.debug('Writing to file took %f' % (t1 - t0))
        self.empty_buffer()

    def record(self, lpoint, draw):
        """
        Record results of a sampling iteration.

        Parameters
        ----------
        lpoint : List of variable values
            Values mapped to variable names
        """

        columns = itertools.chain.from_iterable(
            map(str, value.ravel()) for value in lpoint)

        logger.debug('Writing...: Chain_%i step_%i' % (
            self.chain, draw))
        with open(self.filename, 'a') as fh:
            fh.write(','.join(columns) + '\n')

    def _load_df(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.filename)
            except pd.errors.EmptyDataError:
                logger.warn(
                    'Trace %s is empty and needs to be resampled!' %
                    self.filename)
                os.remove(self.filename)
                self.corrupted_flag = True
            except pd.io.common.CParserError:
                logger.warn(
                    'Trace %s has wrong size!' % self.filename)
                self.corrupted_flag = True
                os.remove(self.filename)

    def __len__(self):
        if self.filename is None:
            return 0

        self._load_df()

        if self.df is None:
            return 0
        else:
            return self.df.shape[0] + len(self.buffer)

    def get_values(self, varname, burn=0, thin=1):
        """
        Get values from trace.

        Parameters
        ----------
        varname : str
            Variable name for which values are to be retrieved.
        burn : int
            Burn-in samples from trace. This is the number of samples to be
            thrown out from the start of the trace
        thin : int
            Nuber of thinning samples. Throw out every 'thin' sample of the
            trace.

        Returns
        -------

        :class:`numpy.array`
        """
        self._load_df()
        var_df = self.df[self.flat_names[varname]]
        shape = (self.df.shape[0],) + self.var_shapes[varname]
        vals = var_df.values.ravel().reshape(shape)
        return vals[burn::thin]

    def _slice(self, idx):
        if idx.stop is not None:
            raise ValueError('Stop value in slice not supported.')
        return ndarray._slice_as_ndarray(self, idx)

    def point(self, idx):
        """
        Get point of current chain with variables names as keys.

        Parameters
        ----------
        idx : int
            Index of the nth step of the chain

        Returns
        -------
        dictionary of point values
        """
        idx = int(idx)
        self._load_df()
        pt = {}
        for varname in self.varnames:
            vals = self.df[self.flat_names[varname]].iloc[idx]
            pt[varname] = vals.values.reshape(self.var_shapes[varname])
        return pt


class TextStage(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.project_dir = os.path.dirname(base_dir)
        self.mode = os.path.basename(base_dir)
        util.ensuredir(self.base_dir)

    def stage_path(self, stage):
        return os.path.join(self.base_dir, 'stage_{}'.format(stage))

    def stage_number(self, stage_path):
        """
        Inverse function of TextStage.path
        """
        return int(os.path.basename(stage_path).split('_')[-1])

    def highest_sampled_stage(self):
        """
        Return stage number of stage that has been sampled before the final
        stage.

        Returns
        -------
        stage number : int
        """
        return max(self.stage_number(s) for s in glob(self.stage_path('*')))

    def atmip_path(self, stage_number):
        """
        Consistent naming for atmip params.
        """
        return os.path.join(
            self.stage_path(stage_number), config.sample_p_outname)

    def load_sampler_params(self, stage_number):
        """
        Load saved parameters from last sampled stage.

        Parameters
        ----------
        stage number : int
            of stage number or -1 for last stage
        """
        if stage_number == -1:
            if not os.path.exists(self.atmip_path(stage_number)):
                prev = self.highest_sampled_stage()
            else:
                prev = stage_number
        elif stage_number == -2:
            prev = stage_number + 1
        else:
            prev = stage_number - 1

        logger.info('Loading parameters from completed stage {}'.format(prev))
        sampler_state, updates = utility.load_objects(self.atmip_path(prev))
        sampler_state['stage'] = stage_number
        return sampler_state, updates

    def dump_atmip_params(self, stage_number, outlist):
        """
        Save atmip params to file.
        """
        utility.dump_objects(self.atmip_path(stage_number), outlist)

    def clean_directory(self, stage, chains, rm_flag):
        """
        Optionally remove directory for the stage.
        Does nothing if rm_flag is False.
        """
        stage_path = self.stage_path(stage)
        if rm_flag:
            if os.path.exists(stage_path):
                logger.info(
                    'Removing previous sampling results ... %s' % stage_path)
                shutil.rmtree(stage_path)
            chains = None
        elif not os.path.exists(stage_path):
            chains = None
        return chains

    def load_multitrace(self, stage, model=None):
        """
        Load TextChain database.

        Parameters
        ----------
        stage : int
            number of stage that should be loaded
        model : Model
            If None, the model is taken from the `with` context.
        Returns
        -------
        A :class:`pymc3.backend.base.MultiTrace` instance
        """
        dirname = self.stage_path(stage)
        return load_multitrace(dirname=dirname, model=model)

    def recover_existing_results(self, stage, draws, step, model=None):
        stage_path = self.stage_path(stage)
        if os.path.exists(stage_path):
            # load incomplete stage results
            logger.info('Reloading existing results ...')
            mtrace = self.load_multitrace(stage, model=model)
            if len(mtrace.chains):
                # continue sampling if traces exist
                logger.info('Checking for corrupted files ...')
                return check_multitrace(
                    mtrace, draws=draws, n_chains=step.n_chains)

        logger.info('Init new trace!')
        return None


def load_multitrace(dirname, model=None):
    """
    Load TextChain database.

    Parameters
    ----------
    dirname : str
        Name of directory with files (one per chain)
    model : Model
        If None, the model is taken from the `with` context.

    Returns
    -------
    A :class:`pymc3.backend.base.MultiTrace` instance
    """

    logger.info('Loading multitrace from %s' % dirname)
    files = glob(os.path.join(dirname, 'chain-*.csv'))
    straces = []
    for f in files:
        chain = int(os.path.splitext(f)[0].rsplit('-', 1)[1])
        strace = TextChain(dirname, model=model)
        strace.chain = chain
        strace.filename = f
        straces.append(strace)
    return base.MultiTrace(straces)


def check_multitrace(mtrace, draws, n_chains):
    """
    Check multitrace for incomplete sampling and return indexes from chains
    that need to be resampled.

    Parameters
    ----------
    mtrace : :class:`pymc3.backend.base.MultiTrace`
        Multitrace object containing the sampling traces
    draws : int
        Number of steps (i.e. chain length for each Markov Chain)
    n_chains : int
        Number of Markov Chains

    Returns
    -------
    list of indexes for chains that need to be resampled
    """
    not_sampled_idx = []
    for chain in range(n_chains):
        if chain in mtrace.chains:
            if len(mtrace._straces[chain]) != draws:
                logger.warn('Trace number %i incomplete' % chain)
                mtrace._straces[chain].corrupted_flag = True
        else:
            not_sampled_idx.append(chain)

    flag_bool = [
        mtrace._straces[chain].corrupted_flag for chain in mtrace.chains]
    corrupted_idx = [i for i, x in enumerate(flag_bool) if x]
    return corrupted_idx + not_sampled_idx


def get_highest_sampled_stage(homedir, return_final=False):
    """
    Return stage number of stage that has been sampled before the final stage.

    Paramaeters
    -----------
    homedir : str
        Directory to the sampled stage results

    Returns
    -------
    stage number : int
    """
    stages = glob(os.path.join(homedir, 'stage_*'))

    stagenumbers = []
    for s in stages:
        stage_ending = os.path.splitext(s)[0].rsplit('_', 1)[1]
        try:
            stagenumbers.append(int(stage_ending))
        except ValueError:
            logger.debug('string - Thats the final stage!')
            if return_final:
                return stage_ending

    return max(stagenumbers)


def load_sampler_params(project_dir, stage_number, mode):
    """
    Load saved parameters from given ATMIP stage.

    Parameters
    ----------
    project_dir : str
        absolute path to directory of BEAT project
    stage number : string
        of stage number or 'final' for last stage
    mode : str
        problem mode that has been solved ('geometry', 'static', 'kinematic')
    """

    stage_path = os.path.join(project_dir, mode, 'stage_%s' % stage_number,
        config.sample_p_outname)
    return utility.load_objects(stage_path)


def concatenate_traces(mtraces):
    """
    Concatenate a List of MultiTraces with same chain indexes.
    """
    base_traces = copy.deepcopy(mtraces)
    cat_trace = base_traces.pop(0)

    cat_dfs = []
    for chain in cat_trace.chains:
        cat_trace._straces[chain]._load_df()
        cat_dfs.append(cat_trace._straces[chain].df)

    for mtrace in base_traces:
        for chain in cat_trace.chains:
            mtrace._straces[chain]._load_df()
            cat_dfs[chain] = cat_dfs[chain].append(mtrace._straces[chain].df)

    for chain in cat_trace.chains:
        cat_trace._straces[chain].df = cat_dfs[chain]

    return cat_trace
