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
import numpy as num

from pymc3.model import modelcontext
from pymc3.backends import base, ndarray
from pymc3.backends import tracetab as ttab
from pymc3.blocking import DictToArrayBijection, ArrayOrdering

from pymc3.step_methods.arraystep import BlockedStep

from beat.config import sample_p_outname, transd_vars_dist
from beat.utility import load_objects, dump_objects, \
    ListArrayOrdering, ListToArrayBijection
from beat.covariance import calc_sample_covariance
from beat import utility

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
        self.lordering = ListArrayOrdering(out_vars, intype='tensor')
        lpoint = [var.tag.test_value for var in out_vars]
        self.shared = {var.name: shared for var, shared in shared.items()}
        self.blocked = blocked
        self.bij = DictToArrayBijection(self.ordering, self.population[0])

        blacklist = list(set(self.lordering.variables) -
                         set([var.name for var in vars]))

        self.lij = ListToArrayBijection(
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
        self.model = None
        self.vars = None
        self.var_shapes = None
        self.chain = None

        if model is not None:
            self.model = modelcontext(model)

        if vars is None and self.model is not None:
            vars = self.model.unobserved_RVs

        if vars is not None:
            self.vars = vars

        if self.vars is not None:
            self.varnames = [var.name for var in vars]

            # Get variable shapes. Most backends will need this
            # information.

            self.var_shapes_list = [var.tag.test_value.shape for var in vars]
            self.var_dtypes_list = [var.tag.test_value.dtype for var in vars]

            self.var_shapes = dict(zip(self.varnames, self.var_shapes_list))
            self.var_dtypes = dict(zip(self.varnames, self.var_dtypes_list))
        else:
            logger.debug('No model or variables given!')

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

    def get_sample_covariance(self, lij, bij, beta):
        """
        Return sample Covariance matrix from buffer.
        """
        self.count -= self.buffer_size
        if self.count < 0:
            raise ValueError('Covariance has been updated already!')

        cov = calc_sample_covariance(self.buffer, lij=lij, bij=bij, beta=beta)
        # reset buffer, keep last sample
        self.buffer = [self.buffer[-1]]
        return cov


def extract_variables_from_df(dataframe):
    """
    Extract random variables and their shapes from the pymc3-pandas data-frame

    Parameters
    ----------
    dataframe : :class:`pandas.DataFrame`

    Returns
    -------
    flat_names : dict
        with variable-names and respective flat-name indexes to data-frame
    var_shapes : dict
        with variable names and shapes
    """
    all_df_indexes = [str(flatvar) for flatvar in dataframe.columns]
    varnames = list(set([index.split('__')[0] for index in all_df_indexes]))

    flat_names = {}
    var_shapes = {}
    for varname in varnames:
        indexes = []
        for index in all_df_indexes:
            if index.split('__')[0] == varname:
                indexes.append(index)

        flat_names[varname] = indexes
        var_shapes[varname] = ttab._create_shape(indexes)

    return flat_names, var_shapes


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
    k : int, optional
        if given dont use shape from testpoint as size of transd variables
    """

    def __init__(
            self, name, model=None, vars=None,
            buffer_size=5000, progressbar=False, k=None):

        if not os.path.exists(name):
            os.mkdir(name)

        super(TextChain, self).__init__(name, model, vars)

        self.flat_names = None
        if self.var_shapes is not None:
            if k is not None:
                self.flat_names = {}
                for var, shape in self.var_shapes.items():
                    if var in transd_vars_dist:
                        shape = (k,)

                    self.flat_names[var] = ttab.create_flat_names(var, shape)

            else:
                self.flat_names = {v: ttab.create_flat_names(v, shape)
                                   for v, shape in self.var_shapes.items()}

        self.k = k
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

            if self.flat_names is None:
                self.flat_names, self.var_shapes = extract_variables_from_df(
                    self.df)
                self.varnames = self.var_shapes.keys()

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

    def get_stage_indexes(self, load_stage=None):
        """
        Return indexes to all sampled stages.

        Parameters
        ----------
        load_stage : int, optional
            if specified only return a list with this stage_index

        Returns
        -------
        list of int, stage_index that have been sampled
        """
        if load_stage is not None and isinstance(load_stage, int):
            return [load_stage]
        elif load_stage is not None and not isinstance(load_stage, int):
            raise ValueError('Requested stage_number has to be of type "int"')
        else:
            stage_number = self.highest_sampled_stage()

        if os.path.exists(self.atmip_path(-1)):
            list_indexes = [i for i in range(-1, stage_number + 1)]
        else:
            list_indexes = [i for i in range(stage_number)]

        return list_indexes

    def atmip_path(self, stage_number):
        """
        Consistent naming for atmip params.
        """
        return os.path.join(
            self.stage_path(stage_number), sample_p_outname)

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
        sampler_state, updates = load_objects(self.atmip_path(prev))
        sampler_state['stage'] = stage_number
        return sampler_state, updates

    def dump_atmip_params(self, stage_number, outlist):
        """
        Save atmip params to file.
        """
        dump_objects(self.atmip_path(stage_number), outlist)

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

    def load_multitrace(self, stage, chains=None, varnames=None):
        """
        Load TextChain database.

        Parameters
        ----------
        stage : int
            number of stage that should be loaded
        chains : list, optional
            of result chains to load, -1 is the summarized trace
        varnames : list
            of varnames in the model

        Returns
        -------
        A :class:`pymc3.backend.base.MultiTrace` instance
        """
        dirname = self.stage_path(stage)
        return load_multitrace(
            dirname=dirname, chains=chains, varnames=varnames)

    def recover_existing_results(self, stage, draws, step, varnames=None):
        stage_path = self.stage_path(stage)
        if os.path.exists(stage_path):
            # load incomplete stage results
            logger.info('Reloading existing results ...')
            mtrace = self.load_multitrace(stage, varnames=varnames)
            if len(mtrace.chains):
                # continue sampling if traces exist
                logger.info('Checking for corrupted files ...')
                return check_multitrace(
                    mtrace, draws=draws, n_chains=step.n_chains)

        logger.info('Init new trace!')
        return None


def istransd(varnames):
    dims = 'dimensions'
    if dims in varnames:
        dims_idx = varnames.index(dims)
        return True, dims_idx
    else:
        logger.debug('Did not find "%s" random variable in model!' % dims)
        return False, None


class TransDTextChain(object):
    """
    Result Trace object for trans-d problems.
    Manages several TextChains one for each dimension.
    """
    def __init__(
            self, name, model=None, vars=None,
            buffer_size=5000, progressbar=False):

        self._straces = {}
        self.buffer_size = buffer_size
        self.progressbar = progressbar

        if vars is None:
            vars = model.unobserved_RVs

        transd, dims_idx = istransd(model)
        if transd:
            self.dims_idx
        else:
            raise ValueError(
                'Model is not trans-d but TransD Chain initialized!')

        dimensions = model.unobserved_RVs[self.dims_idx]

        for k in range(dimensions.lower, dimensions.upper + 1):
            self._straces[k] = TextChain(
                name=name,
                model=model,
                buffer_size=buffer_size,
                progressbar=progressbar,
                k=k)

        # init indexing chain
        self._index = TextChain(
            name=name,
            vars=[],
            buffer_size=self.buffer_size,
            progressbar=self.progressbar)
        self._index.flat_names = {
            'draw__0': (1,), 'k__0': (1,), 'k_idx__0': (1,)}

    def setup(self, draws, chain):
        self.draws = num.zeros(1, dtype='int32')
        for k, trace in self._straces.items():
            trace.setup(draws=draws, chain=k)

        self._index.setup(draws, chain=0)

    def write(self, lpoint, draw):
        self.draws[0] = draw
        ipoint = [self.draws, lpoint[self.dims_idx]]

        self._index.write(ipoint, draw)
        self._straces[lpoint[self.dims_idx]].write(lpoint, draw)

    def __len__(self):
        return int(self._index[-1])

    def record_buffer(self):
        for trace in self._straces:
            trace.record_buffer()

        self._index.record_buffer()

    def point(self, idx):
        """
        Get point of current chain with variables names as keys.

        Parameters
        ----------
        idx : int
            Index of the nth step of the chain

        Returns
        -------
        dict : of point values
        """
        ipoint = self._index.point(idx)
        return self._straces[ipoint['k']].point(ipoint['k_idx'])

    def get_values(self, varname):
        raise NotImplementedError()


def load_multitrace(dirname, varnames=None, chains=None):
    """
    Load TextChain database.

    Parameters
    ----------
    dirname : str
        Name of directory with files (one per chain)
    varnames : list
        of strings with variable names
    chains : list optional

    Returns
    -------
    A :class:`pymc3.backend.base.MultiTrace` instance
    """

    if not istransd(varnames)[0]:
        logger.info('Loading multitrace from %s' % dirname)
        if chains is None:
            files = glob(os.path.join(dirname, 'chain-*.csv'))
            chains = [
                int(os.path.splitext(
                    os.path.basename(f))[0].replace('chain-', ''))
                for f in files]

            final_chain = -1
            if final_chain in chains:
                idx = chains.index(final_chain)
                files.pop(idx)
                chains.pop(idx)
        else:
            files = [
                os.path.join(
                    dirname, 'chain-%i.csv' % chain) for chain in chains]
            for f in files:
                if not os.path.exists(f):
                    raise IOError(
                        'File %s does not exist! Please run:'
                        ' "beat summarize <project_dir>"!' % f)

        straces = []
        for chain, f in zip(chains, files):
            strace = TextChain(dirname)
            strace.chain = chain
            strace.filename = f
            straces.append(strace)
        return base.MultiTrace(straces)
    else:
        logger.info('Loading trans-d trace from %s' % dirname)
        raise NotImplementedError('Loading trans-d trace is not implemented!')


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
            chain_len = len(mtrace._straces[chain])
            if chain_len != draws:
                logger.warn(
                    'Trace number %i incomplete: (%i / %i)' % (
                        chain, chain_len, draws))
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

    Parameters
    ----------
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
        sample_p_outname)
    return load_objects(stage_path)

def get_result_point(stage, config, point_llk='max'):
    """
    Return point of a given stage result.

    Parameters
    ----------
    stage : :class:`models.Stage`
    config : :class:`config.BEATConfig`
    point_llk : str
        with specified llk(max, mean, min).

    Returns
    -------
    dict
    """
    if config.sampler_config.name == 'Metropolis':
        if stage.step is None:
            raise AttributeError(
                'Loading Metropolis results requires'
                ' sampler parameters to be loaded!')

        sc = config.sampler_config.parameters
        from beat.sampler.metropolis import get_trace_stats
        pdict, _ = get_trace_stats(
            stage.mtrace, stage.step, sc.burn, sc.thin)
        point = pdict[point_llk]

    elif config.sampler_config.name == 'SMC':
        llk = stage.mtrace.get_values(
            varname='like',
            combine=True)

        posterior_idxs = utility.get_fit_indexes(llk)

        point = stage.mtrace.point(idx=posterior_idxs[point_llk])

    elif config.sampler_config.name == 'PT':
        params = config.sampler_config.parameters
        llk = stage.mtrace.get_values(
            varname='like',
            burn=int(params.n_samples * params.burn),
            thin=params.thin)

        posterior_idxs = utility.get_fit_indexes(llk)

        point = stage.mtrace.point(idx=posterior_idxs[point_llk])

    else:
        raise NotImplementedError(
            'Sampler "%s" is not supported!' % config.sampler_config.name)

    return point

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
