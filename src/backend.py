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

import pymc3
from pymc3.model import modelcontext
from pymc3.backends import base, ndarray
from pymc3.backends import tracetab as ttab
from pymc3.blocking import DictToArrayBijection, ArrayOrdering

from beat import utility, config

logger = logging.getLogger('backend')


class ArrayStepSharedLLK(pymc3.arraystep.BlockedStep):
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
        self.lij = utility.ListToArrayBijection(self.lordering, lpoint)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        apoint, alist = self.astep(self.bij.map(point))

        return self.bij.rmap(apoint), alist


class BaseSMCTrace(object):
    """Base SMC trace object

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

        ## Get variable shapes. Most backends will need this
        ## information.

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


class Text(BaseSMCTrace):
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
    """

    def __init__(self, name, model=None, vars=None):
        if not os.path.exists(name):
            os.mkdir(name)
        super(Text, self).__init__(name, model, vars)

        self.flat_names = {v: ttab.create_flat_names(v, shape)
                           for v, shape in self.var_shapes.items()}
        self.filename = None
        self.df = None
        self.corrupted_flag = False

    ## Sampling methods

    def setup(self, draws, chain):
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
        self.filename = os.path.join(self.name, 'chain-{}.csv'.format(chain))

        cnames = [fv for v in self.varnames for fv in self.flat_names[v]]

        if os.path.exists(self.filename):
            os.remove(self.filename)

        with open(self.filename, 'w') as fh:
            fh.write(','.join(cnames) + '\n')

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
            except pd.parser.EmptyDataError:
                logger.warn('Trace %s is empty and needs to be resampled!' % \
                    self.filename)
                os.remove(self.filename)
                self.corrupted_flag = True
            except pd.io.common.CParserError:
                logger.warn('Trace %s has wrong size!' % \
                    self.filename)
                self.corrupted_flag = True
                os.remove(self.filename)

    def __len__(self):
        if self.filename is None:
            return 0

        self._load_df()

        if self.df is None:
            return 0
        else:
            return self.df.shape[0]

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
            pt[varname] = vals.reshape(self.var_shapes[varname]).as_matrix()
        return pt


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


def check_multitrace(mtrace, draws, n_chains):
    """
    Check multitrace for incomplete sampling and return indexes from chains
    that need to be resampled.

    Parameters
    ----------
    mtrace : :class:`pymc3.backend.base.MultiTrace`
        Mutlitrace object containing the sampling traces
    draws : int
        Number of steps (i.e. chain length for each Marcov Chain)
    n_chains : int
        Number of Marcov Chains

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
    step, update = utility.load_objects(stage_path)
    return step, update


def load(name, model=None):
    """
    Load Text database.

    Parameters
    ----------
    name : str
        Name of directory with files (one per chain)
    model : Model
        If None, the model is taken from the `with` context.

    Returns
    -------

    A :class:`pymc3.backend.base.MultiTrace` instance
    """
    files = glob(os.path.join(name, 'chain-*.csv'))

    straces = []
    for f in files:
        chain = int(os.path.splitext(f)[0].rsplit('-', 1)[1])
        strace = Text(name, model=model)
        strace.chain = chain
        strace.filename = f
        straces.append(strace)
    return base.MultiTrace(straces)


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


def dump(name, trace, chains=None):
    """
    Store values from NDArray trace as CSV files.

    Parameters
    ----------
    name : str
        Name of directory to store CSV files in
    trace : :class:`pymc3.backend.base.MultiTrace` of NDArray traces
        Result of MCMC run with default NDArray backend
    chains : list
        Chains to dump. If None, all chains are dumped.
    """

    if not os.path.exists(name):
        os.mkdir(name)
    if chains is None:
        chains = trace.chains

    var_shapes = trace._straces[chains[0]].var_shapes
    flat_names = {v: ttab.create_flat_names(v, shape)
                  for v, shape in var_shapes.items()}

    for chain in chains:
        filename = os.path.join(name, 'chain-{}.csv'.format(chain))
        df = ttab.trace_to_dataframe(
            trace, chains=chain, flat_names=flat_names)
        df.to_csv(filename, index=False)
