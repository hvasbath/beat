"""
Text file trace backend modified from pymc3 to work efficiently with
SMC

Store sampling values as CSV files.

File format
-----------

Sampling values for each chain are saved in a separate file (under a
directory specified by the `dir_path` argument).  The rows correspond to
sampling iterations.  The column names consist of variable names and
index labels.  For example, the heading

  x,y__0_0,y__0_1,y__1_0,y__1_1,y__2_0,y__2_1

represents two variables, x and y, where x is a scalar and y has a
shape of (3, 2).
"""
import copy
import itertools
import json
import logging
import os
import shutil
from collections import OrderedDict
from glob import glob
from time import time

import numpy as num
import pandas as pd
from pandas.errors import EmptyDataError

# pandas version control
try:
    from pandas.io.common import CParserError
except ImportError:
    from pandas.errors import ParserError as CParserError

from pymc3.backends import base, ndarray
from pymc3.backends import tracetab as ttab
from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import modelcontext
from pymc3.step_methods.arraystep import BlockedStep
from pyrocko import util

from beat.config import sample_p_outname, transd_vars_dist
from beat.covariance import calc_sample_covariance
from beat.utility import (
    ListArrayOrdering,
    ListToArrayBijection,
    dump_objects,
    list2string,
    load_objects,
)

logger = logging.getLogger("backend")


def thin_buffer(buffer, buffer_thinning, ensure_last=True):
    """
    Reduce a list of objects by a given value.

    Parameters
    ----------
    buffer : list
        of objects to be thinned
    buffer_thinning : int
        every nth object in list is returned
    ensure_last : bool
        enable to ensure that last object in list is returned
    """
    if ensure_last:
        write_buffer = buffer[-1::-buffer_thinning]
        write_buffer.reverse()
    else:
        write_buffer = buffer[::buffer_thinning]
    return write_buffer


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
    blocked : boolean
        (default True)
    """

    def __init__(self, vars, out_vars, shared, blocked=True):
        self.vars = vars
        self.ordering = ArrayOrdering(vars)
        self.lordering = ListArrayOrdering(out_vars, intype="tensor")
        lpoint = [var.tag.test_value for var in out_vars]
        self.shared = {var.name: shared for var, shared in shared.items()}
        self.blocked = blocked
        self.bij = DictToArrayBijection(self.ordering, self.population[0])

        blacklist = list(
            set(self.lordering.variables) - set([var.name for var in vars])
        )

        self.lij = ListToArrayBijection(self.lordering, lpoint, blacklist=blacklist)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        apoint, alist = self.astep(self.bij.map(point))

        return self.bij.rmap(apoint), alist


class BaseChain(object):
    """
    Base chain object, independent of file or memory output.

    Parameters
    ----------

    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """

    def __init__(self, model=None, vars=None, buffer_size=5000, buffer_thinning=1):

        self.model = None
        self.vars = None
        self.var_shapes = None
        self.chain = None

        self.buffer_size = buffer_size
        self.buffer_thinning = buffer_thinning
        self.buffer = []
        self.count = 0
        self.cov_counter = 0

        if model is not None:
            self.model = modelcontext(model)

        if vars is None and self.model is not None:
            vars = self.model.unobserved_RVs

        if vars is not None:
            self.vars = vars

        if self.vars is not None:
            # Get variable shapes. Most backends will need this
            # information.
            self.var_shapes = OrderedDict()
            self.var_dtypes = OrderedDict()
            self.varnames = []
            for var in self.vars:
                self.var_shapes[var.name] = var.tag.test_value.shape
                self.var_dtypes[var.name] = var.tag.test_value.dtype
                self.varnames.append(var.name)
        else:
            logger.debug("No model or variables given!")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)

        try:
            return self.point(int(idx))
        except (ValueError, TypeError):  # Passed variable or variable name.
            raise ValueError("Can only index with slice or integer")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def buffer_write(self, lpoint, draw):
        """
        Write sampling results into buffer.
        If buffer is full trow an error.
        """
        self.count += 1
        self.buffer.append((lpoint, draw))
        if self.count == self.buffer_size:
            raise BufferError("Buffer is full! Needs recording!!")

    def empty_buffer(self):
        self.buffer = []
        self.count = 0

    def get_sample_covariance(self, step):
        """
        Return sample Covariance matrix from buffer.
        """
        sample_difference = self.count - self.buffer_size
        if sample_difference < 0:
            raise ValueError("Covariance has been updated already!")
        elif sample_difference > 0:
            raise BufferError("Buffer is not full and sample covariance may be biased")
        else:
            logger.info(
                "Evaluating sampled trace covariance of worker %i at "
                "sample %i" % (self.chain, step.cumulative_samples)
            )

            cov = calc_sample_covariance(
                self.buffer, lij=step.lij, bij=step.bij, beta=step.beta
            )
            self.cov_counter += 1
        return cov


class FileChain(BaseChain):
    """
    Base class for a trace written to a file with buffer functionality and
    rogressbar. Buffer is a list of tuples of lpoints and a draw index. Inheriting classes
    must define the methods: '_write_data_to_file' and '_load_df'
    """

    def __init__(
        self,
        dir_path="",
        model=None,
        vars=None,
        buffer_size=5000,
        buffer_thinning=1,
        progressbar=False,
        k=None,
    ):

        super(FileChain, self).__init__(
            model=model,
            vars=vars,
            buffer_size=buffer_size,
            buffer_thinning=buffer_thinning,
        )

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.dir_path = dir_path

        self.flat_names = OrderedDict()
        if self.var_shapes is not None:
            if k is not None:
                for var, shape in self.var_shapes.items():
                    if var in transd_vars_dist:
                        shape = (k,)

                    self.flat_names[var] = ttab.create_flat_names(var, shape)
            else:
                for var, shape in self.var_shapes.items():
                    self.flat_names[var] = ttab.create_flat_names(var, shape)

        self.k = k

        self.corrupted_flag = False
        self.progressbar = progressbar

        self.stored_samples = 0
        self.draws = 0
        self._df = None
        self.filename = None

    def __len__(self):
        if self.filename is None:
            return 0

        self._load_df()

        if self._df is None:
            return 0
        else:
            return self._df.shape[0] + len(self.buffer)

    def add_derived_variables(self, varnames, shapes):
        nshapes = len(shapes)
        nvars = len(varnames)
        if nvars != nshapes:
            raise ValueError(
                "Inconsistent number of variables %i and shapes %i!" % (nvars, nshapes)
            )

        for varname, shape in zip(varnames, shapes):
            self.flat_names[varname] = ttab.create_flat_names(varname, shape)
            self.var_shapes[varname] = shape
            self.var_dtypes[varname] = "float64"
            self.varnames.append(varname)

    def _load_df(self):
        raise ValueError("This method must be defined in inheriting classes!")

    def _write_data_to_file(self):
        raise ValueError("This method must be defined in inheriting classes!")

    def data_file(self):
        return self._df

    def record_buffer(self):

        if self.chain is None:
            raise ValueError("Chain has not been setup. Saving samples not possible!")

        else:
            n_samples = len(self.buffer)
            self.stored_samples += n_samples

            if not self.progressbar:
                if n_samples > self.buffer_size // 2:
                    logger.info(
                        "Writing %i / %i samples of chain %i to disk..."
                        % (self.stored_samples, self.draws, self.chain)
                    )

            t0 = time()
            logger.debug("Start Record: Chain_%i" % self.chain)
            self._write_data_to_file()

            t1 = time()
            logger.debug("End Record: Chain_%i" % self.chain)
            logger.debug("Writing to file took %f" % (t1 - t0))
            self.empty_buffer()

    def write(self, lpoint, draw):
        """
        Write sampling results into buffer.
        If buffer is full write samples to file.
        """
        self.count += 1
        self.buffer.append((lpoint, draw))
        if self.count == self.buffer_size:
            self.record_buffer()

    def clear_data(self):
        """
        Clear the data loaded from file.
        """
        self._df = None


class MemoryChain(BaseChain):
    """
    Slim memory trace object. Keeps points in a list in memory.

    Parameters
    ----------
    draws : int
        Number of samples
    chain : int
        Chain number
    """

    def __init__(self, buffer_size=5000):

        super(MemoryChain, self).__init__(buffer_size=buffer_size)

    def setup(self, draws, chain, overwrite=False):
        self.draws = draws
        self.chain = chain

        if self.buffer is None:
            self.buffer = []

        if overwrite:
            self.buffer = []

    def record_buffer(self):
        logger.debug("Emptying buffer of trace %i" % self.chain)
        self.empty_buffer()


class TextChain(FileChain):
    """
    Text trace object based on '.csv' files. Slow in reading and writing.
    Good for debugging.

    Parameters
    ----------

    dir_path : str
        Name of directory to store text files
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    buffer_thinning : int
        every nth sample of the buffer is written to disk
    progressbar : boolean
        flag if a progressbar is active, if not a logmessage is printed
        every time the buffer is written to disk
    k : int, optional
        if given dont use shape from testpoint as size of transd variables
    """

    def __init__(
        self,
        dir_path,
        model=None,
        vars=None,
        buffer_size=5000,
        buffer_thinning=1,
        progressbar=False,
        k=None,
    ):

        super(TextChain, self).__init__(
            dir_path,
            model,
            vars,
            buffer_size=buffer_size,
            progressbar=progressbar,
            k=k,
            buffer_thinning=buffer_thinning,
        )

    def setup(self, draws, chain, overwrite=False):
        """
        Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        logger.debug("SetupTrace: Chain_%i step_%i" % (chain, draws))
        self.chain = chain

        self.draws = draws
        self.filename = os.path.join(self.dir_path, "chain-{}.csv".format(chain))

        cnames = [fv for v in self.varnames for fv in self.flat_names[v]]

        if os.path.exists(self.filename) and not overwrite:
            logger.debug("Found existing trace, appending!")
        else:
            self.count = 0

            # writing header
            with open(self.filename, "w") as fh:
                fh.write(",".join(cnames) + "\n")

    def _write_data_to_file(self, lpoint=None):
        """
        Write the lpoint to file. If lpoint is None it
        will try to write from buffer.

        Parameters
        ----------
        lpoint: list
            of numpy arrays
        """

        def lpoint2file(filehandle, lpoint):
            columns = itertools.chain.from_iterable(
                map(str, value.ravel()) for value in lpoint
            )
            filehandle.write(",".join(columns) + "\n")

        # Write binary
        if lpoint is None and len(self.buffer) == 0:
            logger.debug("There is no data to write into file.")

        try:
            with open(self.filename, mode="a+") as fh:
                if lpoint is None:
                    # write out thinned buffer starting with last sample
                    write_buffer = thin_buffer(
                        self.buffer, self.buffer_thinning, ensure_last=True
                    )
                    for lpoint, draw in write_buffer:
                        lpoint2file(fh, lpoint)

                else:
                    lpoint2file(fh, lpoint)

        except EnvironmentError as e:
            print("Error on write file: ", e)

    def _load_df(self):
        if self._df is None:
            try:
                self._df = pd.read_csv(self.filename)
            except EmptyDataError:
                logger.warning(
                    "Trace %s is empty and needs to be resampled!" % self.filename
                )
                os.remove(self.filename)
                self.corrupted_flag = True
            except CParserError:
                logger.warning("Trace %s has wrong size!" % self.filename)
                self.corrupted_flag = True
                os.remove(self.filename)

            if len(self.flat_names) == 0 and not self.corrupted_flag:
                self.flat_names, self.var_shapes = extract_variables_from_df(self._df)
                self.varnames = list(self.var_shapes.keys())

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
            Number of thinning samples. Throw out every 'thin' sample of the
            trace.

        Returns
        -------

        :class:`numpy.array`
        """
        self._load_df()

        try:
            var_df = self._df[self.flat_names[varname]]
            shape = (self._df.shape[0],) + self.var_shapes[varname]
            vals = var_df.values.ravel().reshape(shape)
            return vals[burn::thin]
        except (KeyError):
            raise ValueError(
                'Did not find varname "%s" in sampling ' "results! Fixed?" % varname
            )

    def _slice(self, idx):
        if idx.stop is not None:
            raise ValueError("Stop value in slice not supported.")
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
            # needs deepcopy otherwise reference to df is kept repetead calls
            # lead to memory leak
            vals = self._df[self.flat_names[varname]].iloc[idx]
            pt[varname] = copy.deepcopy(vals.values.reshape(self.var_shapes[varname]))
            del vals
        return pt


class NumpyChain(FileChain):
    """
    Numpy binary trace object based on '.bin' files. Fast in reading and
    writing. Bad for debugging.

    Parameters
    ----------

    dir_path : str
        Name of directory to store text files
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    buffer_size : int
        this is the number of samples after which the buffer is written to disk
        or if the chain end is reached
    buffer_thinning : int
        every nth sample of the buffer is written to disk
    progressbar : boolean
        flag if a progressbar is active, if not a logmessage is printed
        every time the buffer is written to disk
    k : int, optional
        if given dont use shape from testpoint as size of transd variables
    """

    flat_names_tag = "flat_names"
    var_shape_tag = "var_shapes"
    var_dtypes_tag = "var_dtypes"
    __data_structure = None

    def __init__(
        self,
        dir_path,
        model=None,
        vars=None,
        buffer_size=5000,
        progressbar=False,
        k=None,
        buffer_thinning=1,
    ):

        super(NumpyChain, self).__init__(
            dir_path,
            model,
            vars,
            progressbar=progressbar,
            buffer_size=buffer_size,
            buffer_thinning=buffer_thinning,
            k=k,
        )

        self.k = k

    def __repr__(self):
        return "NumpyChain({},{},{},{},{},{})".format(
            self.dir_path,
            self.model,
            self.vars,
            self.buffer_size,
            self.progressbar,
            self.k,
        )

    @property
    def data_structure(self):
        return self.__data_structure

    @property
    def file_header(self):
        with open(self.filename, mode="rb") as file:
            # read header.
            file_header = file.readline().decode()
            return file_header

    def setup(self, draws, chain, overwrite=False):
        """
        Perform chain-specific setup. Creates file with header.
        If exist not overwritten again unless flag is set.

        Parameters
        ----------
        draws: int.
            Expected number of draws
        chain:
            int. Chain number
        overwrite:
            Bool (optional). True(default) if file need to be overwrite,
            false otherwise.
        """

        logger.debug("SetupTrace: Chain_%i step_%i" % (chain, draws))
        self.chain = chain

        self.draws = draws
        self.filename = os.path.join(self.dir_path, "chain-{}.bin".format(chain))
        self.__data_structure = self.construct_data_structure()
        if os.path.exists(self.filename) and not overwrite:
            logger.info("Found existing trace, appending!")
        else:
            logger.debug('Setup new "bin" trace for chain %i' % chain)
            self.count = 0
            data_type = OrderedDict()
            with open(self.filename, "wb") as fh:
                for k, v in self.var_dtypes.items():
                    data_type[k] = "{}".format(v)

                header_data = {
                    self.flat_names_tag: self.flat_names,
                    self.var_shape_tag: self.var_shapes,
                    self.var_dtypes_tag: data_type,
                }
                header = (json.dumps(header_data) + "\n").encode()
                fh.write(header)

    def extract_variables_from_header(self, file_header):
        header_data = json.loads(file_header, object_pairs_hook=OrderedDict)
        flat_names = header_data[self.flat_names_tag]
        var_shapes = OrderedDict()
        for k, v in header_data[self.var_shape_tag].items():
            var_shapes[k] = tuple(v)
        var_dtypes = header_data[self.var_dtypes_tag]
        varnames = list(flat_names.keys())
        return flat_names, var_shapes, var_dtypes, varnames

    def construct_data_structure(self):
        """
        Create a dtype to store the data based on varnames in a numpy array.

        Returns
        -------
            A numpy.dtype
        """

        if len(self.flat_names) == 0 and not self.corrupted_flag:
            (
                self.flat_names,
                self.var_shapes,
                self.var_dtypes,
                self.varnames,
            ) = self.extract_variables_from_header(self.file_header)

        formats = [
            "{shape}{dtype}".format(
                shape=self.var_shapes[name], dtype=self.var_dtypes[name]
            )
            for name in self.varnames
        ]

        # set data structure
        return num.dtype({"names": self.varnames, "formats": formats})

    def _write_data_to_file(self, lpoint=None):
        """
        Writes lpoint to file. If lpoint is None it
        will try to write from buffer.

        Parameters
        ----------
        lpoint: list
            of numpy arrays.
        """

        def lpoint2file(filehandle, varnames, data, lpoint):
            for names, array in zip(varnames, lpoint):
                data[names] = array

            data.tofile(filehandle)

        # Write binary
        if lpoint is None and len(self.buffer) == 0:
            logger.debug("There is no data to write into file.")

        try:
            # create initial data using the data structure.
            data = num.zeros(1, dtype=self.data_structure)

            with open(self.filename, mode="ab+") as fh:
                if lpoint is None:
                    write_buffer = thin_buffer(
                        self.buffer, self.buffer_thinning, ensure_last=True
                    )
                    for lpoint, draw in write_buffer:
                        lpoint2file(fh, self.varnames, data, lpoint)
                else:
                    lpoint2file(fh, self.varnames, data, lpoint)

        except EnvironmentError as e:
            print("Error on write file: ", e)

    def _load_df(self):

        if not self.__data_structure:
            try:
                self.__data_structure = self.construct_data_structure()
            except json.decoder.JSONDecodeError:
                logger.warning(
                    "File header of %s is corrupted!" " Resampling!" % self.filename
                )
                self.corrupted_flag = True

        if self._df is None and not self.corrupted_flag:
            try:
                with open(self.filename, mode="rb") as file:
                    # skip header.
                    next(file)
                    # read data
                    self._df = num.fromfile(file, dtype=self.data_structure)
            except EOFError as e:
                print(e)
                self.corrupted_flag = True

    def get_values(self, varname, burn=0, thin=1):
        self._load_df()
        try:
            data = self._df[varname]
            shape = (self._df.shape[0],) + self.var_shapes[varname]
            vals = data.ravel().reshape(shape)
            return vals[burn::thin]
        except (ValueError):
            raise ValueError(
                'Did not find varname "%s" in sampling ' "results! Fixed?" % varname
            )

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
            data = self._df[varname][idx]
            pt[varname] = data.reshape(self.var_shapes[varname])

        return pt


backend_catalog = {"csv": TextChain, "bin": NumpyChain}


class TransDTextChain(object):
    """
    Result Trace object for trans-d problems.
    Manages several TextChains one for each dimension.
    """

    def __init__(
        self, name, model=None, vars=None, buffer_size=5000, progressbar=False
    ):

        self._straces = {}
        self.buffer_size = buffer_size
        self.progressbar = progressbar

        if vars is None:
            vars = model.unobserved_RVs

        transd, dims_idx = istransd(model)
        if transd:
            self.dims_idx
        else:
            raise ValueError("Model is not trans-d but TransD Chain initialized!")

        dimensions = model.unobserved_RVs[self.dims_idx]

        for k in range(dimensions.lower, dimensions.upper + 1):
            self._straces[k] = TextChain(
                dir_path=name,
                model=model,
                buffer_size=buffer_size,
                progressbar=progressbar,
                k=k,
            )

        # init indexing chain
        self._index = TextChain(
            dir_path=name,
            vars=[],
            buffer_size=self.buffer_size,
            progressbar=self.progressbar,
        )
        self._index.flat_names = {"draw__0": (1,), "k__0": (1,), "k_idx__0": (1,)}

    def setup(self, draws, chain):
        self.draws = num.zeros(1, dtype="int32")
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
        return self._straces[ipoint["k"]].point(ipoint["k_idx"])

    def get_values(self, varname):
        raise NotImplementedError()


class SampleStage(object):
    def __init__(self, base_dir, backend="csv"):
        self.base_dir = base_dir
        self.project_dir = os.path.dirname(base_dir)
        self.mode = os.path.basename(base_dir)
        self.backend = backend
        util.ensuredir(self.base_dir)

    def stage_path(self, stage):
        return os.path.join(self.base_dir, "stage_{}".format(stage))

    def trans_stage_path(self, stage):
        return os.path.join(self.base_dir, "trans_stage_{}".format(stage))

    def stage_number(self, stage_path):
        """
        Inverse function of SampleStage.path
        """
        return int(os.path.basename(stage_path).split("_")[-1])

    def highest_sampled_stage(self):
        """
        Return stage number of stage that has been sampled before the final
        stage.

        Returns
        -------
        stage number : int
        """
        return max(self.stage_number(s) for s in glob(self.stage_path("*")))

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
        return os.path.join(self.stage_path(stage_number), sample_p_outname)

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

        logger.info("Loading parameters from completed stage {}".format(prev))
        sampler_state, updates = load_objects(self.atmip_path(prev))
        sampler_state["stage"] = stage_number
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
                logger.info("Removing previous sampling results ... %s" % stage_path)
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
            dirname=dirname, chains=chains, varnames=varnames, backend=self.backend
        )

    def recover_existing_results(
        self, stage, draws, step, buffer_thinning=1, varnames=None, update=None
    ):

        if stage > 0:
            prev = stage - 1
            if update is not None:
                prev_stage_path = self.trans_stage_path(prev)
            else:
                prev_stage_path = self.stage_path(prev)

            logger.info(
                "Loading end points of last completed stage: " "%s" % prev_stage_path
            )
            mtrace = load_multitrace(
                dirname=prev_stage_path, varnames=varnames, backend=self.backend
            )

            (
                step.population,
                step.array_population,
                step.likelihoods,
            ) = step.select_end_points(mtrace)

        stage_path = self.stage_path(stage)
        if os.path.exists(stage_path):
            # load incomplete stage results
            logger.info("Reloading existing results ...")
            mtrace = self.load_multitrace(stage, varnames=varnames)
            if len(mtrace.chains):
                # continue sampling if traces exist
                logger.info("Checking for corrupted files ...")
                return check_multitrace(
                    mtrace,
                    draws=draws,
                    n_chains=step.n_chains,
                    buffer_thinning=buffer_thinning,
                )
        else:
            logger.info("Found no sampling results under %s " % stage_path)
            logger.info("Init new trace!")
        return None


def istransd(varnames):
    dims = "dimensions"
    if dims in varnames:
        dims_idx = varnames.index(dims)
        return True, dims_idx
    else:
        logger.debug('Did not find "%s" random variable in model!' % dims)
        return False, None


def load_multitrace(dirname, varnames=[], chains=None, backend="csv"):
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
        logger.info("Loading multitrace from %s" % dirname)
        if chains is None:
            files = glob(os.path.join(dirname, "chain-*.%s" % backend))
            chains = [
                int(os.path.splitext(os.path.basename(f))[0].replace("chain-", ""))
                for f in files
            ]

            final_chain = -1
            if final_chain in chains:
                idx = chains.index(final_chain)
                files.pop(idx)
                chains.pop(idx)
        else:
            files = [
                os.path.join(dirname, "chain-%i.%s" % (chain, backend))
                for chain in chains
            ]
            for f in files:
                if not os.path.exists(f):
                    raise IOError(
                        "File %s does not exist! Please run:"
                        ' "beat summarize <project_dir>"!' % f
                    )

        straces = []
        for chain, f in zip(chains, files):
            strace = backend_catalog[backend](dirname)
            strace.chain = chain
            strace.filename = f
            straces.append(strace)
        return base.MultiTrace(straces)
    else:
        logger.info("Loading trans-d trace from %s" % dirname)
        raise NotImplementedError("Loading trans-d trace is not implemented!")


def check_multitrace(mtrace, draws, n_chains, buffer_thinning=1):
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
    # apply buffer thinning
    draws = int(num.ceil(draws / buffer_thinning))

    for chain in range(n_chains):
        if chain in mtrace.chains:
            chain_len = len(mtrace._straces[chain])
            if chain_len != draws:
                logger.warn(
                    "Trace number %i incomplete: (%i / %i)" % (chain, chain_len, draws)
                )
                mtrace._straces[chain].corrupted_flag = True
        else:
            not_sampled_idx.append(chain)

    flag_bool = [mtrace._straces[chain].corrupted_flag for chain in mtrace.chains]
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
    stages = glob(os.path.join(homedir, "stage_*"))

    stagenumbers = []
    for s in stages:
        stage_ending = os.path.splitext(s)[0].rsplit("_", 1)[1]
        try:
            stagenumbers.append(int(stage_ending))
        except ValueError:
            logger.debug("string - That's the final stage!")
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

    stage_path = os.path.join(
        project_dir, mode, "stage_%s" % stage_number, sample_p_outname
    )
    return load_objects(stage_path)


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
    varnames = list(set([index.split("__")[0] for index in all_df_indexes]))

    flat_names = OrderedDict()
    var_shapes = OrderedDict()
    for varname in varnames:
        indexes = []
        for index in all_df_indexes:
            if index.split("__")[0] == varname:
                indexes.append(index)

        flat_names[varname] = indexes
        var_shapes[varname] = ttab._create_shape(indexes)

    return flat_names, var_shapes


def extract_bounds_from_summary(summary, varname, shape, roundto=None, alpha=0.01):
    """
    Extract lower and upper bound of random variable.

    Returns
    -------
    list of num.Ndarray
    """

    def do_nothing(value):
        return value

    indexes = ttab.create_flat_names(varname, shape)
    lower_quant = "hpd_{0:g}".format(100 * alpha / 2)
    upper_quant = "hpd_{0:g}".format(100 * (1 - alpha / 2))

    bounds = []
    for quant in [lower_quant, upper_quant]:
        values = num.empty(shape, "float64")
        for i, idx in enumerate(indexes):
            if roundto is not None:
                adjust = 10.0**roundto
                if quant == lower_quant:
                    operation = num.floor
                elif quant == upper_quant:
                    operation = num.ceil
            else:
                adjust = 1.0
                operation = do_nothing
            values[i] = operation(summary[quant][idx] * adjust) / adjust

        bounds.append(values)

    return bounds
