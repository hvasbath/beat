from logging import getLogger
from utility import ListArrayOrdering
from pymc3.blocking import DictToArrayBijection, ArrayOrdering
from beat.config import k_name, transd_vars_dist
from theano import config as tconfig
import numpy as num


logger = getLogger('transd')


class TransDListArrayOrdering(object):
    """
    Holds orderings for trans-d problems.
    """
    def __init__(self, dimensions, list_arrays, intype='numpy'):

        self.korderings = {}
        for k in range(*dimensions):
            # arrays sizes need to be updated with k here!
            for array in list_arrays:
                if intype == 'tensor':
                    if array.name in transd_vars_dist:
                        array.tag.test_value = num.zeros(
                            (k,), dtype=tconfig.floatX)
                else:
                    raise ValueError(
                        'Trans-d mapping only supported for tensors')

            self.korderings[k] = ListArrayOrdering(
                list_arrays, intype=intype)

    def ks(self):
        return num.array(self.korderings.keys(), dtype='int16')

    @property
    def size(self):
        pass


class TransDArrayOrdering(object):
    """
    Holds orderings for trans-d problems.
    """
    def __init__(self, dimensions, vars):

        self.korderings = {}
        for k in range(*dimensions):
            # vars sizes need to be updated with k here!
            for var in vars:
                if var.name in transd_vars_dist:
                    var.dshape = (k,)
                    var.dsize = k
            self.korderings[k] = ArrayOrdering(vars)

    def ks(self):
        return num.array(self.korderings.keys(), dtype='int16')

    @property
    def size(self):
        pass


class TransDDictToArrayBijection(object):

    def __init__(self, kordering, point):

        self.ordering = kordering
        self.point = point
        self.mappings = {}
        self.current_k = None

        for k in self.ordering.ks():
            self.mappings[k] = DictToArrayBijection(
                self.ordering.korderings[k], self.point)

        # determine smallest float dtype that will fit all data
        if all([bij.array_dtype == 'float16'
                for bij in self.mappings.values()]):
            self.array_dtype = 'float16'
        elif all([bij.array_dtype == 'float32'
                  for bij in self.mappings.values()]):
            self.array_dtype = 'float32'
        else:
            self.array_dtype = 'float64'

    def map(self, dpt):
        """
        Maps value from transd dict space to transd array space
        Caches k from point for use in backtransform, removes it from point.

        Parameters
        ----------
        dpt : dict
        """
        if self.current_k is None:
            self.current_k = dpt.pop([k_name])
        else:
            raise ValueError(
                'Trans-D mapping dimension would be overwritten!'
                'Transform back or clear cache before mapping again!')

        return self.mappings[self.current_k].map(dpt)

    def rmap(self, apt, k=None):
        """
        Maps value from array space to dict space
        Adds k again to point

        Parameters
        ----------
        apt : array
        """
        if self.current_k is None and k is None:
            raise ValueError('Dimension needed!')
        elif self.current_k is None and k is not None:
            self.current_k = k
        elif self.current_k is not None and k is not None:
            logger.warning('Overwriting "k" in mapping!')
            self.current_k = k

        kpoint = self.mappings.bij.rmap(apt)
        kpoint[k_name] = self.current_k
        return kpoint


class TransDListToArrayBijection(object):
    """
    A mapping between a transd List of arrays and an transd array space

    Parameters
    ----------
    kordering : :class:`ListArrayOrdering`
    list_arrays : list
        of :class:`numpy.ndarray`
    blacklist : list
        of strings of variables to remove from point during transformation
        from list to dict
    """

    def __init__(self, lordering, list_arrays, blacklist=[]):
        self.lordering = lordering
        self.list_arrays = list_arrays
        self.dummy = -9.e40
        self.blacklist = blacklist