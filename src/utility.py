from pyproj import Proj
import numpy as num
import collections
import copy

DataMap = collections.namedtuple('DataMap', 'list_ind, slc, shp, dtype')


class ArrayOrdering(object):
    """
    An ordering for an array space. Modified from pymc3 blocking.
    """
    def __init__(self, gtargets):
        self.vmap = []
        dim = 0

        count = 0
        for target in gtargets:
            slc = slice(dim, dim + target.displacement.size)
            self.vmap.append(DataMap(count, slc, target.displacement.shape,
                                                 target.displacement.dtype))
            dim += target.displacement.size
            count += 1

        self.dimensions = dim


class ListToArrayBijection(object):
    """
    A mapping between a List of arrays and an array space
    """
    def __init__(self, ordering, list_arrays):
        self.ordering = ordering
        self.list_arrays = list_arrays

    def fmap(self, list_arrays):
        """
        Maps values from List space to array space

        Parameters
        ----------
        list_arrays : list of numpy arrays
        """
        a_list = num.empty(self.ordering.dimensions)
        for list_ind, slc, _, _ in self.ordering.vmap:
            a_list[slc] = list_arrays[list_ind].ravel()
        return a_list

    def rmap(self, array):
        """
        Maps value from array space to List space

        Parameters
        ----------
        array - numpy-array non-symbolic
        """
        a_list = copy.copy(self.list_arrays)

        for list_ind, slc, shp, dtype in self.ordering.vmap:
            a_list[list_ind] = num.atleast_1d(
                                        array)[slc].reshape(shp).astype(dtype)

        return a_list

    def srmap(self, tarray):
        """
        Maps value from symbolic variable array space to List space

        Parameters
        ----------
        tarray - theano-array symbolic
        """
        a_list = copy.copy(self.list_arrays)

        for list_ind, slc, shp, dtype in self.ordering.vmap:
            a_list[list_ind] = tarray[slc].reshape(shp).astype(dtype)

        return a_list


def utm_to_loc(utmx, utmy, zone, event):
    '''
    Convert UTM[m] to local coordinates with reference to the :py:class:`Event`
    Input: Numpy arrays with UTM easting(utmx) and northing(utmy)
           zone - Integer number with utm zone
    Returns: Local coordinates [m] x, y
    '''
    p = Proj(proj='utm', zone=zone, ellps='WGS84')
    ref_x, ref_y = p(event.lon, event.lat)
    utmx -= ref_x
    utmy -= ref_y
    return utmx, utmy


def utm_to_lonlat(utmx, utmy, zone):
    '''
    Convert UTM[m] to Latitude and Longitude
    Input: Numpy arrays with UTM easting(utmx) and northing(utmy)
           zone - Integer number with utm zone
    Returns: Longitude, Latitude [deg]
    '''
    p = Proj(proj='utm', zone=zone, ellps='WGS84')
    lon, lat = p(utmx, utmy, inverse=True)
    return lon, lat
