from beat import heart
import os
import logging
import collections


logger = logging.getLogger('ffy')


PatchMap = collections.namedtuple(
    'PatchMap', 'count, slc, shp, npatches')


class FaultOrdering(object):
    """
    A mapping of source patches to the arrays of optimization results.

    Parameters
    ----------
    npls : list
        of number of patches in strike-direction
    npws : list
        of number of patches in dip-direction
    """

    def __init__(self, npls, npws):

        self.vmap = []
        dim = 0
        count = 0

        for npl, npw in zip(npls, npws):
            npatches = npl * npw
            slc = slice(dim, dim + npatches)
            shp = (npw, npl)
            self.vmap.append(PatchMap(count, slc, shp, npatches))
            dim += npatches
            count += 1

        self.npatches = dim


slip_directions = {
    'Uparr': {'slip': 1., 'rake': 0.},
    'Uperp': {'slip': 1., 'rake': -90.},
    'Utensile': {'slip': 0., 'rake': 0., 'opening': 1.}}


class FaultGeometry(gf.seismosizer.Cloneable):
    """
    Object to construct complex fault-geometries with several subfaults.
    Stores information for subfault geometries and
    inversion variables (e.g. slip-components).
    Yields patch objects for requested subfault, dataset and component.

    Parameters
    ----------
    datatypes : list
        of str of potential dataset fault geometries to be stored
    components : list
        of str of potential inversion variables (e.g. slip-components) to
        be stored
    ordering : :class:`FaultOrdering`
        comprises patch information related to subfaults
    """

    def __init__(self, datatypes, components, ordering):
        self.datatypes = datatypes
        self.components = components
        self._ext_sources = {}
        self.ordering = ordering

    def _check_datatype(self, datatype):
        if datatype not in self.datatypes:
            raise TypeError('Datatype not included in FaultGeometry')

    def _check_component(self, component):
        if component not in self.components:
            raise TypeError('Component not included in FaultGeometry')

    def _check_index(self, index):
        if index > self.nsubfaults - 1:
            raise TypeError('Subfault with index %i not defined!' % index)

    def get_subfault_key(self, index, datatype, component):

        if datatype is not None:
            self._check_datatype(datatype)
        else:
            datatype = self.datatypes[0]

        if component is not None:
            self._check_component(component)
        else:
            component = self.components[0]

        self._check_index(index)

        return datatype + '_' + component + '_' + str(index)

    def setup_subfaults(self, datatype, component, ext_sources, replace=False):

        self._check_datatype(datatype)
        self._check_component(component)

        if len(ext_sources) != self.nsubfaults:
            raise Exception('Setup does not match fault ordering!')

        for i, source in enumerate(ext_sources):
            source_key = self.get_subfault_key(i, datatype, component)

            if source_key not in self._ext_sources.keys() or replace:
                self._ext_sources[source_key] = copy.deepcopy(source)
            else:
                raise Exception('Subfault already specified in geometry!')

    def get_subfault(self, index, datatype=None, component=None):

        source_key = self.get_subfault_key(index, datatype, component)

        if source_key in self._ext_sources.keys():
            return self._ext_sources[source_key]
        else:
            raise Exception('Requested subfault not defined!')

    def get_subfault_patches(self, index, datatype=None, component=None):

        self._check_index(index)

        subfault = self.get_subfault(
            index, datatype=datatype, component=component)
        npw, npl = self.ordering.vmap[index].shp

        return subfault.patches(nl=npl, nw=npw, datatype=datatype)

    def get_all_patches(self, datatype=None, component=None):
        """
        Get all RectangularSource patches for the full complex fault.

        Parameters
        ----------
        datatype : str
            'geodetic' or 'seismic'
        component : str
            slip component to return may be %s
        """ % utility.list2string(slip_directions.keys())

        patches = []
        for i in range(self.nsubfaults):
            patches += self.get_subfault_patches(
                i, datatype=datatype, component=component)

        return patches

    def get_patch_indexes(self, index):
        """
        Return indexes for sub-fault patches that translate to the solution
        array.

        Parameters
        ----------
        index : int
            to the sub-fault

        Returns
        -------
        slice : slice
            to the solution array that is being extracted from the related
            :class:`pymc3.backends.base.MultiTrace`
        """
        self._check_index(index)
        return self.ordering.vmap[index].slc

    @property
    def nsubfaults(self):
        return len(self.ordering.vmap)

    @property
    def nsubpatches(self):
        return self.ordering.npatches


def discretize_sources(
        sources=None, extension_width=0.1, extension_length=0.1,
        patch_width=5000., patch_length=5000., datatypes=['geodetic'],
        varnames=['']):
    """
    Build complex discretized fault.

    Extend sources into all directions and discretize sources into patches.
    Rounds dimensions to have no half-patches.

    Parameters
    ----------
    sources : :class:`sources.RectangularSource`
        Reference plane, which is being extended and
    extension_width : float
        factor to extend source in width (dip-direction)
    extension_length : float
        factor extend source in length (strike-direction)
    patch_width : float
        Width [m] of subpatch in dip-direction
    patch_length : float
        Length [m] of subpatch in strike-direction
    varnames : list
        of str with variable names that are being optimized for

    Returns
    -------
    :class:'FaultGeometry'
    """

    npls = []
    npws = []
    for source in sources:
        s = copy.deepcopy(source)
        ext_source = s.extent_source(
            extension_width, extension_length,
            patch_width, patch_length)

        npls.append(int(num.ceil(ext_source.length / patch_length)))
        npws.append(int(num.ceil(ext_source.width / patch_width)))

    ordering = utility.FaultOrdering(npls, npws)

    fault = FaultGeometry(datatypes, varnames, ordering)

    for datatype in datatypes:
        logger.info('Discretizing %s source(s)' % datatype)

        for var in varnames:
            logger.info('%s slip component' % var)
            param_mod = copy.deepcopy(slip_directions[var])

            ext_sources = []
            for source in sources:
                s = copy.deepcopy(source)
                param_mod['rake'] += s.rake
                s.update(**param_mod)

                ext_source = s.extent_source(
                    extension_width, extension_length,
                    patch_width, patch_length)

                npls.append(
                    ext_source.get_n_patches(patch_length, 'length'))
                npws.append(
                    ext_source.get_n_patches(patch_width, 'width'))
                ext_sources.append(ext_source)
                logger.info('Extended fault(s): \n %s' % ext_source.__str__())

            fault.setup_subfaults(datatype, var, ext_sources)

    return fault


def geo_construct_gf_linear(
        engine, outpath, crust_ind=0, datasets=None,
        targets=None, fault=None, varnames=[''], force=False):
    """
    Create geodetic Greens Function matrix for defined source geometry.

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        main path to directory containing the different Greensfunction stores
    outpath : str
        absolute path to the directory and filename where to store the
        Green's Functions
    crust_ind : int
        of index of Greens Function store to use
    datasets : list
        of :class:`heart.GeodeticDataset` for which the GFs are calculated
    targets : list
        of :class:`heart.GeodeticDataset`
    fault : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry
    varnames : list
        of str with variable names that are being optimized for
    force : bool
        Force to overwrite existing files.
    """

    if os.path.exists(outpath) and not force:
        logger.info(
            "Green's Functions exist! Use --force to"
            " overwrite!")
    else:
        out_gfs = {}
        for var in varnames:
            logger.info('For slip component: %s' % var)

            gfs = []
            for source in fault.get_all_patches('geodetic', component=var):
                disp = geo_synthetics(
                    engine=engine,
                    targets=targets,
                    sources=[source],
                    outmode='stacked_arrays')

                gfs_data = []
                for d, data in zip(disp, datasets):
                    logger.debug('Target %s' % data.__str__())
                    gfs_data.append((
                        d[:, 0] * data.los_vector[:, 0] +
                        d[:, 1] * data.los_vector[:, 1] +
                        d[:, 2] * data.los_vector[:, 2]) *
                        data.odw)

                gfs.append(num.vstack(gfs_data).T)

        out_gfs[var] = gfs
        logger.info("Dumping Green's Functions to %s" % outpath)
        utility.dump_objects(outpath, [out_gfs])


def seis_construct_gf_linear(
        engine, targets, fault, risetimes, varnames,
        lower_corner_f, upper_corner_f, cut_interval,
        outpath, saveflag=False):
    """
    Create seismic Greens Function matrix for defined source geometry
    by convolution of the GFs with the source time function (STF).

    Parameters
    ----------
    engine : :class:`pyrocko.gf.seismosizer.LocalEngine`
        main path to directory containing the different Greensfunction stores
    targets - list of pyrocko target objects for respective phase to compute
    fault : :class:`FaultGeometry`
        fault object that may comprise of several sub-faults. thus forming a
        complex fault-geometry

        risetimes - vector of risetimes of the STF for each patch to convolve
        lower/upper_corner_f - frequency range for filtering the GFs after
                               convolution
        cut_interval - list[time before, after] tapering each
                       phase arrival (target)

        outpath - path for storage
        saveflag - boolean to save Library at outpath

    Returns
    -------
    GFLibrary : list of Greensfunctions in the form
                [targets][patches][risetimes, cut_interval]
    GFTimes : list of respective times of begin, phase arrival and end of
                  traces in the form [targets][patches][start,arrival,end]
    """

    GFLibrary = []
    Times = []
    logger.info('Storing seismic linear GF Library under ', outpath)
    npatches = fault.nsubfaults
    ntargets = len(targets)

    for i in range(ntargets):
        GFLibrary.append([0] * npatches)
        Times.append([0] * npatches)

    for var in varnames:
        logger.info('For slip component: %s' % var)

        for i, patch in enumerate(
                fault.get_all_patches('seismic', component=var)):

            source_patches_risetimes = []
            logger.info('Patch Number %i', i)

            for risetime in risetimes:
                pcopy = patch.clone()
                pcopy.update(risetime=risetime)
                source_patches_risetimes.append(copy)

            for j, target in enumerate(targets):

                traces, tmins = seis_synthetics(
                    engine=engine,
                    sources=source_patches_risetimes,
                    targets=[target],
                    arrival_taper=,
                    wavename='any_P',
                    filterer=None,
                    reference_taperer=None,
                    outmode='array')



                # have to add event time as traces are with respect to that
                arrival_time = store.t(wave, (sdepth, sdist)) + dist_patches[p].time
                cut_start = arrival_time - cut_interval[0]
                cut_end = arrival_time + cut_interval[1]

                GF_traces = response.pyrocko_traces()

                post_process_trace(trace, taper, filterer, taper_tolerance_factor=0.,
                       outmode=None)

                # bandpass
                _ = [GF_traces[rt].bandpass(corner_hp = lower_corner_f, corner_lp = upper_corner_f, order=4) for rt in range(len(GF_traces))]

                # get trace times
                trcs_tmax = num.array([GF_traces[rt].tmax for rt in range(len(GF_traces))]).max()
                trcs_tmin = num.array([GF_traces[rt].tmin for rt in range(len(GF_traces))]).min()


                # zero padding of traces
                if trcs_tmin > cut_start:
                    trcs_tmin = cut_start
                if trcs_tmax < cut_end:
                    trcs_tmax = cut_end
                _ = [GF_traces[rt].extend(trcs_tmin, trcs_tmax) for rt in range(len(GF_traces))]

                # tapering around phase arrivals
                taperer = trace.CosTaper(cut_start, 
                                         cut_start + 10,
                                         cut_end - 10,
                                         cut_end)
                _ = [GF_traces[rt].taper(taperer,inplace=True,chop=True) for rt in range(len(GF_traces))]

                # bandpass again
                _ = [GF_traces[rt].bandpass(corner_hp = lower_corner_f, corner_lp = upper_corner_f, order=4) for rt in range(len(GF_traces))]

                # get trace times of tapered traces
                trcs_tmax = num.array([GF_traces[rt].tmax for rt in range(len(GF_traces))]).max()
                trcs_tmin = num.array([GF_traces[rt].tmin for rt in range(len(GF_traces))]).min()

                # re-arranging to matrix and put into list
                GFLibrary[t][p] = (num.vstack([GF_traces[rt].ydata for rt in xrange(len(patches_risetimes))]))
                Times[t][p] = num.vstack([trcs_tmin, arrival_time, trcs_tmax])

    if saveflag:
        with open(outpath,'w') as f:
            pickle.dump([GFLibrary, Times], f)

    return GFLibrary, Times
