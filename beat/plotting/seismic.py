



def n_model_plot(models, axes=None, draw_bg=True, highlightidx=[]):
    """
    Plot cake layered earth models.
    """
    fontsize = 10
    if axes is None:
        mpl_init(fontsize=fontsize)
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize('a6', 'portrait'))
        labelpos = mpl_margins(
            fig, left=6, bottom=4, top=1.5, right=0.5, units=fontsize)
        labelpos(axes, 2., 1.5)

    def plot_profile(mod, axes, vp_c, vs_c, lw=0.5):
        z = mod.profile('z')
        vp = mod.profile('vp')
        vs = mod.profile('vs')
        axes.plot(vp, z, color=vp_c, lw=lw)
        axes.plot(vs, z, color=vs_c, lw=lw)

    cp.labelspace(axes)
    cp.labels_model(axes=axes)
    if draw_bg:
        cp.sketch_model(models[0], axes=axes)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)

    ref_vp_c = scolor('aluminium5')
    ref_vs_c = scolor('aluminium5')
    vp_c = scolor('scarletred2')
    vs_c = scolor('skyblue2')

    for i, mod in enumerate(models):
        plot_profile(
            mod, axes, vp_c=light(vp_c, 0.3), vs_c=light(vs_c, 0.3), lw=1.)

    for count, i in enumerate(sorted(highlightidx)):
        if count == 0:
            vpcolor = ref_vp_c
            vscolor = ref_vs_c
        else:
            vpcolor = vp_c
            vscolor = vs_c

        plot_profile(
            models[i], axes, vp_c=vpcolor, vs_c=vscolor, lw=2.)

    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    xmin = 0.
    my = (ymax - ymin) * 0.05
    mx = (xmax - xmin) * 0.2
    axes.set_ylim(ymax, ymin - my)
    axes.set_xlim(xmin, xmax + mx)
    return fig, axes


def load_earthmodels(store_superdir, store_ids, depth_max='cmb'):

    ems = []
    emr = []
    for store_id in store_ids:
        path = os.path.join(store_superdir, store_id, 'config')
        config = load(filename=path)
        em = config.earthmodel_1d.extract(depth_max=depth_max)
        ems.append(em)

        if config.earthmodel_receiver_1d is not None:
            emr.append(config.earthmodel_receiver_1d)

    return [ems, emr]


def draw_earthmodels(problem, plot_options):

    po = plot_options

    for datatype, composite in problem.composites.items():

        if datatype == 'seismic':
            models_dict = {}
            sc = problem.config.seismic_config

            if sc.gf_config.reference_location is None:
                plot_stations = composite.datahandler.stations
            else:
                plot_stations = [composite.datahandler.stations[0]]
                plot_stations[0].station = \
                    sc.gf_config.reference_location.station

            for station in plot_stations:
                outbasepath = os.path.join(
                    problem.outfolder, po.figure_dir,
                    '%s_%s_velocity_model' % (
                        datatype, station.station))

                if not os.path.exists(outbasepath) or po.force:
                    targets = init_seismic_targets(
                        [station],
                        earth_model_name=sc.gf_config.earth_model_name,
                        channels=sc.get_unique_channels()[0],
                        sample_rate=sc.gf_config.sample_rate,
                        crust_inds=list(range(*sc.gf_config.n_variations)),
                        interpolation='multilinear')
                    store_ids = [t.store_id for t in targets]

                    models = load_earthmodels(
                        composite.engine.store_superdirs[0], store_ids,
                        depth_max=sc.gf_config.depth_limit_variation * km)

                    for i, mods in enumerate(models):
                        if i == 0:
                            site = 'source'
                        elif i == 1:
                            site = 'receiver'

                        outpath = outbasepath + \
                            '_%s.%s' % (site, po.outformat)

                        models_dict[outpath] = mods

                else:
                    logger.info(
                        '%s earthmodel plot for station %s exists. Use '
                        'force=True for replotting!' % (
                            datatype, station.station))

        elif datatype == 'geodetic':
            gc = problem.config.geodetic_config

            models_dict = {}
            outpath = os.path.join(
                problem.outfolder, po.figure_dir,
                '%s_%s_velocity_model.%s' % (
                    datatype, 'psgrn', po.outformat))

            if not os.path.exists(outpath) or po.force:
                targets = init_geodetic_targets(
                    datasets=composite.datasets,
                    earth_model_name=gc.gf_config.earth_model_name,
                    interpolation='multilinear',
                    crust_inds=list(range(*gc.gf_config.n_variations)),
                    sample_rate=gc.gf_config.sample_rate)

                models = load_earthmodels(
                    store_superdir=composite.engine.store_superdirs[0],
                    targets=targets,
                    depth_max=gc.gf_config.source_depth_max * km)
                models_dict[outpath] = models[0]  # select only source site

            else:
                logger.info(
                    '%s earthmodel plot exists. Use force=True for'
                    ' replotting!' % datatype)
                return

        else:
            raise TypeError(
                'Plot for datatype %s not (yet) supported' % datatype)

        figs = []
        axes = []
        tobepopped = []
        for path, models in models_dict.items():
            if len(models) > 0:
                fig, axs = n_model_plot(
                    models, axes=None,
                    draw_bg=po.reference, highlightidx=[0])
                figs.append(fig)
                axes.append(axs)
            else:
                tobepopped.append(path)

        for entry in tobepopped:
            models_dict.pop(entry)

        if po.outformat == 'display':
            plt.show()
        else:
            for fig, outpath in zip(figs, models_dict.keys()):
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)


def fuzzy_waveforms(
        ax, traces, linewidth, zorder=0, extent=None,
        grid_size=(500, 500), cmap=None, alpha=0.6):
    """
    Fuzzy waveforms

    traces : list
        of class:`pyrocko.trace.Trace`, the times of the traces should not
        vary too much
    zorder : int
        the higher number is drawn above the lower number
    extent : list
        of [xmin, xmax, ymin, ymax] (tmin, tmax, min/max of amplitudes)
        if None, the default is to determine it from traces list
    """

    if cmap is None:

        from matplotlib.colors import LinearSegmentedColormap

        ncolors = 256
        cmap = LinearSegmentedColormap.from_list(
            'dummy', ['white', scolor('chocolate2'), scolor('scarletred2')],
            N=ncolors)
        # cmap = plt.cm.gist_earth_r

    if extent is None:
        key = traces[0].channel
        skey = lambda tr: tr.channel

        ymin, ymax = trace.minmax(traces, key=skey)[key]
        xmin, xmax = trace.minmaxtime(traces, key=skey)[key]

        ymax = max(abs(ymin), abs(ymax))
        ymin = -ymax

        extent = [xmin, xmax, ymin, ymax]

    grid = num.zeros(grid_size, dtype='float64')

    for tr in traces:

        draw_line_on_array(
            tr.get_xdata(), tr.ydata,
            grid=grid,
            extent=extent,
            grid_resolution=grid.shape,
            linewidth=linewidth)

    # increase contrast reduce high intense values
    # truncate = len(traces) / 2
    # grid[grid > truncate] = truncate
    ax.imshow(
        grid, extent=extent, origin='lower', cmap=cmap, aspect='auto',
        alpha=alpha, zorder=zorder)


def extract_time_shifts(point, wmap):
    if wmap.config.domain == 'time':
        try:
            time_shifts = point[wmap.time_shifts_id][
                wmap.station_correction_idxs]
        except KeyError:
            raise ValueError(
                'Sampling results do not contain time-shifts for wmap'
                ' %s!' % wmap.time_shifts_id)
    else:
        time_shifts = [0] * wmap.n_t
    return time_shifts


def seismic_fits(problem, stage, plot_options):
    """
    Modified from grond. Plot synthetic and data waveforms and the misfit for
    the selected posterior model.
    """

    time_shift_color = scolor('aluminium3')
    obs_color = scolor('aluminium5')
    syn_color = scolor('scarletred2')
    misfit_color = scolor('scarletred2')

    tap_color_annot = (0.35, 0.35, 0.25)
    tap_color_edge = (0.85, 0.85, 0.80)
    # tap_color_fill = (0.95, 0.95, 0.90)

    composite = problem.composites['seismic']

    lowest_corner = num.min(
        [[filterer.get_lower_corner() for filterer in wmap.filterer]
            for wmap in problem.config.seismic_config.waveforms])
    highest_corner = num.max(
        [[filterer.get_upper_corner() for filterer in wmap.filterer]
            for wmap in problem.config.seismic_config.waveforms])

    fontsize = 8
    fontsize_title = 10

    target_index = dict(
        (target, i) for (i, target) in enumerate(composite.targets))

    po = plot_options

    if not po.reference:
        best_point = get_result_point(stage, problem.config, po.post_llk)
    else:
        best_point = po.reference

    try:
        composite.point2sources(best_point)
        source = composite.sources[0]
        chop_bounds = ['a', 'd']
    except AttributeError:
        logger.info('FFI waveform fit, using reference source ...')
        source = composite.config.gf_config.reference_sources[0]
        source.time = composite.event.time
        chop_bounds = ['b', 'c']

    if best_point:  # for source individual contributions
        bresults = composite.assemble_results(
            best_point, outmode='tapered_data', chop_bounds=chop_bounds)
        synth_plot_flag = True
    else:
        # get dummy results for data
        logger.warning(
            'Got "None" post_llk, still loading MAP for VR calculation')
        best_point = get_result_point(stage, problem.config, 'max')
        bresults = composite.assemble_results(
            best_point, chop_bounds=chop_bounds)
        synth_plot_flag = False

    composite.analyse_noise(best_point, chop_bounds=chop_bounds)
    composite.update_weights(best_point, chop_bounds=chop_bounds)
    if plot_options.nensemble > 1:
        from tqdm import tqdm
        logger.info(
            'Collecting ensemble of %i synthetic waveforms ...' % po.nensemble)
        nchains = len(stage.mtrace)
        csteps = float(nchains) / po.nensemble
        idxs = num.floor(num.arange(0, nchains, csteps)).astype('int32')
        ens_results = []
        points = []
        ens_var_reductions = []
        for idx in tqdm(idxs):
            point = stage.mtrace.point(idx=idx)
            points.append(point)
            results = composite.assemble_results(
                point, chop_bounds=chop_bounds)
            ens_results.append(results)
            ens_var_reductions.append(
                composite.get_variance_reductions(
                    point, weights=composite.weights,
                    results=results, chop_bounds=chop_bounds))

    bvar_reductions = composite.get_variance_reductions(
        best_point, weights=composite.weights,
        results=bresults, chop_bounds=chop_bounds)

    # collecting results for targets
    logger.info('Mapping results to targets ...')

    for target in composite.targets:
        allresults = []
        synths = []
        allvar_reductions = []

        i = target_index[target]

        nslcd_id = target.nslcd_id
        allresults.append(bresults[i])
        synths.append(bresults[i].processed_syn)
        allvar_reductions.append(bvar_reductions[nslcd_id])

        if plot_options.nensemble > 1:
            for results, var_reductions in zip(
                    ens_results, ens_var_reductions):
                # put all results per target here not only single

                allresults.append(results[i])
                synths.append(results[i].processed_syn)
                allvar_reductions.append(var_reductions[nslcd_id])

        target.results = allresults
        target.synths = synths
        target.var_reductions = num.array(allvar_reductions) * 100.

    # collecting time-shifts:
    station_corr = composite.config.station_corrections
    time_shift_bounds = [0, 0]
    if station_corr:
        tshifts = problem.config.problem_config.hierarchicals['time_shift']
        time_shift_bounds = [tshifts.lower, tshifts.upper]

        logger.info('Collecting time-shifts ...')
        if plot_options.nensemble > 1:
            ens_time_shifts = []
            for point in points:
                comp_time_shifts = []
                for wmap in composite.wavemaps:
                    comp_time_shifts.append(
                        extract_time_shifts(point, wmap))

                ens_time_shifts.append(
                    num.hstack(comp_time_shifts))

        btime_shifts = num.hstack(
            [extract_time_shifts(best_point, wmap)
                for wmap in composite.wavemaps])

        logger.info('Mapping time-shifts to targets ...')

        for target in composite.targets:
            target_time_shifts = []
            i = target_index[target]
            target_time_shifts.append(btime_shifts[i])

            if plot_options.nensemble > 1:
                for time_shifts in ens_time_shifts:
                    target_time_shifts.append(time_shifts[i])

            target.time_shifts = num.array(target_time_shifts)

    ### Mahdi target mess
    # gather
    spectrum_targets = dict(
        (utility.list2string(target.codes[:3]), target)
         for target in composite.targets if isinstance(target, SpectrumTarget))

    plotted_spectargets = []
    for target in composite.targets:
        nslc_id = utility.list2string(target.codes[:3])
        if isinstance(target, DynamicTarget) and \
            nslc_id in spectrum_targets and \
                nslc_id not in plotted_spectargets:

            target.spectarget = spectrum_targets[nslc_id]
            plotted_spectargets.append(spectrum_targets[nslc_id].nslcd_id)

    skey = lambda tr: tr.channel

    # plot remaining targets
    cg_to_targets = utility.gather(
        composite.targets,
        lambda t: t.codes[3],
        filter=lambda t: t.nslcd_id not in plotted_spectargets)

    cgs = cg_to_targets.keys()

    figs = []
    logger.info('Plotting waveforms ...')
    for cg in cgs:
        targets = cg_to_targets[cg]

        # can keep from here ... until
        nframes = len(targets)

        nx = int(math.ceil(math.sqrt(nframes)))
        ny = (nframes - 1) // nx + 1

        logger.debug('nx %i, ny %i' % (nx, ny))

        nxmax = 4
        nymax = 4

        nxx = (nx - 1) // nxmax + 1
        nyy = (ny - 1) // nymax + 1

        xs = num.arange(nx) // ((max(2, nx) - 1.0) / 2.)
        ys = num.arange(ny) // ((max(2, ny) - 1.0) / 2.)

        xs -= num.mean(xs)
        ys -= num.mean(ys)

        fxs = num.tile(xs, ny)
        fys = num.repeat(ys, nx)

        data = []
        for target in targets:
            azi = source.azibazi_to(target)[0]
            dist = source.distance_to(target)
            x = dist * num.sin(num.deg2rad(azi))
            y = dist * num.cos(num.deg2rad(azi))
            data.append((x, y, dist))

        gxs, gys, dists = num.array(data, dtype=num.float).T

        iorder = num.argsort(dists)

        gxs = gxs[iorder]
        gys = gys[iorder]
        targets_sorted = [targets[ii] for ii in iorder]

        gxs -= num.mean(gxs)
        gys -= num.mean(gys)

        gmax = max(num.max(num.abs(gys)), num.max(num.abs(gxs)))
        if gmax == 0.:
            gmax = 1.

        gxs /= gmax
        gys /= gmax

        dists = num.sqrt(
            (fxs[num.newaxis, :] - gxs[:, num.newaxis]) ** 2 +
            (fys[num.newaxis, :] - gys[:, num.newaxis]) ** 2)

        distmax = num.max(dists)

        availmask = num.ones(dists.shape[1], dtype=num.bool)
        frame_to_target = {}
        for itarget, target in enumerate(targets_sorted):
            iframe = num.argmin(
                num.where(availmask, dists[itarget], distmax + 1.))
            availmask[iframe] = False
            iy, ix = num.unravel_index(iframe, (ny, nx))
            frame_to_target[iy, ix] = target

        figures = {}
        for iy in range(ny):
            for ix in range(nx):
                if (iy, ix) not in frame_to_target:
                    continue

                ixx = ix // nxmax
                iyy = iy // nymax
                if (iyy, ixx) not in figures:
                    figures[iyy, ixx] = plt.figure(
                        figsize=mpl_papersize('a4', 'landscape'))

                    figures[iyy, ixx].subplots_adjust(
                        left=0.03,
                        right=1.0 - 0.03,
                        bottom=0.03,
                        top=1.0 - 0.06,
                        wspace=0.20,
                        hspace=0.30)

                    figs.append(figures[iyy, ixx])

                logger.debug('iyy %i, ixx %i' % (iyy, ixx))
                logger.debug('iy %i, ix %i' % (iy, ix))
                fig = figures[iyy, ixx]

                target = frame_to_target[iy, ix]

                itarget = target_index[target]

                # get min max of all traces
                key = target.codes[3]
                amin, amax = trace.minmax(
                    target.synths,
                    key=skey)[key]
                # need target specific minmax
                absmax = max(abs(amin), abs(amax))

                ny_this = nymax  # min(ny, nymax)
                nx_this = nxmax  # min(nx, nxmax)
                i_this = (iy % ny_this) * nx_this + (ix % nx_this) + 1
                logger.debug('i_this %i' % i_this)
                logger.debug('Station {}'.format(
                    utility.list2string(target.codes)))
                axes2 = fig.add_subplot(ny_this, nx_this, i_this)

                space = 0.4
                space_factor = 0.7 + space
                axes2.set_axis_off()
                axes2.set_ylim(-1.05 * space_factor, 1.05)

                axes = axes2.twinx()
                axes.set_axis_off()

                ymin, ymax = - absmax * 1.5 * space_factor, absmax * 1.5
                try:
                    axes.set_ylim(ymin, ymax)
                except ValueError:
                    logger.debug(
                        'These traces contain NaN or Inf open in snuffler?')
                    input('Press enter! Otherwise Ctrl + C')
                    from pyrocko.trace import snuffle
                    snuffle(target.synths)
                    
                if isinstance(target, DynamicTarget):
                    target.plot_waveformfits(axes=axes, axes2=axes2, po=po, source=source,
                        time_shift_bounds=time_shift_bounds, synth_plot_flag=synth_plot_flag, absmax=absmax, mode=composite._mode, 
                        fontsize=fontsize, tap_color_edge=tap_color_edge, mpl_graph_color=mpl_graph_color,
                        syn_color=syn_color, obs_color=obs_color, time_shift_color=time_shift_color,
                        tap_color_annot=tap_color_annot)

                    if target.spectarget:
                        target.spectarget.plot_waveformfits(axes=axes2, po=po, synth_plot_flag=synth_plot_flag,
                                        lowest_corner=lowest_corner, highest_corner=highest_corner, fontsize=fontsize, allaxe=False,
                                        syn_color=syn_color, obs_color=obs_color, misfit_color=misfit_color, tap_color_annot=tap_color_annot)

                elif isinstance(target, SpectrumTarget):
                    target.plot_waveformfits(axes=axes, po=po, synth_plot_flag=synth_plot_flag,
                            lowest_corner=lowest_corner, highest_corner=highest_corner, fontsize=fontsize, allaxe=True,
                            syn_color=syn_color, obs_color=obs_color, misfit_color=misfit_color, tap_color_annot=tap_color_annot)
    
                scale_string = None

                infos = []
                if scale_string:
                    infos.append(scale_string)

                infos.append('.'.join(x for x in target.codes if x))
                dist = source.distance_to(target)
                azi = source.azibazi_to(target)[0]
                infos.append(str_dist(dist))
                infos.append('%.0f\u00B0' % azi)
                # infos.append('%.3f' % gcms[itarget])
                axes2.annotate(
                    '\n'.join(infos),
                    xy=(0., 1.),
                    xycoords='axes fraction',
                    xytext=(1., 1.),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    fontsize=fontsize,
                    fontstyle='normal', zorder=10)

                axes2.set_zorder(10)

        for (iyy, ixx), fig in figures.items():
            title = '.'.join(x for x in cg if x)
            if len(figures) > 1:
                title += ' (%i/%i, %i/%i)' % (iyy + 1, nyy, ixx + 1, nxx)

            fig.suptitle(title, fontsize=fontsize_title)

    return figs


def draw_seismic_fits(problem, po):

    if 'seismic' not in list(problem.composites.keys()):
        raise TypeError('No seismic composite defined for this problem!')

    logger.info('Drawing Waveform fits ...')

    stage = Stage(homepath=problem.outfolder,
                  backend=problem.config.sampler_config.backend)

    mode = problem.config.problem_config.mode

    if not po.reference:
        llk_str = po.post_llk
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model, stage_number=po.load_stage,
            load='trace', chains=[-1])
    else:
        llk_str = 'ref'

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'waveforms_%s_%s_%i' % (
            stage.number, llk_str, po.nensemble))

    if not os.path.exists(outpath) or po.force:
        figs = seismic_fits(problem, stage, po)
    else:
        logger.info('waveform plots exist. Use force=True for replotting!')
        return

    if po.outformat == 'display':
        plt.show()
    else:
        logger.info('saving figures to %s' % outpath)
        if po.outformat == 'pdf':
            with PdfPages(outpath + '.pdf') as opdf:
                for fig in figs:
                    opdf.savefig(fig)
        else:
            for i, fig in enumerate(figs):
                fig.savefig(outpath + '_%i.%s' % (i, po.outformat), dpi=po.dpi)


def point2array(point, varnames, rpoint=None):
    """
    Concatenate values of point according to order of given varnames.
    """
    if point is not None:
        array = num.empty((len(varnames)), dtype='float64')
        for i, varname in enumerate(varnames):
            try:
                array[i] = point[varname].ravel()
            except KeyError:  # in case fixed variable
                if rpoint:
                    array[i] = rpoint[varname].ravel()
                else:
                    raise ValueError(
                        'Fixed Component "%s" no fixed value given!' % varname)

        return array
    else:
        return None


def extract_mt_components(problem, po, include_magnitude=False):
    """
    Extract Moment Tensor components from problem results for plotting.
    """
    source_type = problem.config.problem_config.source_type
    if source_type in ['MTSource', 'MTQTSource']:
        varnames = ['mnn', 'mee', 'mdd', 'mne', 'mnd', 'med']
    elif source_type == 'DCSource':
        varnames = ['strike', 'dip', 'rake']
    else:
        raise ValueError(
            'Plot is only supported for point "MTSource" and "DCSource"')

    if include_magnitude:
        varnames += ['magnitude']

    rpoint = None
    stage = load_stage(
        problem, stage_number=po.load_stage, load='trace', chains=[-1])

    n_mts = len(stage.mtrace)
    m6s = num.empty((n_mts, len(varnames)), dtype='float64')
    for i, varname in enumerate(varnames):
        try:
            m6s[:, i] = stage.mtrace.get_values(
                varname, combine=True, squeeze=True).ravel()
        except ValueError:  # if fixed value add that to the ensemble
            rpoint = problem.get_random_point()
            mtfield = num.full_like(
                num.empty((n_mts), dtype=num.float64), rpoint[varname])
            m6s[:, i] = mtfield

    if po.nensemble:
        logger.info(
            'Drawing %i solutions from ensemble ...' % po.nensemble)
        csteps = float(n_mts) / po.nensemble
        idxs = num.floor(
            num.arange(0, n_mts, csteps)).astype('int32')
        m6s = m6s[idxs, :]
    else:
        logger.info('Drawing full ensemble ...')

    if not po.reference:
        llk_str = po.post_llk
        point = get_result_point(stage, problem.config, po.post_llk)
        best_mt = point2array(point, varnames=varnames, rpoint=rpoint)
    else:
        llk_str = 'ref'
        if source_type == 'MTQTSource':
            composite = problem.composites[
                problem.config.problem_config.datatypes[0]]
            composite.point2sources(po.reference)
            best_mt = composite.sources[0].get_derived_parameters()[0:6]

    return m6s, best_mt, llk_str


def draw_ray_piercing_points_bb(
        ax, takeoff_angles_rad, azimuths_rad, polarities,
        size=1, position=(0, 0), transform=None, projection='lambert'):

    # TODO other color coding for any_SH/V radiation patterns?

    toa_idx = takeoff_angles_rad >= (num.pi / 2.)
    takeoff_angles_rad[toa_idx] = num.pi - takeoff_angles_rad[toa_idx]
    azimuths_rad[toa_idx] += num.pi

    # use project instead?
    r = size * num.sqrt(2) * num.sin(0.5 * takeoff_angles_rad)
    x = r * num.sin(azimuths_rad) + position[1]
    y = r * num.cos(azimuths_rad) + position[0]

    xp, yp = x[polarities >= 0], y[polarities >= 0]
    xt, yt = x[polarities < 0], y[polarities < 0]
    ax.plot(
        xp, yp, 'D',
        ms=5, mew=0.5, mec='black', mfc='white', transform=transform)
    ax.plot(
        xt, yt, 's',
        ms=6, mew=0.5, mec='white', mfc='black', transform=transform)


def draw_fuzzy_beachball(problem, po):

    if problem.config.problem_config.n_sources > 1:
        raise NotImplementedError(
            'Fuzzy beachball is not yet implemented for more than one source!')

    if po.load_stage is None:
        po.load_stage = -1

    m6s, best_mt, llk_str = extract_mt_components(problem, po)

    logger.info('Drawing Fuzzy Beachball ...')

    kwargs = {
        'beachball_type': 'full',
        'size': 8,
        'size_units': 'data',
        'linewidth': 2.,
        'alpha': 1.,
        'position': (5, 5),
        'color_t': 'black',
        'edgecolor': 'black',
        'projection': 'lambert',
        'zorder': 0,
        'grid_resolution': 400}

    fig = plt.figure(figsize=(4., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axes = fig.add_subplot(1, 1, 1)

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        'fuzzy_beachball_%i_%s_%i.%s' % (
            po.load_stage, llk_str, po.nensemble, po.outformat))

    if not os.path.exists(outpath) or po.force or po.outformat == 'display':
        transform, position, size = beachball.choose_transform(
            axes, kwargs['size_units'], kwargs['position'], kwargs['size'])

        beachball.plot_fuzzy_beachball_mpl_pixmap(
            m6s, axes, best_mt=best_mt, best_color='white', **kwargs)

        best_amps, bx, by = beachball.mts2amps(
            [best_mt],
            grid_resolution=kwargs['grid_resolution'],
            projection=kwargs['projection'],
            beachball_type=kwargs['beachball_type'],
            mask=False)

        axes.contour(
            position[0] + by * size, position[1] + bx * size, best_amps.T,
            levels=[0.],
            colors=['black'],
            linestyles='dashed',
            linewidths=kwargs['linewidth'],
            transform=transform,
            zorder=kwargs['zorder'],
            alpha=kwargs['alpha'])

        if 'polarity' in problem.config.problem_config.datatypes:
            composite = problem.composites['polarity']
            for pmap in composite.polmaps:
                draw_ray_piercing_points_bb(
                    axes, pmap.get_takeoff_angles_rad(),
                    pmap.get_azimuths_rad(), pmap.dataset,
                    size=size, position=position, transform=transform)

            # axes.legend(['','Compression','Tensile'])

        axes.set_xlim(0., 10.)
        axes.set_ylim(0., 10.)
        axes.set_axis_off()

        if not po.outformat == 'display':
            logger.info('saving figure to %s' % outpath)
            fig.savefig(outpath, dpi=po.dpi)
        else:
            plt.show()

    else:
        logger.info('Plot already exists! Please use --force to overwrite!')


def fuzzy_mt_decomposition(
        axes, list_m6s,
        labels=None, colors=None, fontsize=12):
    """
    Plot fuzzy moment tensor decompositions for list of mt ensembles.
    """
    from pymc3 import quantiles

    logger.info('Drawing Fuzzy MT Decomposition ...')

    # beachball kwargs
    kwargs = {
        'beachball_type': 'full',
        'size': 1.,
        'size_units': 'data',
        'edgecolor': 'black',
        'linewidth': 1,
        'grid_resolution': 200}

    def get_decomps(source_vals):

        isos = []
        dcs = []
        clvds = []
        devs = []
        tots = []
        for val in source_vals:
            m = mt.MomentTensor.from_values(val)
            iso, dc, clvd, dev, tot = m.standard_decomposition()
            isos.append(iso)
            dcs.append(dc)
            clvds.append(clvd)
            devs.append(dev)
            tots.append(tot)
        return isos, dcs, clvds, devs, tots

    yscale = 1.3
    nlines = len(list_m6s)
    nlines_max = nlines * yscale

    if colors is None:
        colors = nlines * [None]

    if labels is None:
        labels = ['Ensemble'] + ([None] * (nlines - 1))

    lines = []
    for i, (label, m6s, color) in enumerate(zip(labels, list_m6s, colors)):
        if color is None:
            color = mpl_graph_color(i)

        lines.append(
            (label, m6s, color))

    magnitude_full_max = max(m6s.mean(axis=0)[-1] for (_, m6s, _) in lines)

    for xpos, label in [
            (0., 'Full'),
            (2., 'Isotropic'),
            (4., 'Deviatoric'),
            (6., 'CLVD'),
            (8., 'DC')]:

        axes.annotate(
            label,
            xy=(1 + xpos, nlines_max),
            xycoords='data',
            xytext=(0., 0.),
            textcoords='offset points',
            ha='center',
            va='center',
            color='black',
            fontsize=fontsize)

    for i, (label, m6s, color_t) in enumerate(lines):
        ypos = nlines_max - (i * yscale) - 1.0
        mean_magnitude = m6s.mean(0)[-1]
        size0 = mean_magnitude / magnitude_full_max

        isos, dcs, clvds, devs, tots = get_decomps(m6s)
        axes.annotate(
            label,
            xy=(-2., ypos),
            xycoords='data',
            xytext=(0., 0.),
            textcoords='offset points',
            ha='left',
            va='center',
            color='black',
            fontsize=fontsize)

        for xpos, decomp, ops in [
                (0., tots, '-'),
                (2., isos, '='),
                (4., devs, '='),
                (6., clvds, '+'),
                (8., dcs, None)]:

            ratios = num.array([comp[1] for comp in decomp])
            ratio = ratios.mean()
            ratios_diff = ratios.max() - ratios.min()

            ratios_qu = quantiles(ratios * 100.)
            mt_parts = [comp[2] for comp in decomp]

            if ratio > 1e-4:
                try:
                    size = math.sqrt(ratio) * 0.95 * size0
                    kwargs['position'] = (1. + xpos, ypos)
                    kwargs['size'] = size
                    kwargs['color_t'] = color_t
                    beachball.plot_fuzzy_beachball_mpl_pixmap(
                        mt_parts, axes, best_mt=None, **kwargs)

                    if ratios_diff > 0.:
                        label = '{:03.1f}-{:03.1f}%'.format(
                            ratios_qu[2.5], ratios_qu[97.5])
                    else:
                        label = '{:03.1f}%'.format(ratios_qu[2.5])

                    axes.annotate(
                        label,
                        xy=(1. + xpos, ypos - 0.65),
                        xycoords='data',
                        xytext=(0., 0.),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        color='black',
                        fontsize=fontsize - 2)

                except beachball.BeachballError as e:
                    logger.warn(str(e))

                    axes.annotate(
                        'ERROR',
                        xy=(1. + xpos, ypos),
                        ha='center',
                        va='center',
                        color='red',
                        fontsize=fontsize)

            else:
                axes.annotate(
                    'N/A',
                    xy=(1. + xpos, ypos),
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=fontsize)

                label = '{:03.1f}%'.format(0.)
                axes.annotate(
                    label,
                    xy=(1. + xpos, ypos - 0.65),
                    xycoords='data',
                    xytext=(0., 0.),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=fontsize - 2)

            if ops is not None:
                axes.annotate(
                    ops,
                    xy=(2. + xpos, ypos),
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=fontsize)

    axes.axison = False
    axes.set_xlim(-2.25, 9.75)
    axes.set_ylim(-0.7, nlines_max + 0.5)
    axes.set_axis_off()


def draw_fuzzy_mt_decomposition(problem, po):

    fontsize = 10

    if problem.config.problem_config.n_sources > 1:
        raise NotImplementedError(
            'Fuzzy MT decomposition is not yet'
            'implemented for more than one source!')

    if po.load_stage is None:
        po.load_stage = -1

    m6s, _, llk_str = extract_mt_components(
        problem, po, include_magnitude=True)

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        'fuzzy_mt_decomposition_%i_%s_%i.%s' % (
            po.load_stage, llk_str, po.nensemble, po.outformat))

    if not os.path.exists(outpath) or po.force or po.outformat == 'display':

        fig = plt.figure(figsize=(6., 2.))
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        axes = fig.add_subplot(1, 1, 1)

        fuzzy_mt_decomposition(axes, list_m6s=[m6s], fontsize=fontsize)

        if not po.outformat == 'display':
            logger.info('saving figure to %s' % outpath)
            fig.savefig(outpath, dpi=po.dpi)
        else:
            plt.show()

    else:
        logger.info('Plot already exists! Please use --force to overwrite!')


def draw_hudson(problem, po):
    """
    Modified from grond. Plot the hudson graph for the reference event(grey)
    and the best solution (red beachball).
    Also a random number of models from the
    selected stage are plotted as smaller beachballs on the hudson graph.
    """

    from pyrocko.plot import beachball, hudson
    from pyrocko import moment_tensor as mtm
    from numpy import random
    if problem.config.problem_config.n_sources > 1:
        raise NotImplementedError(
            'Hudson plot is not yet implemented for more than one source!')

    if po.load_stage is None:
        po.load_stage = -1

    m6s, best_mt, llk_str = extract_mt_components(problem, po)

    logger.info('Drawing Hudson plot ...')

    fontsize = 12
    beachball_type = 'full'
    color = 'red'
    markersize = fontsize * 1.5
    markersize_small = markersize * 0.2
    beachballsize = markersize
    beachballsize_small = beachballsize * 0.5

    fig = plt.figure(figsize=(4., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axes = fig.add_subplot(1, 1, 1)
    hudson.draw_axes(axes)

    data = []
    for m6 in m6s:
        mt = mtm.as_mt(m6)
        u, v = hudson.project(mt)

        if random.random() < 0.05:
            try:
                beachball.plot_beachball_mpl(
                    mt, axes,
                    beachball_type=beachball_type,
                    position=(u, v),
                    size=beachballsize_small,
                    color_t='black',
                    alpha=0.5,
                    zorder=1,
                    linewidth=0.25)
            except beachball.BeachballError as e:
                logger.warn(str(e))

        else:
            data.append((u, v))

    if data:
        u, v = num.array(data).T
        axes.plot(
            u, v, 'o',
            color=color,
            ms=markersize_small,
            mec='none',
            mew=0,
            alpha=0.25,
            zorder=0)

    if best_mt is not None:
        mt = mtm.as_mt(best_mt)
        u, v = hudson.project(mt)

        try:
            beachball.plot_beachball_mpl(
                mt, axes,
                beachball_type=beachball_type,
                position=(u, v),
                size=beachballsize,
                color_t=color,
                alpha=0.5,
                zorder=2,
                linewidth=0.25)
        except beachball.BeachballError as e:
            logger.warn(str(e))

    if isinstance(problem.event.moment_tensor, mtm.MomentTensor):
        mt = problem.event.moment_tensor
        u, v = hudson.project(mt)

        if not po.reference:
            try:
                beachball.plot_beachball_mpl(
                    mt, axes,
                    beachball_type=beachball_type,
                    position=(u, v),
                    size=beachballsize,
                    color_t='grey',
                    alpha=0.5,
                    zorder=2,
                    linewidth=0.25)
                logger.info('drawing reference event in grey ...')
            except beachball.BeachballError as e:
                logger.warn(str(e))
    else:
        logger.info(
            'No reference event moment tensor information given, '
            'skipping drawing ...')

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        'hudson_%i_%s_%i.%s' % (
            po.load_stage, llk_str, po.nensemble, po.outformat))

    if not os.path.exists(outpath) or po.force or po.outformat == 'display':

        if not po.outformat == 'display':
            logger.info('saving figure to %s' % outpath)
            fig.savefig(outpath, dpi=po.dpi)
        else:
            plt.show()

    else:
        logger.info('Plot already exists! Please use --force to overwrite!')



def draw_data_stations(
        gmt, stations, data, dist, data_cpt=None,
        scale_label=None, *args):
    """
    Draw MAP time-shifts at station locations as colored triangles
    """
    miny = data.min()
    maxy = data.max()
    bound = num.ceil(max(num.abs(miny), maxy))

    if data_cpt is None:
        data_cpt = '/tmp/tempfile.cpt'

        gmt.makecpt(
            C='blue,white,red',
            Z=True,
            T='%g/%g' % (-bound, bound),
            out_filename=data_cpt, suppress_defaults=True)

    for i, station in enumerate(stations):
        logger.debug('%s, %f' % (station.station, data[i]))

    st_lons = [station.lon for station in stations]
    st_lats = [station.lat for station in stations]

    gmt.psxy(
        in_columns=(st_lons, st_lats, data.tolist()),
        C=data_cpt,
        *args)

    if dist > 30.:
        D = 'x1.25c/0c+w5c/0.5c+jMC+h'
        F = False
    else:
        D = 'x5.5c/4.1c+w5c/0.5c+jMC+h'
        F = '+gwhite'

    if scale_label:
        # add a colorbar
        gmt.psscale(
            B='xa%s +l %s' % (num.floor(bound), scale_label),
            D=D,
            F=F,
            C=data_cpt)
    else:
        logger.info('Not plotting scale as "scale_label" is None')


def draw_events(gmt, events, *args, **kwargs):

    ev_lons = [ev.lon for ev in events]
    ev_lats = [ev.lat for ev in events]

    gmt.psxy(
        in_columns=(ev_lons, ev_lats),
        *args, **kwargs)


def gmt_station_map_azimuthal(
        gmt, stations, event, data_cpt=None,
        data=None, max_distance=90, width=20, bin_width=15,
        fontsize=12, font='1', plot_names=True, scale_label='time-shifts [s]'):
    """
    Azimuth equidistant station map, if data given stations are colored
    accordingly

    Parameters
    ----------
    gmt : :class:`pyrocko.plot.gmtpy.GMT`
    stations : list
        of :class:`pyrocko.model.station.Station`
    event : :class:`pyrocko.model.event.Event`
    data_cpt : str
        path to gmt '*.cpt' file for coloring
    data : :class:`numoy.NdArray`
        1d vector length of stations to color stations
    max_distance : float
        maximum distance [deg] of event to map bound
    width : float
        plot width [cm]
    bin_width : float
        grid spacing [deg] for distance/ azimuth grid
    fontsize : int
        font-size in points for station labels
    font : str
        GMT font specification (number or name)
    """

    max_distance = max_distance * 1.05  # add interval to have bound

    J_basemap = 'E0/-90/%s/%i' % (max_distance, width)
    J_location = 'E%s/%s/%s/%i' % (event.lon, event.lat, max_distance, width)
    R_location = '0/360/-90/0'

    gmt.psbasemap(
        R=R_location,
        J='S0/-90/90/%i' % width,
        B='xa%sf%s' % (bin_width * 2, bin_width))
    gmt.pscoast(
        R='g',
        J=J_location,
        D='c',
        G='darkgrey')

    # plotting equal distance circles
    bargs = ['-Bxg%f' % bin_width, '-Byg%f' % (2 * bin_width)]
    gmt.psbasemap(
        R='g', J=J_basemap, *bargs)

    if data is not None:
        draw_data_stations(
            gmt, stations, data, max_distance, data_cpt, scale_label, *(
                '-J%s' % J_location, '-R%s' % R_location, '-St14p'))
    else:
        st_lons = [station.lon for station in stations]
        st_lats = [station.lat for station in stations]

        gmt.psxy(
            R=R_location,
            J=J_location,
            in_columns=(st_lons, st_lats),
            G='red',
            S='t14p')

    if plot_names:
        rows = []
        alignment = 'TC'
        for st in stations:
            if gmt.is_gmt5():
                row = (
                    st.lon, st.lat,
                    '%i,%s,%s' % (fontsize, font, 'black'),
                    alignment,
                    '{}.{}'.format(st.network, st.station))
                farg = ['-F+f+j']
            else:
                raise gmtpy.GmtPyError('Only GMT version 5.x supported!')

            rows.append(row)

        gmt.pstext(
            in_rows=rows,
            R=R_location,
            J=J_location,
            N=True, *farg)

    draw_events(
        gmt, [event], *('-J%s' % J_location, '-R%s' % R_location),
        **dict(G='orange', S='a14p'))


def draw_station_map_gmt(problem, po):
    """
    Draws distance dependend for teleseismic vs regional/local setups
    """

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError(
            'GMT needs to be installed for station_map plot!')

    if po.outformat == 'svg':
        raise NotImplementedError('SVG format is not supported for this plot!')

    ts = 'time_shift'
    if ts in po.varnames:
        logger.info('Plotting time-shifts on station locations')
        stage = load_stage(
            problem, stage_number=po.load_stage, load='trace', chains=[-1])

        point = get_result_point(stage, problem.config, po.post_llk)
        value_string = '%i' % po.load_stage
    else:
        point = None
        value_string = '0'
        if len(po.varnames) > 0:
            raise ValueError(
                'Requested variables %s is not supported for plotting!'
                'Supported: %s' % (utility.list2string(po.varnames), ts))

    fontsize = 12
    font = '1'
    bin_width = 15  # major grid and tick increment in [deg]
    h = 15  # outsize in cm
    w = h - 5

    logger.info('Drawing Station Map ...')
    sc = problem.composites['seismic']
    event = problem.config.event

    gmtconfig = get_gmt_config(gmtpy, h=h, w=h)
    gmtconfig['MAP_LABEL_OFFSET'] = '4p'
    for wmap in sc.wavemaps:
        outpath = os.path.join(
            problem.outfolder, po.figure_dir, 'station_map_%s_%i_%s.%s' % (
                wmap.name, wmap.mapnumber, value_string, po.outformat))

        dist = max(wmap.config.distances)
        if not os.path.exists(outpath) or po.force:

            if point:
                time_shifts = extract_time_shifts(point, wmap)
            else:
                time_shifts = None

            if dist > 30:
                logger.info(
                    'Using equidistant azimuthal projection for'
                    ' teleseismic setup of wavemap %s.' % wmap._mapid)

                gmt = gmtpy.GMT(config=gmtconfig)
                gmt_station_map_azimuthal(
                    gmt, wmap.stations, event,
                    data=time_shifts, max_distance=dist, width=w,
                    bin_width=bin_width,
                    fontsize=fontsize, font=font)

                gmt.save(outpath, resolution=po.dpi, size=w)

            else:
                logger.info(
                    'Using equidistant projection for regional setup '
                    'of wavemap %s.' % wmap._mapid)
                from pyrocko.automap import Map
                m = Map(
                    lat=event.lat,
                    lon=event.lon,
                    radius=dist * otd.d2m,
                    width=h,
                    height=h,
                    show_grid=True,
                    show_topo=True,
                    show_scale=True,
                    color_dry=(143, 188, 143),  # grey
                    illuminate=True,
                    illuminate_factor_ocean=0.15,
                    # illuminate_factor_land = 0.2,
                    show_rivers=True,
                    show_plates=False,
                    gmt_config=gmtconfig)

                if time_shifts:
                    sargs = m.jxyr + ['-St14p']
                    draw_data_stations(
                        m.gmt, wmap.stations, time_shifts, dist,
                        data_cpt=None, scale_label='time shifts [s]', *sargs)

                    for st in wmap.stations:
                        text = '{}.{}'.format(st.network, st.station)
                        m.add_label(lat=st.lat, lon=st.lon, text=text)
                else:
                    m.add_stations(
                        wmap.stations, psxy_style=dict(S='t14p', G='red'))

                draw_events(
                    m.gmt, [event], *m.jxyr, **dict(G='yellow', S='a14p'))
                m.save(outpath, resolution=po.dpi, oversample=2., size=w)

            logger.info('saving figure to %s' % outpath)
        else:
            logger.info('Plot exists! Use --force to overwrite!')


def draw_lune_plot(problem, po):

    if po.outformat == 'svg':
        raise NotImplementedError('SVG format is not supported for this plot!')

    if problem.config.problem_config.n_sources > 1:
        raise NotImplementedError(
            'Lune plot is not yet implemented for more than one source!')

    if po.load_stage is None:
        po.load_stage = -1

    stage = load_stage(
        problem, stage_number=po.load_stage, load='trace', chains=[-1])
    n_mts = len(stage.mtrace)

    result_ensemble = {}
    for varname in ['v', 'w']:
        try:
            result_ensemble[varname] = stage.mtrace.get_values(
                varname, combine=True, squeeze=True).ravel()
        except ValueError:  # if fixed value add that to the ensemble
            rpoint = problem.get_random_point()
            result_ensemble[varname] = num.full_like(
                num.empty((n_mts), dtype=num.float64), rpoint[varname])

    if po.reference:
        reference_v_tape = po.reference['v']
        reference_w_tape = po.reference['w']
        llk_str = 'ref'
    else:
        reference_v_tape = None
        reference_w_tape = None
        llk_str = po.post_llk

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        'lune_%i_%s_%i.%s' % (
            po.load_stage, llk_str, po.nensemble, po.outformat))

    if po.nensemble > 1:
        logger.info('Plotting selected ensemble as nensemble > 1 ...')
        selected = num.linspace(
            0, n_mts, po.nensemble, dtype='int', endpoint=False)
        v_tape = result_ensemble['v'][selected]
        w_tape = result_ensemble['w'][selected]
    else:
        logger.info('Plotting whole posterior ...')
        v_tape = result_ensemble['v']
        w_tape = result_ensemble['w']

    if not os.path.exists(outpath) or po.force or po.outformat == 'display':
        logger.info('Drawing Lune plot ...')
        gmt = lune_plot(
            v_tape=v_tape, w_tape=w_tape,
            reference_v_tape=reference_v_tape,
            reference_w_tape=reference_w_tape)

        logger.info('saving figure to %s' % outpath)
        gmt.save(outpath, resolution=300, size=10)
    else:
        logger.info('Plot exists! Use --force to overwrite!')


def lune_plot(
        v_tape=None, w_tape=None,
        reference_v_tape=None, reference_w_tape=None):

    from beat.sources import v_to_gamma, w_to_delta

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError(
            'GMT needs to be installed for lune_plot!')

    fontsize = 14
    font = '1'

    def draw_lune_arcs(gmt, R, J):

        lons = [30., -30., 30., -30.]
        lats = [54.7356, 35.2644, -35.2644, -54.7356]

        gmt.psxy(
            in_columns=(lons, lats), N=True, W='1p,black', R=R, J=J)

    def draw_lune_points(gmt, R, J, labels=True):

        lons = [0., -30., -30., -30., 0., 30., 30., 30., 0.]
        lats = [-90., -54.7356, 0., 35.2644, 90., 54.7356, 0., -35.2644, 0.]
        annotations = [
            '-ISO', '', '+CLVD', '+LVD', '+ISO', '', '-CLVD', '-LVD', 'DC']
        alignments = ['TC', 'TC', 'RM', 'RM', 'BC', 'BC', 'LM', 'LM', 'TC']

        gmt.psxy(in_columns=(lons, lats), N=True, S='p6p', W='1p,0', R=R, J=J)

        rows = []
        if labels:
            farg = ['-F+f+j']
            for lon, lat, text, align in zip(
                    lons, lats, annotations, alignments):

                rows.append((
                    lon, lat,
                    '%i,%s,%s' % (fontsize, font, 'black'),
                    align, text))

            gmt.pstext(
                in_rows=rows,
                N=True, R=R, J=J, D='j5p', *farg)

    def draw_lune_kde(
            gmt, v_tape, w_tape, grid_size=(200, 200), R=None, J=None):

        def check_fixed(a, varname):
            if a.std() < 0.1:
                logger.info(
                    'Spread of variable "%s" is %f, which is below necessary'
                    ' width to estimate a spherical kde, adding some jitter to'
                    ' make kde estimate possible' % (varname, a.std()))
                a += num.random.normal(loc=0., scale=0.05, size=a.size)

        gamma = num.rad2deg(v_to_gamma(v_tape))   # lune longitude [rad]
        delta = num.rad2deg(w_to_delta(w_tape))   # lune latitude [rad]

        check_fixed(gamma, varname='v')
        check_fixed(delta, varname='w')

        lats_vec, lats_inc = num.linspace(
            -90., 90., grid_size[0], retstep=True)
        lons_vec, lons_inc = num.linspace(
            -30., 30., grid_size[1], retstep=True)
        lons, lats = num.meshgrid(lons_vec, lats_vec)

        kde_vals, _, _ = spherical_kde_op(
            lats0=delta, lons0=gamma,
            lons=lons, lats=lats, grid_size=grid_size)
        Tmin = num.min([0., kde_vals.min()])
        Tmax = num.max([0., kde_vals.max()])

        cptfilepath = '/tmp/tempfile.cpt'
        gmt.makecpt(
            C='white,yellow,orange,red,magenta,violet',
            Z=True, D=True,
            T='%f/%f' % (Tmin, Tmax),
            out_filename=cptfilepath, suppress_defaults=True)

        grdfile = gmt.tempfilename()
        gmt.xyz2grd(
            G=grdfile, R=R, I='%f/%f' % (lons_inc, lats_inc),
            in_columns=(lons.ravel(), lats.ravel(), kde_vals.ravel()),  # noqa
            out_discard=True)

        gmt.grdimage(grdfile, R=R, J=J, C=cptfilepath)

        # gmt.pscontour(
        #    in_columns=(lons.ravel(), lats.ravel(),  kde_vals.ravel()),
        #    R=R, J=J, I=True, N=True, A=True, C=cptfilepath)
        # -Ctmp_$out.cpt -I -N -A- -O -K >> $ps

    def draw_reference_lune(gmt, R, J, reference_v_tape, reference_w_tape):

        gamma = num.rad2deg(
            v_to_gamma(reference_v_tape))  # lune longitude [rad]
        delta = num.rad2deg(
            w_to_delta(reference_w_tape))   # lune latitude [rad]

        gmt.psxy(
            in_rows=[(float(gamma), float(delta))],
            N=True, G='blue', W='1p,black', S='p3p', R=R, J=J)

    h = 20.
    w = h / 1.9

    gmtconfig = get_gmt_config(gmtpy, h=h, w=w)
    bin_width = 15  # tick increment

    J = 'H0/%f' % (w - 5.)
    R = '-30/30/-90/90'
    B = 'f%ig%i/f%ig%i' % (bin_width, bin_width, bin_width, bin_width)
    # range_arg="-T${zmin}/${zmax}/${dz}"

    gmt = gmtpy.GMT(config=gmtconfig)

    draw_lune_kde(
        gmt, v_tape=v_tape, w_tape=w_tape, grid_size=(701, 301), R=R, J=J)
    gmt.psbasemap(R=R, J=J, B=B)
    draw_lune_arcs(gmt, R=R, J=J)
    draw_lune_points(gmt, R=R, J=J)

    if reference_v_tape is not None:
        draw_reference_lune(
            gmt, R=R, J=J,
            reference_v_tape=reference_v_tape,
            reference_w_tape=reference_w_tape)

    return gmt


def draw_station_map_cartopy(problem, po):
    import matplotlib.ticker as mticker

    logger.info('Drawing Station Map ...')
    try:
        import cartopy as ctp
    except ImportError:
        logger.error(
            'Cartopy is not installed.'
            'For a station map cartopy needs to be installed!')
        return

    def draw_gridlines(ax):
        gl = ax.gridlines(crs=grid_proj, color='black', linewidth=0.5)
        gl.n_steps = 300
        gl.xlines = False
        gl.ylocator = mticker.FixedLocator([30, 60, 90])

    fontsize = 12

    if 'seismic' not in problem.config.problem_config.datatypes:
        raise TypeError(
            'Station map is available only for seismic stations!'
            ' However, the datatypes do not include "seismic" data')

    event = problem.config.event

    sc = problem.composites['seismic']

    mpl_init(fontsize=fontsize)
    stations_proj = ctp.crs.PlateCarree()
    for wmap in sc.wavemaps:
        outpath = os.path.join(
            problem.outfolder, po.figure_dir, 'station_map_%s_%i.%s' % (
                wmap.name, wmap.mapnumber, po.outformat))

        if not os.path.exists(outpath) or po.force:
            if max(wmap.config.distances) > 30:
                map_proj = ctp.crs.Orthographic(
                    central_longitude=event.lon, central_latitude=event.lat)
                extent = None
            else:
                max_dist = math.ceil(wmap.config.distances[1])
                map_proj = ctp.crs.PlateCarree()
                extent = [
                    event.lon - max_dist, event.lon + max_dist,
                    event.lat - max_dist, event.lat + max_dist]

            grid_proj = ctp.crs.RotatedPole(
                pole_longitude=event.lon, pole_latitude=event.lat)
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=mpl_papersize('a6', 'landscape'),
                subplot_kw={'projection': map_proj})

            stations_meta = [
                (station.lat, station.lon, station.station)
                for station in wmap.stations]

            if extent:
                # regional map
                labelpos = mpl_margins(
                    fig, left=2, bottom=2, top=2, right=2, units=fontsize)

                import cartopy.feature as cfeature
                from cartopy.mpl.gridliner import \
                    LONGITUDE_FORMATTER, LATITUDE_FORMATTER

                ax.set_extent(extent, crs=map_proj)
                ax.add_feature(cfeature.NaturalEarthFeature(
                    category='physical', name='land',
                    scale='50m', **cfeature.LAND.kwargs))
                ax.add_feature(cfeature.NaturalEarthFeature(
                    category='physical', name='ocean',
                    scale='50m', **cfeature.OCEAN.kwargs))

                gl = ax.gridlines(
                    color='black', linewidth=0.5, draw_labels=True)
                gl.ylocator = tick.MaxNLocator(nbins=5)
                gl.xlocator = tick.MaxNLocator(nbins=5)
                gl.xlabels_top = False
                gl.ylabels_right = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER

            else:
                # global teleseismic map
                labelpos = mpl_margins(
                    fig, left=1, bottom=1, top=1, right=1, units=fontsize)

                ax.coastlines(linewidth=0.2)
                draw_gridlines(ax)
                ax.stock_img()

            for (lat, lon, name) in stations_meta:
                ax.plot(
                    lon, lat, 'r^', transform=stations_proj,
                    markeredgecolor='black', markeredgewidth=0.3)
                ax.text(
                    lon, lat, name, fontsize=10, transform=stations_proj,
                    horizontalalignment='center', verticalalignment='top')

            ax.plot(
                event.lon, event.lat, '*', transform=stations_proj,
                markeredgecolor='black', markeredgewidth=0.3, markersize=12,
                markerfacecolor=scolor('butter1'))
            if po.outformat == 'display':
                plt.show()
            else:
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)

        else:
            logger.info('Plot exists! Use --force to overwrite!')

