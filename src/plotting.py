from pyrocko import cake_plot as cp
from pymc3 import plots as pmp

import math
import os
import logging
import copy

from beat import utility, backend, config
from beat.models import load_stage
from beat.metropolis import get_trace_stats
from beat.heart import init_targets

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.ndimage.interpolation import rotate

import numpy as num
from pyrocko.guts import Object, String, Dict, Bool, Int
from pyrocko import util, trace
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light

from pyrocko.plot import mpl_papersize

logger = logging.getLogger('plotting')

km = 1000.


__all__ = ['PlotOptions', 'correlation_plot', 'correlation_plot_hist',
    'get_result_point', 'seismic_fits', 'geodetic_fits', 'traceplot',
    'select_transform']

u_km = '[km]'
u_deg = '[deg]'
u_m = '[m]'
u_s = '[s]'
u_hyp = ''

plot_units = {
    'east_shift': u_km,
    'north_shift': u_km,
    'depth': u_km,
    'width': u_km,
    'length': u_km,
    'dip': u_deg,
    'strike': u_deg,
    'rake': u_deg,
    'slip': u_m,
    'time': u_s,
    'duration': u_s,
    'geo_S': u_hyp,
    'geo_G': u_hyp,
    'seis_Z': u_hyp,
    'seis_T': u_hyp,
    'like': u_hyp,
            }


class PlotOptions(Object):
    post_llk = String.T(
        default='max',
        help='Which model to plot on the specified plot; Default: "max";'
             ' Options: "max", "min", "mean", "all"')
    plot_projection = String.T(
        default='utm',
        help='Projection to use for plotting geodetic data; options: "latlon"')
    utm_zone = Int.T(
        default=36,
        optional=True,
        help='Only relevant if plot_projection is "utm"')
    load_stage = String.T(
        default='final',
        help='Which stage to select for plotting')
    figure_dir = String.T(
        default='figures',
        help='Name of the output directory of plots')
    reference = Dict.T(
        help='Reference point for example from a synthetic test.',
        optional=True)
    outformat = String.T(default='pdf')
    dpi = Int.T(default=300)
    force = Bool.T(default=False)


def str_dist(dist):
    """
    Return string representation of distance.
    """
    if dist < 10.0:
        return '%g m' % dist
    elif 10. <= dist < 1. * km:
        return '%.0f m' % dist
    elif 1. * km <= dist < 10. * km:
        return '%.1f km' % (dist / km)
    else:
        return '%.0f km' % (dist / km)


def str_duration(t):
    """
    Convert time to str representation.
    """
    s = ''
    if t < 0.:
        s = '-'

    t = abs(t)

    if t < 10.0:
        return s + '%.2g s' % t
    elif 10.0 <= t < 3600.:
        return s + util.time_to_str(t, format='%M:%S min')
    elif 3600. <= t < 24 * 3600.:
        return s + util.time_to_str(t, format='%H:%M h')
    else:
        return s + '%.1f d' % (t / (24. * 3600.))


def choose_round_digit(twosigma):
    if twosigma < 0.01:
        return 3
    elif twosigma < 0.1:
        return 2
    elif twosigma < 1.:
        return 1
    elif twosigma < 10.:
        return 0
    elif twosigma < 100.:
        return -1
    elif twosigma < 1000.:
        return -2
    elif twosigma < 10000.:
        return -3
    else:
        return -4


def get_tickmarks(leftb, rightb, ntickmarks=5):
    """
    Get tickmarks according to range of given values and number of tickmarks!
    """
    digits = choose_round_digit((rightb - leftb) / ntickmarks)
    return num.round(
            num.linspace(leftb, rightb, ntickmarks), digits).tolist()


def correlation_plot(mtrace, varnames=None,
        transform=lambda x: x, figsize=None, cmap=None, grid=200, point=None,
        point_style='.', point_color='white', point_size='8'):
    """
    Plot 2d marginals (with kernel density estimation) showing the correlations
    of the model parameters.

    Parameters
    ----------
    mtrace : :class:`pymc3.base.MutliTrace`
        Mutlitrace instance containing the sampling results
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    cmap : matplotlib colormap
    grid : resolution of kernel density estimation
    point : dict
        Dictionary of variable name / value  to be overplotted as marker
        to the posteriors e.g. mean of posteriors, true values of a simulation
    point_style : str
        style of marker according to matplotlib conventions
    point_color : str or tuple of 3
        color according to matplotlib convention
    point_size : str
        marker size according to matplotlib conventions

    Returns
    -------
    fig : figure object
    axs : subplot axis handles
    """

    if varnames is None:
        varnames = mtrace.varnames

    nvar = len(varnames)

    if figsize is None:
        figsize = mpl_papersize('a4', 'landscape')

    fig, axs = plt.subplots(sharey='row', sharex='col',
        nrows=nvar - 1, ncols=nvar - 1, figsize=figsize)

    d = dict()
    for var in varnames:
        d[var] = transform(mtrace.get_values(
                var, combine=True, squeeze=True))

    for k in range(nvar - 1):
        a = d[varnames[k]]
        for l in range(k + 1, nvar):
            logger.debug('%s, %s' % (varnames[k], varnames[l]))
            b = d[varnames[l]]

            pmp.kde2plot(
                a, b, grid=grid, ax=axs[l - 1, k], cmap=cmap, aspect='auto')

            if point is not None:
                axs[l - 1, k].plot(point[varnames[k]], point[varnames[l]],
                    color=point_color, marker=point_style,
                    markersize=point_size)

            axs[l - 1, k].tick_params(direction='in')

            if k == 0:
                axs[l - 1, k].set_ylabel(varnames[l])

        axs[l - 1, k].set_xlabel(varnames[k])

    for k in range(nvar - 1):
        for l in range(k):
            fig.delaxes(axs[l, k])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig, axs


def correlation_plot_hist(mtrace, varnames=None,
        transform=lambda x: x, figsize=None, hist_color='orange', cmap=None,
        grid=50, chains=None, ntickmarks=2, point=None,
        point_style='.', point_color='red', point_size='6', alpha=0.35):
    """
    Plot 2d marginals (with kernel density estimation) showing the correlations
    of the model parameters. In the main diagonal is shown the parameter
    histograms.

    Parameters
    ----------
    mtrace : :class:`pymc3.base.MutliTrace`
        Mutlitrace instance containing the sampling results
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    cmap : matplotlib colormap
    hist_color : str or tuple of 3
        color according to matplotlib convention
    grid : resolution of kernel density estimation
    chains : int or list of ints
        chain indexes to select from the trace
    ntickmarks : int
        number of ticks at the axis labels
    point : dict
        Dictionary of variable name / value  to be overplotted as marker
        to the posteriors e.g. mean of posteriors, true values of a simulation
    point_style : str
        style of marker according to matplotlib conventions
    point_color : str or tuple of 3
        color according to matplotlib convention
    point_size : str
        marker size according to matplotlib conventions

    Returns
    -------
    fig : figure object
    axs : subplot axis handles
    """

    logger.info('Drawing correlation figure ...')

    if varnames is None:
        varnames = mtrace.varnames

    nvar = len(varnames)

    if figsize is None:
        figsize = mpl_papersize('a4', 'landscape')

    fig, axs = plt.subplots(nrows=nvar, ncols=nvar, figsize=figsize,
            subplot_kw={'adjustable': 'box-forced'})

    d = dict()

    for var in varnames:
        d[var] = transform(mtrace.get_values(
                var, chains=chains, combine=True, squeeze=True))

    for k in range(nvar):
        v_namea = varnames[k]
        a = d[v_namea]

        for l in range(k, nvar):
            v_nameb = varnames[l]
            logger.debug('%s, %s' % (v_namea, v_nameb))
            if l == k:
                if point is not None:
                    reference = point[v_namea]
                    axs[l, k].axvline(
                        x=reference, color=point_color,
                        lw=int(point_size) / 4.)
                else:
                    reference = None

                histplot_op(
                    axs[l, k], pmp.make_2d(a), alpha=alpha, color='orange',
                    tstd=0., reference=reference, ntickmarks=ntickmarks)
                axs[l, k].get_yaxis().set_visible(False)

                xticks = axs[l, k].get_xticks()
            else:
                b = d[v_nameb]

                pmp.kde2plot(
                    a, b, grid=grid, ax=axs[l, k], cmap=cmap, aspect='auto')

                bmin = b.min()
                bmax = b.max()

                if point is not None:
                    axs[l, k].plot(point[v_namea], point[v_nameb],
                        color=point_color, marker=point_style,
                        markersize=point_size)

                    bmin = num.minimum(bmin, point[v_nameb])
                    bmax = num.maximum(bmax, point[v_nameb])

                ytickmarks = get_tickmarks(bmin, bmax, ntickmarks=ntickmarks)
                axs[l, k].set_xticks(xticks)
                axs[l, k].set_yticks(ytickmarks)

            if l != nvar - 1:
                axs[l, k].get_xaxis().set_ticklabels([])

            if k == 0:
                axs[l, k].set_ylabel(v_nameb + '\n ' + plot_units[v_nameb])
            else:
                axs[l, k].get_yaxis().set_ticklabels([])

            axs[l, k].tick_params(direction='in')

        axs[l, k].set_xlabel(v_namea + '\n ' + plot_units[v_namea])

    for k in range(nvar):
        for l in range(k):
            fig.delaxes(axs[l, k])

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axs


def plot(uwifg, point_size=20):
    """
    Very simple scatter plot of given IFG for fast inspections.

    Parameters
    ----------
    point_size : int
        determines the size of the scatter plot points
    """

    #colim = num.max([disp.max(), num.abs(disp.min())])
    ax = plt.axes()
    im = ax.scatter(uwifg.lons, uwifg.lats, point_size, uwifg.displacement,
        edgecolors='none')
    plt.colorbar(im)
    plt.title('Displacements [m] %s' % uwifg.track)
    plt.show()


def plot_cov(target, point_size=20):

    ax = plt.axes()
    im = ax.scatter(target.lons, target.lats, point_size,
             num.array(target.covariance.pred_v.sum(axis=0)).flatten(),
             edgecolors='none')
    plt.colorbar(im)
    plt.title('Prediction Covariance [m2] %s' % target.track)
    plt.show()


def plot_matrix(A):
    """
    Very simple plot of a matrix for fast inspections.
    """
    ax = plt.axes()
    im = ax.matshow(A)
    plt.colorbar(im)
    plt.show()


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
        sc = config.sampler_config.parameters
        pdict, _ = get_trace_stats(
            stage.mtrace, stage.step, sc.burn, sc.thin)
        point = pdict[point_llk]

    elif config.sampler_config.name == 'SMC':
        _, _, llk = stage.step.select_end_points(stage.mtrace)
        posterior_idxs = utility.get_fit_indexes(llk)

        n_steps = config.sampler_config.parameters.n_steps - 1
        point = stage.mtrace.point(
            idx=n_steps, chain=posterior_idxs[point_llk])

    return point


def plot_scene(ax, target, data, scattersize, colim,
               outmode='latlon', **kwargs):
    if outmode == 'latlon':
        x = target.lons
        y = target.lats
    elif outmode == 'utm':
        x = target.utme / km
        y = target.utmn / km
    elif outmode == 'local':
        x = target.locx / km
        y = target.locy / km

    return ax.scatter(
        x, y, scattersize, data,
        edgecolors='none', vmin=-colim, vmax=colim, **kwargs)


def geodetic_fits(problem, stage, plot_options):
    """
    Plot geodetic data, synthetics and residuals.
    """
    scattersize = 16
    fontsize = 10
    fontsize_title = 12
    ndmax = 3
    nxmax = 3
    cmap = plt.cm.jet

    po = plot_options

    composite = problem.composites['geodetic']

    if po.reference is not None:
        composite.point2sources(po.reference)
        ref_sources = copy.deepcopy(composite.sources)

    target_index = dict(
        (target, i) for (i, target) in enumerate(composite.targets))

    point = get_result_point(stage, problem.config, po.post_llk)

    results = composite.assemble_results(point)
    nrmax = len(results)

    target_to_result = {}
    for target, result in zip(composite.targets, results):
        target_to_result[target] = result

    nfigs = int(num.ceil(float(nrmax) / float(ndmax)))

    figures = []
    axes = []

    for f in range(nfigs):
        fig, ax = plt.subplots(
            nrows=ndmax, ncols=nxmax, figsize=mpl_papersize('a4', 'portrait'))
        fig.tight_layout()
        fig.subplots_adjust(
                        left=0.08,
                        right=1.0 - 0.03,
                        bottom=0.06,
                        top=1.0 - 0.06,
                        wspace=0.,
                        hspace=0.3)
        figures.append(fig)
        axes.append(ax)

    def axis_config(axes, po):
        axes[1].get_yaxis().set_ticklabels([])
        axes[2].get_yaxis().set_ticklabels([])
        axes[1].get_xaxis().set_ticklabels([])
        axes[2].get_xaxis().set_ticklabels([])

        if po.plot_projection == 'latlon':
            ystr = 'Latitude [deg]'
            xstr = 'Longitude [deg]'
        elif po.plot_projection == 'utm':
            ystr = 'UTM Northing [km]'
            xstr = 'UTM Easting [km]'
        elif po.plot_projection == 'local':
            ystr = 'Distance [km]'
            xstr = 'Distance [km]'
        else:
            raise Exception(
                'Plot projection %s not available' % po.plot_projection)

        axes[0].set_ylabel(ystr, fontsize=fontsize)
        axes[0].set_xlabel(xstr, fontsize=fontsize)

    def draw_sources(ax, sources, po, **kwargs):
        for source in sources:
            if po.plot_projection == 'latlon':
                outline = source.outline(cs='lonlat')
            elif po.plot_projection == 'utm':
                outline = source.outline(cs='lonlat')
                utme, utmn = utility.lonlat_to_utm(
                    lon=outline[:, 0], lat=outline[:, 1], zone=po.utm_zone)
                outline = num.vstack([utme / km, utmn / km]).T
            elif po.plot_projection == 'local':
                outline = source.outline(cs='xy')
            ax.plot(outline[:, 0], outline[:, 1], '-', linewidth=1.0, **kwargs)
            ax.plot(
                outline[0:2, 0], outline[0:2, 1], '-k', linewidth=1.0)

    def cbtick(x):
        rx = math.floor(x * 1000.) / 1000.
        return [-rx, rx]

    def str_title(track):
        if track[0] == 'A':
            orbit = 'ascending'
        elif track[0] == 'D':
            orbit = 'descending'

        title = 'Orbit: ' + orbit
        return title

    orbits_to_targets = utility.gather(
        composite.targets,
        lambda t: t.track,
        filter=lambda t: t in target_to_result)

    ott = orbits_to_targets.keys()

    colims = [num.max([
        num.max(num.abs(r.processed_obs)),
        num.max(num.abs(r.processed_syn))]) for r in results]
    dcolims = [num.max(num.abs(r.processed_res)) for r in results]

    for o in ott:
        targets = orbits_to_targets[o]

        for target in targets:
            if po.plot_projection == 'local':
                target.update_local_coords(composite.event)

            result = target_to_result[target]
            tidx = target_index[target]

            figidx, rowidx = utility.mod_i(tidx, ndmax)

            plot_scene(
                axes[figidx][rowidx, 0],
                target,
                result.processed_obs,
                scattersize,
                colim=colims[tidx],
                outmode=po.plot_projection,
                cmap=cmap)

            syn = plot_scene(
                axes[figidx][rowidx, 1],
                target,
                result.processed_syn,
                scattersize,
                colim=colims[tidx],
                outmode=po.plot_projection,
                cmap=cmap)

            res = plot_scene(
                axes[figidx][rowidx, 2],
                target,
                result.processed_res,
                scattersize,
                colim=dcolims[tidx],
                outmode=po.plot_projection,
                cmap=cmap)

            titley = 0.91
            titlex = 0.16

            axes[figidx][rowidx, 0].annotate(
                o,
                xy=(titlex, titley),
                xycoords='axes fraction',
                xytext=(2., 2.),
                textcoords='offset points',
                weight='bold',
                fontsize=fontsize_title)

            syn_color = scolor('plum1')
            ref_color = scolor('aluminium3')

            draw_sources(
                axes[figidx][rowidx, 1],
                composite.sources, po, color=syn_color)

            if po.reference is not None:
                draw_sources(
                    axes[figidx][rowidx, 1],
                    ref_sources, po, color=ref_color)

            cbb = 0.68 - (0.3175 * rowidx)
            cbl = 0.46
            cbw = 0.15
            cbh = 0.01

            cbaxes = figures[figidx].add_axes([cbl, cbb, cbw, cbh])
            dcbaxes = figures[figidx].add_axes([cbl + 0.3, cbb, cbw, cbh])

            cblabel = 'LOS displacement [m]'
            cbs = plt.colorbar(syn,
                ax=axes[figidx][rowidx, 0],
                ticks=cbtick(colims[tidx]),
                cax=cbaxes,
                orientation='horizontal',
                cmap=cmap)
            cbs.set_label(cblabel, fontsize=fontsize)

            cbr = plt.colorbar(res,
                ax=axes[figidx][rowidx, 2],
                ticks=cbtick(dcolims[tidx]),
                cax=dcbaxes,
                orientation='horizontal',
                cmap=cmap)
            cbr.set_label(cblabel, fontsize=fontsize)

            axis_config(axes[figidx][rowidx, :], po)

            title = str_title(o) + ' Llk_' + po.post_llk
            figures[figidx].suptitle(
                title, fontsize=fontsize_title, weight='bold')

    nplots = ndmax * nfigs
    for delidx in range(nrmax, nplots):
        figidx, rowidx = utility.mod_i(delidx, ndmax)
        for colidx in range(nxmax):
            figures[figidx].delaxes(axes[figidx][rowidx, colidx])

    return figures


def draw_geodetic_fits(problem, plot_options):

    if 'geodetic' not in problem.composites.keys():
        raise Exception('No geodetic composite defined in the problem!')

    po = plot_options

    stage = load_stage(problem, stage_number=po.load_stage, load='full')

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'scenes_%s_%s.%s' % (
            stage.number, po.post_llk, po.outformat))

    if not os.path.exists(outpath) or po.force:
        figs = geodetic_fits(problem, stage, po)
    else:
        logger.info('scene plots exist. Use force=True for replotting!')
        return

    if po.outformat == 'display':
        plt.show()
    else:
        logger.info('saving figures to %s' % outpath)
        with PdfPages(outpath) as opdf:
            for fig in figs:
                opdf.savefig(fig)


def plot_trace(axes, tr, **kwargs):
    return axes.plot(tr.get_xdata(), tr.get_ydata(), **kwargs)


def plot_taper(axes, t, taper, **kwargs):
    y = num.ones(t.size) * 0.9
    taper(y, t[0], t[1] - t[0])
    y2 = num.concatenate((y, -y[::-1]))
    t2 = num.concatenate((t, t[::-1]))
    axes.fill(t2, y2, **kwargs)


def plot_dtrace(axes, tr, space, mi, ma, **kwargs):
    t = tr.get_xdata()
    y = tr.get_ydata()
    y2 = (num.concatenate((y, num.zeros(y.size))) - mi) / \
        (ma - mi) * space - (1.0 + space)
    t2 = num.concatenate((t, t[::-1]))
    axes.fill(
        t2, y2,
        clip_on=False,
        **kwargs)


def seismic_fits(problem, stage, plot_options):
    """
    Modified from grond. Plot synthetic and data waveforms and the misfit for
    the selected posterior model.
    """

    composite = problem.composites['seismic']

    fontsize = 8
    fontsize_title = 10

    target_index = dict(
        (target, i) for (i, target) in enumerate(composite.targets))

    po = plot_options

    point = get_result_point(stage, problem.config, po.post_llk)

    gcms = point['seis_like']
    # gcm_max = d['like']

    results = composite.assemble_results(point)
    source = composite.sources[0]

    logger.info('Plotting waveforms ...')
    target_to_result = {}
    all_syn_trs = []
    dtraces = []
    for target in composite.targets:
        i = target_index[target]
        target_to_result[target] = results[i]

        all_syn_trs.append(results[i].processed_syn)
        dtraces.append(results[i].processed_res)

    skey = lambda tr: tr.channel

    trace_minmaxs = trace.minmax(all_syn_trs, skey)
    dminmaxs = trace.minmax(dtraces, skey)

    for tr in dtraces:
        if tr:
            dmin, dmax = dminmaxs[skey(tr)]
            tr.ydata /= max(abs(dmin), abs(dmax))

    cg_to_targets = utility.gather(
        composite.targets,
        lambda t: t.codes[3],
        filter=lambda t: t in target_to_result)

    cgs = cg_to_targets.keys()

    figs = []

    for cg in cgs:
        targets = cg_to_targets[cg]

        # can keep from here ... until
        nframes = len(targets)

        nx = int(math.ceil(math.sqrt(nframes)))
        ny = (nframes - 1) / nx + 1

        nxmax = 4
        nymax = 4

        nxx = (nx - 1) / nxmax + 1
        nyy = (ny - 1) / nymax + 1

        xs = num.arange(nx) / ((max(2, nx) - 1.0) / 2.)
        ys = num.arange(ny) / ((max(2, ny) - 1.0) / 2.)

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
        for iy in xrange(ny):
            for ix in xrange(nx):
                if (iy, ix) not in frame_to_target:
                    continue

                ixx = ix / nxmax
                iyy = iy / nymax
                if (iyy, ixx) not in figures:
                    figures[iyy, ixx] = plt.figure(
                        figsize=mpl_papersize('a4', 'landscape'))

                    figures[iyy, ixx].subplots_adjust(
                        left=0.03,
                        right=1.0 - 0.03,
                        bottom=0.03,
                        top=1.0 - 0.06,
                        wspace=0.2,
                        hspace=0.2)

                    figs.append(figures[iyy, ixx])

                fig = figures[iyy, ixx]

                target = frame_to_target[iy, ix]

                amin, amax = trace_minmaxs[target.codes[3]]
                absmax = max(abs(amin), abs(amax))

                ny_this = nymax  # min(ny, nymax)
                nx_this = nxmax  # min(nx, nxmax)
                i_this = (iy % ny_this) * nx_this + (ix % nx_this) + 1

                axes2 = fig.add_subplot(ny_this, nx_this, i_this)

                space = 0.5
                space_factor = 1.0 + space
                axes2.set_axis_off()
                axes2.set_ylim(-1.05 * space_factor, 1.05)

                axes = axes2.twinx()
                axes.set_axis_off()

                axes.set_ylim(- absmax * 1.33 * space_factor, absmax * 1.33)

                itarget = target_index[target]
                result = target_to_result[target]

                dtrace = dtraces[itarget]

                tap_color_annot = (0.35, 0.35, 0.25)
                tap_color_edge = (0.85, 0.85, 0.80)
                tap_color_fill = (0.95, 0.95, 0.90)

                plot_taper(
                    axes2, result.processed_obs.get_xdata(), result.taper,
                    fc=tap_color_fill, ec=tap_color_edge)

                obs_color = scolor('aluminium5')
                obs_color_light = light(obs_color, 0.5)

                syn_color = scolor('scarletred2')
                syn_color_light = light(syn_color, 0.5)

                misfit_color = scolor('scarletred2')

                plot_dtrace(
                    axes2, dtrace, space, 0., 1.,
                    fc=light(misfit_color, 0.3),
                    ec=misfit_color)

                plot_trace(
                    axes, result.filtered_syn,
                    color=syn_color_light, lw=1.0)

                plot_trace(
                    axes, result.filtered_obs,
                    color=obs_color_light, lw=0.75)

                plot_trace(
                    axes, result.processed_syn,
                    color=syn_color, lw=1.0)

                plot_trace(
                    axes, result.processed_obs,
                    color=obs_color, lw=0.75)

                xdata = result.filtered_obs.get_xdata()
                axes.set_xlim(xdata[0], xdata[-1])

                tmarks = [
                    result.processed_obs.tmin,
                    result.processed_obs.tmax]

                for tmark in tmarks:
                    axes2.plot(
                        [tmark, tmark], [-0.9, 0.1], color=tap_color_annot)

                for tmark, text, ha, va in [
                        (tmarks[0],
                         '$\,$ ' + str_duration(tmarks[0] - source.time),
                         'right',
                         'bottom'),
                        (tmarks[1],
                         '$\Delta$ ' + str_duration(tmarks[1] - tmarks[0]),
                         'left',
                         'top')]:

                    axes2.annotate(
                        text,
                        xy=(tmark, -0.9),
                        xycoords='data',
                        xytext=(
                            fontsize * 0.4 * [-1, 1][ha == 'left'],
                            fontsize * 0.2),
                        textcoords='offset points',
                        ha=ha,
                        va=va,
                        color=tap_color_annot,
                        fontsize=fontsize)

#                rel_c = num.exp(gcms[itarget] - gcm_max)

#                sw = 0.25
#                sh = 0.1
#                ph = 0.01

#                for (ih, rw, facecolor, edgecolor) in [
#                        (1, rel_c,  light(misfit_color, 0.5), misfit_color)]:

#                    bar = patches.Rectangle(
#                        (1.0 - rw * sw, 1.0 - (ih + 1) * sh + ph),
#                        rw * sw,
#                        sh - 2 * ph,
#                        facecolor=facecolor, edgecolor=edgecolor,
#                        zorder=10,
#                        transform=axes.transAxes, clip_on=False)

#                    axes.add_patch(bar)

                scale_string = None

                infos = []
                if scale_string:
                    infos.append(scale_string)

                infos.append('.'.join(x for x in target.codes if x))
                dist = source.distance_to(target)
                azi = source.azibazi_to(target)[0]
                infos.append(str_dist(dist))
                infos.append(u'%.0f\u00B0' % azi)
                infos.append('%.3f' % gcms[itarget])
                axes2.annotate(
                    '\n'.join(infos),
                    xy=(0., 1.),
                    xycoords='axes fraction',
                    xytext=(2., 2.),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    fontsize=fontsize,
                    fontstyle='normal')

        for (iyy, ixx), fig in figures.iteritems():
            title = '.'.join(x for x in cg if x)
            if len(figures) > 1:
                title += ' (%i/%i, %i/%i)' % (iyy + 1, nyy, ixx + 1, nxx)

            fig.suptitle(title, fontsize=fontsize_title)

    return figs


def draw_seismic_fits(problem, po):

    if 'seismic' not in problem.composites.keys():
        raise Exception('No seismic composite defined for this problem!')

    stage = load_stage(problem, stage_number=po.load_stage, load='full')

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'waveforms_%s_%s.%s' % (
            stage.number, po.post_llk, po.outformat))

    if not os.path.exists(outpath) or po.force:
        figs = seismic_fits(problem, stage, po)
    else:
        logger.info('waveform plots exist. Use force=True for replotting!')
        return

    if po.outformat == 'display':
        plt.show()
    else:
        logger.info('saving figures to %s' % outpath)
        with PdfPages(outpath) as opdf:
            for fig in figs:
                opdf.savefig(fig)


def histplot_op(ax, data, reference=None, alpha=.35, color=None, bins=None,
            ntickmarks=5, tstd=None):
    """
    Modified from pymc3. Additional color argument.
    """
    for i in range(data.shape[1]):
        d = data[:, i]

        mind = num.min(d)
        maxd = num.max(d)

        if reference is not None:
            mind = num.minimum(mind, reference)
            maxd = num.maximum(maxd, reference)

        if tstd is None:
            tstd = num.std(d)

        step = (maxd - mind) / 40.

        if bins is None:
            bins = int(num.ceil((maxd - mind) / step))

        ax.hist(d, bins=bins, normed=True, stacked=True, alpha=alpha,
            align='left', color=color, edgecolor=color)

        leftb = mind - tstd
        rightb = maxd + tstd

        l, r = ax.get_xlim()
        if l != 0.0 and r != 1.0:
            leftb = num.minimum(leftb, l)
            rightb = num.maximum(rightb, r)

        xticklabels = get_tickmarks(leftb, rightb, ntickmarks=ntickmarks)

        ax.set_xlim(leftb, rightb)
#        ax.get_yaxis().set_ticklabels([])
        xax = ax.get_xaxis()
        xax.set_ticklabels(xticklabels)
        xax.set_ticks(xticklabels)


def traceplot(trace, varnames=None, transform=lambda x: x, figsize=None,
              lines=None, chains=None, combined=False, grid=False,
              varbins=None, nbins=40, color=None,
              alpha=0.35, priors=None, prior_alpha=1, prior_style='--',
              axs=None, posterior=None, fig=None):
    """
    Plots posterior pdfs as histograms from multiple mtrace objects.

    Modified from pymc3.

    Parameters
    ----------

    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    posterior : str
        To mark posterior value in distribution 'max', 'min', 'mean', 'all'
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    lines : dict
        Dictionary of variable name / value  to be overplotted as vertical
        lines to the posteriors and horizontal lines on sample values
        e.g. mean of posteriors, true values of a simulation
    chains : int or list of ints
        chain indexes to select from the trace
    combined : bool
        Flag for combining multiple chains into a single chain. If False
        (default), chains will be plotted separately.
    grid : bool
        Flag for adding gridlines to histogram. Defaults to True.
    varbins : list of arrays
        List containing the binning arrays for the variables, if None they will
        be created.
    nbins : int
        Number of bins for each histogram
    color : tuple
        mpl color tuple
    alpha : float
        Alpha value for plot line. Defaults to 0.35.
    axs : axes
        Matplotlib axes. Defaults to None.
    fig : figure
        Matplotlib figure. Defaults to None.

    Returns
    -------

    ax : matplotlib axes
    """

    def make_bins(data, nbins=40):
        d = data.flatten()
        mind = d.min()
        maxd = d.max()
        return num.linspace(mind, maxd, nbins)

    def remove_var(varnames, varname):
        idx = varnames.index(varname)
        varnames.pop(idx)

        varnames = [name for name in trace.varnames if not name.endswith('_')]

    if 'geo_like' in varnames:
        remove_var(varnames, varname='geo_like')

    if 'seis_like' in varnames:
        remove_var(varnames, varname='seis_like')

    if posterior:
        llk = trace.get_values(
            'like', combine=combined, chains=chains, squeeze=False)
        llk = num.squeeze(transform(llk[0]))
        llk = pmp.make_2d(llk)

        posterior_idxs = utility.get_fit_indexes(llk)

        colors = {
            'mean': scolor('orange1'),
            'min': scolor('butter1'),
            'max': scolor('scarletred2')}

    n = len(varnames)
    nrow = int(num.ceil(n / 2.))
    ncol = 2

    n_fig = nrow * ncol

    if figsize is None:
        figsize = (8.2, 11.7)

    if axs is None:
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
        axs = num.atleast_2d(axs)
    elif axs.shape != (nrow, ncol):
        logger.warn('traceplot requires n*2 subplots')
        return None

    if varbins is None:
        make_bins_flag = True
        varbins = []
    else:
        make_bins_flag = False

    for i in range(n_fig):
        coli, rowi = utility.mod_i(i, nrow)

        if i > len(varnames) - 1:
            try:
                fig.delaxes(axs[rowi, coli])
            except KeyError:
                pass
        else:
            v = varnames[i]

            for d in trace.get_values(
                    v, combine=combined, chains=chains, squeeze=False):
                d = num.squeeze(transform(d))
                d = pmp.make_2d(d)

                if make_bins_flag:
                    varbin = make_bins(d, nbins=nbins)
                    varbins.append(varbin)
                else:
                    varbin = varbins[i]

                if lines:
                    if v in config.hyper_pars.values():
                        reference = None
                        if v in lines.keys():
                            lines.pop(v)
                    elif v in lines:
                        reference = lines[v]
                    else:
                        reference = None
                else:
                    reference = None

                if color is None:
                    color = scolor('aluminium3')

                histplot_op(
                    axs[rowi, coli], d, reference=reference,
                    bins=varbin, alpha=alpha, color=color)

                axs[rowi, coli].set_title(str(v) + ' ' + plot_units[v])
                axs[rowi, coli].grid(grid)
                axs[rowi, coli].set_yticks([])
                axs[rowi, coli].set_yticklabels([])
#                axs[rowi, coli].set_ylabel("Frequency")

                if lines:
                    try:
                        axs[rowi, coli].axvline(x=lines[v], color="k", lw=0.5)
                    except KeyError:
                        pass

                if posterior:
                    if posterior == 'all':
                        for k, idx in posterior_idxs.iteritems():
                            axs[rowi, coli].axvline(
                                x=d[idx], color=colors[k], lw=0.5)
                    else:
                        idx = posterior_idxs[posterior]
                        axs[rowi, coli].axvline(
                            x=d[idx], color=colors[posterior], lw=0.5)

    fig.tight_layout()
    return fig, axs, varbins


def select_transform(sc, n_steps):
    """
    Select transform function to be applied after loading the sampling results.

    Parameters
    ----------
    sc : :class:`config.SamplerConfig`
        Name of the sampler that has been used in sampling the posterior pdf
    n_steps : int
        Number of chains to select last samples of each trace.

    Returns
    -------
    func : instance
    """

    pa = sc.parameters

    def last_sample(x):
        return x[(n_steps - 1)::n_steps].flatten()

    def burn_sample(x):
        if n_steps == 1:
            return x
        else:
            nchains = x.shape[0] / n_steps
            xout = []
            for i in range(nchains):
                nstart = int((n_steps * i) + (n_steps * pa.burn))
                nend = int(n_steps * (i + 1) - 1)
                xout.append(x[nstart:nend:pa.thin])

            return num.vstack(xout).flatten()

    if sc.name == 'SMC':
        return last_sample
    elif sc.name == 'Metropolis':
        return burn_sample


def select_metropolis_chains(problem, mtrace, post_llk):
    """
    Select chains from Multitrace
    """
    draws = len(mtrace)

    llks = num.array([mtrace.point(
        draws - 1, chain)[
            problem._like_name] for chain in mtrace.chains])

    chain_idxs = utility.get_fit_indexes(llks)
    return chain_idxs[post_llk]


def draw_posteriors(problem, plot_options):
    """
    Identify which stage is the last complete stage and plot posteriors.
    """

    hypers = utility.check_hyper_flag(problem)
    po = plot_options

    stage = load_stage(problem, stage_number=po.load_stage, load='trace')
    pc = problem.config.problem_config

    if po.load_stage is not None:
        list_indexes = [po.load_stage]
    else:
        if stage.number == 'final':
            stage_number = backend.get_highest_sampled_stage(
                problem.outfolder, return_final=False)
            list_indexes = [
                str(i) for i in range(stage_number + 1)] + ['final']
        else:
            list_indexes = [
                str(i) for i in range(int(stage.number) + 1)]

    if hypers:
        sc = problem.config.hyper_sampler_config
        varnames = pc.hyperparameters.keys() + ['like']
    else:
        sc = problem.config.sampler_config
        varnames = pc.select_variables() + pc.hyperparameters.keys() + ['like']

    figs = []

    for s in list_indexes:
        if s == '0':
            draws = 1
        elif s == 'final' and not hypers and sc.name == 'Metropolis':
            draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
        else:
            draws = sc.parameters.n_steps

        transform = select_transform(sc=sc, n_steps=draws)

        stage_path = os.path.join(
            problem.outfolder, 'stage_%s' % s)

        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            'stage_%s_%s.%s' % (s, po.post_llk, po.outformat))

        if not os.path.exists(outpath) or po.force:
            logger.info('plotting stage: %s' % stage_path)
            mtrace = backend.load(stage_path, model=problem.model)

            if sc.name == 'Metropolis' and po.post_llk != 'all':
                chains = select_metropolis_chains(problem, mtrace, po.post_llk)
                logger.info('plotting result: %s of Metropolis chain %i' % (
                    po.post_llk, chains))
            else:
                chains = None

            fig, _, _ = traceplot(
                mtrace,
                varnames=varnames,
                transform=transform,
                chains=chains,
                combined=True,
                lines=po.reference,
                posterior='max')

            if not po.outformat == 'display':
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)
            else:
                figs.append(fig)

        else:
            logger.info('plot for stage %s exists. Use force=True for'
                ' replotting!' % s)

    if format == 'display':
        plt.show()


def draw_correlation_hist(problem, plot_options):
    """
    Draw parameter correlation plot and histograms from the final atmip stage.
    Only feasible for 'geometry' problem.
    """

    po = plot_options
    mode = problem.config.problem_config.mode

    assert mode == 'geometry'
    assert po.load_stage != 0

    hypers = utility.check_hyper_flag(problem)

    if hypers:
        sc = problem.config.hyper_sampler_config
        varnames = problem.config.problem_config.hyperparameters.keys()
    else:
        sc = problem.config.sampler_config
        varnames = problem.config.problem_config.select_variables()

    if len(varnames) < 2:
        raise Exception('Need at least two parameters to compare!'
                        'Found only %i variables! ' % len(varnames))

    if po.load_stage is None and not hypers and not sc.name == 'SMC':
        draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
    else:
        draws = sc.parameters.n_steps

    transform = select_transform(sc=sc, n_steps=draws)

    stage = load_stage(problem, po.load_stage, load='trace')

    if sc.name == 'Metropolis' and po.post_llk != 'all':
        chains = select_metropolis_chains(problem, stage.mtrace, po.post_llk)
        logger.info('plotting result: %s of Metropolis chain %i' % (
            po.post_llk, chains))
    else:
        chains = None

    outpath = os.path.join(
        problem.outfolder, po.figure_dir, 'corr_hist_%s_%s.%s' % (
        stage.number, po.post_llk, po.outformat))

    if not os.path.exists(outpath) or po.force:
        fig, axs = correlation_plot_hist(
            mtrace=stage.mtrace,
            varnames=varnames,
            transform=transform,
            cmap=plt.cm.gist_earth_r,
            chains=chains,
            point=po.reference,
            point_size='8',
            point_color='red')
    else:
        logger.info('correlation plot exists. Use force=True for replotting!')
        return

    if po.outformat == 'display':
        plt.show()
    else:
        logger.info('saving figure to %s' % outpath)
        fig.savefig(outpath, format=po.outformat, dpi=po.dpi)


def n_model_plot(models, axes=None):
    """
    Plot cake layered earth models.
    """
    if axes is None:
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize('a5', 'portrait'))

    cp.labelspace(axes)
    cp.labels_model(axes=axes)
    cp.sketch_model(models[0], axes=axes)

    ref = models.pop(0)

    for mod in models:
        z = mod.profile('z')
        vp = mod.profile('vp')
        vs = mod.profile('vs')
        axes.plot(vp, z, color=scolor('scarletred1'), lw=0.5)
        axes.plot(vs, z, color=scolor('skyblue1'), lw=0.5)

    z = ref.profile('z')
    vp = ref.profile('vp')
    vs = ref.profile('vs')
    axes.plot(vp, z, color=scolor('aluminium6'), lw=1.5)
    axes.plot(vs, z, color=scolor('aluminium3'), lw=1.5)

    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    xmin = 0.
    my = (ymax - ymin) * 0.05
    mx = (xmax - xmin) * 0.2
    axes.set_ylim(ymax + my, ymin - my)
    axes.set_xlim(xmin, xmax + mx)
    return fig, axes


def load_earthmodels(engine, targets, depth_max='cmb'):
    earthmodels = []
    for t in targets:
        store = engine.get_store(t.store_id)
        em = store.config.earthmodel_1d.extract(depth_max=depth_max)
        earthmodels.append(em)

    return earthmodels


def draw_earthmodels(problem, plot_options):

    po = plot_options

    for dataset, composite in problem.composites.iteritems():

        if dataset == 'seismic':
            models_dict = {}
            sc = problem.config.seismic_config

            if sc.gf_config.reference_location is None:
                plot_stations = composite.stations
            else:
                plot_stations = [sc.gf_config.reference_location]

            for station in plot_stations:
                outpath = os.path.join(
                    problem.outfolder, po.figure_dir,
                    '%s_%s_velocity_models.%s' % (
                        dataset, station.station, po.outformat))

                if not os.path.exists(outpath) or po.force:
                    targets = init_targets(
                        [station],
                        earth_model=sc.gf_config.earth_model,
                        channels=[sc.channels[0]],
                        sample_rate=sc.gf_config.sample_rate,
                        crust_inds=range(sc.gf_config.n_variations + 1),
                        interpolation='multilinear')

                    models = load_earthmodels(
                        composite.engine, targets,
                        depth_max=sc.gf_config.depth_limit_variation * km)
                    models_dict[outpath] = models

                else:
                    logger.info(
                        '%s earthmodel plot for station %s exists. Use '
                        'force=True for replotting!' % (
                            dataset, station.station))

        elif dataset == 'geodetic':
            gc = problem.config.geodetic_config

            models_dict = {}
            models = []
            outpath = os.path.join(
                problem.outfolder, po.figure_dir,
                '%s_%s_velocity_models.%s' % (
                    dataset, 'psgrn', po.outformat))

            if not os.path.exists(outpath) or po.force:
                for crust_ind in range(gc.gf_config.n_variations + 1):
                    psgrn_input_path = os.path.join(
                        gc.gf_config.store_superdir,
                        'psgrn_green_' + str(crust_ind),
                        'input')
                    models.append(
                        utility.PsGrnArray2LayeredModel(psgrn_input_path))

                models_dict[outpath] = models

            else:
                logger.info(
                    '%s earthmodel plot exists. Use force=True for'
                    ' replotting!' % dataset)
                return

        else:
            raise Exception(
                'Plot for dataset %s not (yet) supported' % dataset)

        figs = []
        axes = []
        for models in models_dict.itervalues():
            fig, axs = n_model_plot(models, axes=None)
            figs.append(fig)
            axes.append(axs)

        if po.outformat == 'display':
            plt.show()
        else:
            for fig, outpath in zip(figs, models_dict.keys()):
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)


def fault_slip_distribution(patches, ):
    """
    Draw discretized fault geometry rotated to the 2-d view of the foot-wall
    of the fault.

    Parameters
    ----------
    patches : list
        of RectangularSources
    """

    fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize('a5', 'landscape'))

    for patch in patches:
        rot_coords = rotate(matrix, angle, rotationaxes=())


    return figs, axs


def draw_static_dist(problem, plot_options):


    if 'geodetic' not in problem.composites.keys():
        raise Exception('No geodetic composite defined for this problem!')

    gc = problem.composites['geodetic']

    dsources = gc.load_fault_geometry()
    patches = dsources[problem.config.problem_config.priors.keys()[0]]

    figs, axs = fault_slip_distribution(patches, )

    outpaths = []
    for i in range(len(figs)):
        outpaths.append(os.path.join(
                problem.outfolder, po.figure_dir,
                'static_slip_dist_%i.%s' % (i, po.outformat)))

    if po.outformat == 'display':
        plt.show()
    else:
        for fig, outpath in zip(figs, problem.outpath keys()):
            logger.info('saving figure to %s' % outpath)
            fig.savefig(outpath, format=po.outformat, dpi=po.dpi)


plots_catalog = {
    'correlation_hist': draw_correlation_hist,
    'stage_posteriors': draw_posteriors,
    'waveform_fits': draw_seismic_fits,
    'scene_fits': draw_geodetic_fits,
    'velocity_models': draw_earthmodels,
    'static_slip_dist' : draw_static_dist,
                }


def available_plots():
    return list(plots_catalog.keys())
