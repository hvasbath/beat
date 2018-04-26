from pyrocko import cake_plot as cp
from pyrocko import gf
from pyrocko import orthodrome as otd

from pymc3 import plots as pmp

import math
import os
import logging
import copy

from beat import utility
from beat.models import Stage
from beat.metropolis import get_trace_stats
from beat.heart import init_seismic_targets, init_geodetic_targets
from beat.colormap import slip_colormap

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick

from scipy.stats import kde
import numpy as num
from pyrocko.guts import Object, String, Dict, List, Bool, Int, load
from pyrocko import util, trace
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light

import pyrocko.moment_tensor as mt
from pyrocko.plot import mpl_papersize, mpl_init

logger = logging.getLogger('plotting')

km = 1000.


__all__ = [
    'PlotOptions', 'correlation_plot', 'correlation_plot_hist',
    'get_result_point', 'seismic_fits', 'geodetic_fits', 'traceplot',
    'select_transform']

u_nm = '$[Nm]$'
u_km = '$[km]$'
u_deg = '$[^{\circ}]$'
u_m = '$[m]$'
u_v = '$[m^3]$'
u_s = '$[s]$'
u_hyp = ''

plot_units = {
    'east_shift': u_km,
    'north_shift': u_km,
    'depth': u_km,
    'width': u_km,
    'length': u_km,

    'dip': u_deg,
    'dip1': u_deg,
    'dip2': u_deg,
    'strike': u_deg,
    'strike1': u_deg,
    'strike2': u_deg,
    'rake': u_deg,
    'rake1': u_deg,
    'rake2': u_deg,
    'mix': u_hyp,

    'volume_change': u_v,
    'diameter': u_km,
    'slip': u_m,
    'azimuth': u_deg,
    'bl_azimuth': u_deg,
    'amplitude': u_nm,
    'bl_amplitude': u_m,
    'locking_depth': u_km,

    'nucleation_dip': u_km,
    'nucleation_strike': u_km,
    'time_shift': u_s,

    'mnn': u_nm,
    'mee': u_nm,
    'mdd': u_nm,
    'mne': u_nm,
    'mnd': u_nm,
    'med': u_nm,
    'magnitude': u_hyp,

    'distance': u_km,
    'delta_depth': u_km,
    'delta_time': u_s,
    'time': u_s,
    'duration': u_s,
    'peak_ratio': u_hyp,
    'h_': u_hyp,
    'like': u_hyp}


def hypername(varname):
    if varname[0:2] == 'h_':
        return 'h_'
    return varname


class PlotOptions(Object):
    post_llk = String.T(
        default='max',
        help='Which model to plot on the specified plot; Default: "max";'
             ' Options: "max", "min", "mean", "all"')
    plot_projection = String.T(
        default='local',
        help='Projection to use for plotting geodetic data; options: "latlon"')
    utm_zone = Int.T(
        default=36,
        optional=True,
        help='Only relevant if plot_projection is "utm"')
    load_stage = Int.T(
        default=-1,
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
    varnames = List.T(
        default=[], optional=True, help='Names of variables to plot')


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


def kde2plot_op(ax, x, y, grid=200, **kwargs):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    extent = kwargs.pop('extent', [])
    if len(extent) != 4:
        extent = [xmin, xmax, ymin, ymax]

    grid = grid * 1j
    X, Y = num.mgrid[xmin:xmax:grid, ymin:ymax:grid]
    positions = num.vstack([X.ravel(), Y.ravel()])
    values = num.vstack([x.ravel(), y.ravel()])
    kernel = kde.gaussian_kde(values)
    Z = num.reshape(kernel(positions).T, X.shape)

    ax.imshow(num.rot90(Z), extent=extent, **kwargs)


def kde2plot(x, y, grid=200, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid, **kwargs)
    return ax


def correlation_plot(
        mtrace, varnames=None,
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

            kde2plot(
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
        if nvar < 5:
            figsize = mpl_papersize('a5', 'landscape')
        else:
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
                    if v_namea in point.keys():
                        reference = point[v_namea]
                        axs[l, k].axvline(
                            x=reference, color=point_color,
                            lw=int(point_size) / 4.)
                    else:
                        reference = None
                else:
                    reference = None

                histplot_op(
                    axs[l, k], pmp.utils.make_2d(a), alpha=alpha,
                    color='orange', tstd=0., reference=reference,
                    ntickmarks=ntickmarks)
                axs[l, k].get_yaxis().set_visible(False)

                xticks = axs[l, k].get_xticks()
                xlim = axs[l, k].get_xlim()
            else:
                b = d[v_nameb]

                kde2plot(
                    a, b, grid=grid, ax=axs[l, k], cmap=cmap, aspect='auto')

                bmin = b.min()
                bmax = b.max()

                if point is not None:
                    if v_namea and v_nameb in point.keys():
                        axs[l, k].plot(point[v_namea], point[v_nameb],
                            color=point_color, marker=point_style,
                            markersize=point_size)

                        bmin = num.minimum(bmin, point[v_nameb])
                        bmax = num.maximum(bmax, point[v_nameb])

                yticker = tick.MaxNLocator(nbins=ntickmarks)
                axs[l, k].set_xticks(xticks)
                axs[l, k].set_xlim(xlim)
                yax = axs[l, k].get_yaxis()
                yax.set_major_locator(yticker)

            if l != nvar - 1:
                axs[l, k].get_xaxis().set_ticklabels([])

            if k == 0:
                axs[l, k].set_ylabel(
                    v_nameb + '\n ' + plot_units[hypername(v_nameb)])
            else:
                axs[l, k].get_yaxis().set_ticklabels([])

            axs[l, k].tick_params(direction='in')

        axs[l, k].set_xlabel(v_namea + '\n ' + plot_units[hypername(v_namea)])

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
    plt.title('Displacements [m] %s' % uwifg.name)
    plt.show()


def plot_cov(target, point_size=20):

    ax = plt.axes()
    im = ax.scatter(target.lons, target.lats, point_size,
             num.array(target.covariance.pred_v.sum(axis=0)).flatten(),
             edgecolors='none')
    plt.colorbar(im)
    plt.title('Prediction Covariance [m2] %s' % target.name)
    plt.show()


def plot_matrix(A):
    """
    Very simple plot of a matrix for fast inspections.
    """
    ax = plt.axes()
    im = ax.matshow(A)
    plt.colorbar(im)
    plt.show()


def plot_log_cov(cov_mat):
    ax = plt.axes()
    mask = num.ones_like(cov_mat)
    mask[cov_mat < 0] = -1.
    im = ax.imshow(num.multiply(num.log(num.abs(cov_mat)), mask))
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
        llk = stage.mtrace.get_values(
            varname=stage.step['likelihood_name'],
            burn=stage.step['n_steps'] - 1,
            combine=True)

        posterior_idxs = utility.get_fit_indexes(llk)

        n_steps = config.sampler_config.parameters.n_steps - 1
        point = stage.mtrace.point(
            idx=n_steps, chain=posterior_idxs[point_llk])

    return point


def plot_quadtree(ax, data, target, cmap, colim, alpha=0.8):
    """
    Plot UnwrappedIFG displacements on the respective quadtree rectangle.
    """
    rectangles = []
    for E, N, sE, sN in target.quadtree.iter_leaves():
        rectangles.append(
            Rectangle(
                (E / km, N / km),
                width=sE / km,
                height=sN / km,
                edgecolor='black'))

    patch_col = PatchCollection(
        rectangles, match_original=True, alpha=alpha, linewidth=0.5)
    patch_col.set(array=data, cmap=cmap)
    patch_col.set_clim((-colim, colim))

    E = target.quadtree.east_shifts
    N = target.quadtree.north_shifts
    xmin = E.min() / km
    xmax = (E + target.quadtree.sizeE).max() / km
    ymin = N.min() / km
    ymax = (N + target.quadtree.sizeN).max() / km

    ax.add_collection(patch_col)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    return patch_col


def plot_scene(ax, target, data, scattersize, colim,
               outmode='latlon', **kwargs):
    if outmode == 'latlon':
        x = target.lons
        y = target.lats
    elif outmode == 'utm':
        x = target.utme / km
        y = target.utmn / km
    elif outmode == 'local':
        if target.quadtree is not None:
            cmap = kwargs.pop('cmap', plt.cm.jet)
            return plot_quadtree(ax, data, target, cmap, colim)
        else:
            x = target.east_shifts / km
            y = target.north_shifts / km

    return ax.scatter(
        x, y, scattersize, data,
        edgecolors='none', vmin=-colim, vmax=colim, **kwargs)


def geodetic_fits(problem, stage, plot_options):
    """
    Plot geodetic data, synthetics and residuals.
    """
    from pyrocko.dataset import gshhg

    mode = problem.config.problem_config.mode

    scattersize = 16
    fontsize = 10
    fontsize_title = 12
    ndmax = 3
    nxmax = 3
    cmap = plt.cm.jet

    po = plot_options

    composite = problem.composites['geodetic']

    try:
        sources = composite.sources
    except AttributeError:
        logger.info('FFI waveform fit, using reference source ...')
        sources = composite.config.gf_config.reference_sources

    if po.reference is not None:
        if mode != 'ffi':
            composite.point2sources(po.reference)
            ref_sources = copy.deepcopy(composite.sources)
        point = po.reference
    else:
        point = get_result_point(stage, problem.config, po.post_llk)

    dataset_index = dict(
        (data, i) for (i, data) in enumerate(composite.datasets))

    results = composite.assemble_results(point)
    nrmax = len(results)

    dataset_to_result = {}
    for dataset, result in zip(composite.datasets, results):
        dataset_to_result[dataset] = result

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

    def draw_coastlines(ax, event, po):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        logger.debug('Drawing coastlines ...')
        coasts = gshhg.GSHHG.full()

        if po.plot_projection == 'latlon':
            west, east = xlim
            south, north = ylim

        elif po.plot_projection == 'utm':
            lons, lats = utility.utm_to_lonlat(
                utmx=xlim, utmy=ylim, zone=po.utm_zone)
            west, east = lons
            south, north = lats

        elif po.plot_projection == 'local':
            lats, lons = otd.ne_to_latlon(
                event.lat, event.lon,
                north_m=num.array(ylim) * km, east_m=num.array(xlim) * km)
            south, north = lats
            west, east = lons

        polygons = coasts.get_polygons_within(
            west=west, east=east, south=south, north=north)

        for p in polygons:
            if (p.is_land() or p.is_antarctic_grounding_line() or
               p.is_island_in_lake()):

                if po.plot_projection == 'latlon':
                    xs = p.lons
                    ys = p.lats

                elif po.plot_projection == 'utm':
                    xs, ys = utility.lonlat_to_utm(p.lons, p.lats, po.utm_zone)

                elif po.plot_projection == 'local':
                    ys, xs = otd.latlon_to_ne_numpy(
                        event.lat, event.lon, p.lats, p.lons)
                    ys /= km
                    xs /= km

                ax.plot(xs, ys, '-k', linewidth=0.5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

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
                outline = source.outline(cs='xy') / km
                outline = utility.swap_columns(outline, 0, 1)

            if outline.shape[0] > 1:
                ax.plot(outline[:, 0], outline[:, 1], '-',
                    linewidth=1.0, **kwargs)
                ax.plot(
                    outline[0:2, 0], outline[0:2, 1], '-k', linewidth=1.0)
            else:
                ax.plot(outline[:, 0], outline[:, 1], marker='*',
                    markersize=10, **kwargs)

    def cbtick(x):
        rx = math.floor(x * 1000.) / 1000.
        return [-rx, rx]

    def str_title(name):
        if name[0] == 'A':
            title = 'Orbit: ascending'
        elif name[0] == 'D':
            title = 'Orbit: descending'
        else:
            title = 'GPS'

        return title

    orbits_to_datasets = utility.gather(
        composite.datasets,
        lambda t: t.name,
        filter=lambda t: t in dataset_to_result)

    ott = orbits_to_datasets.keys()

    colims = [num.max([
        num.max(num.abs(r.processed_obs)),
        num.max(num.abs(r.processed_syn))]) for r in results]
    dcolims = [num.max(num.abs(r.processed_res)) for r in results]

    for o in ott:
        datasets = orbits_to_datasets[o]

        for dataset in datasets:
            if po.plot_projection == 'local':
                dataset.update_local_coords(composite.event)

            result = dataset_to_result[dataset]
            tidx = dataset_index[dataset]

            figidx, rowidx = utility.mod_i(tidx, ndmax)

            plot_scene(
                axes[figidx][rowidx, 0],
                dataset,
                result.processed_obs,
                scattersize,
                colim=colims[tidx],
                outmode=po.plot_projection,
                cmap=cmap)

            draw_coastlines(axes[figidx][rowidx, 0], composite.event, po)

            syn = plot_scene(
                axes[figidx][rowidx, 1],
                dataset,
                result.processed_syn,
                scattersize,
                colim=colims[tidx],
                outmode=po.plot_projection,
                cmap=cmap)

            draw_coastlines(axes[figidx][rowidx, 1], composite.event, po)

            res = plot_scene(
                axes[figidx][rowidx, 2],
                dataset,
                result.processed_res,
                scattersize,
                colim=dcolims[tidx],
                outmode=po.plot_projection,
                cmap=cmap)

            draw_coastlines(axes[figidx][rowidx, 2], composite.event, po)

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
            ref_color = scolor('aluminium5')

            draw_sources(
                axes[figidx][rowidx, 1],
                sources, po, color=syn_color)

            if po.reference is not None and mode != 'ffi':
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

    stage = Stage(homepath=problem.outfolder)

    if po.reference is None:
        stage.load_results(
            model=problem.model, stage_number=po.load_stage, load='full')
        llk_str = po.post_llk
    else:
        llk_str = 'ref'

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'scenes_%s_%s.%s' % (
            stage.number, llk_str, po.outformat))

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

    if po.reference is None:
        point = get_result_point(stage, problem.config, po.post_llk)
    else:
        point = po.reference

    # gcms = point['seis_like']
    # gcm_max = d['like']

    results = composite.assemble_results(point)
    try:
        source = composite.sources[0]
    except AttributeError:
        logger.info('FFI waveform fit, using reference source ...')
        source = composite.config.gf_config.reference_sources[0]
        source.time += problem.config.event.time

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
                #infos.append('%.3f' % gcms[itarget])
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

    stage = Stage(homepath=problem.outfolder)

    mode = problem.config.problem_config.mode

    if po.reference is None:
        llk_str = po.post_llk
        stage.load_results(
            model=problem.model, stage_number=po.load_stage, load='full')
    else:
        llk_str = 'ref'

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'waveforms_%s_%s.%s' % (
            stage.number, llk_str, po.outformat))

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
        mind = d.min()
        maxd = d.max()
        # bins, mind, maxd = pmp.artists.fast_kde(data[:,i])

        if reference is not None:
            mind = num.minimum(mind, reference)
            maxd = num.maximum(maxd, reference)

        if tstd is None:
            tstd = num.std(d)

        step = (maxd - mind) / 40.

        if bins is None:
            bins = int(num.ceil((maxd - mind) / step))

        ax.hist(d, bins=bins, normed=True, stacked=True, alpha=alpha,
            align='left', histtype='stepfilled', color=color, edgecolor=color)

        l, r = ax.get_xlim()
        leftb = mind - tstd
        rightb = maxd + tstd

        if l != 0.0 or r != 1.0:
            leftb = num.minimum(leftb, l)
            rightb = num.maximum(rightb, r)

        ax.set_xlim(leftb, rightb)
        xax = ax.get_xaxis()
        xticker = tick.MaxNLocator(nbins=ntickmarks)
        xax.set_major_locator(xticker)


def traceplot(trace, varnames=None, transform=lambda x: x, figsize=None,
              lines=None, chains=None, combined=False, grid=False,
              varbins=None, nbins=40, color=None,
              alpha=0.35, priors=None, prior_alpha=1, prior_style='--',
              axs=None, posterior=None, fig=None, plot_style='kde'):
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
        llk = pmp.utils.make_2d(llk)

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
        if n < 5:
            figsize = (5.8, 4.1)
        elif n < 7:
            figsize = (5.8, 8.2)
        else:
            figsize = (8.2, 11.7)

    if axs is None:
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
        axs = num.atleast_2d(axs)
    elif axs.shape != (nrow, ncol):
        raise TypeError('traceplot requires n*2 subplots %i, %i' % (
                        nrow, ncol))

    if varbins is None:
        make_bins_flag = True
        varbins = []
    else:
        make_bins_flag = False

    input_color = copy.deepcopy(color)
    for i in range(n_fig):
        coli, rowi = utility.mod_i(i, nrow)

        if i > len(varnames) - 1:
            try:
                fig.delaxes(axs[rowi, coli])
            except KeyError:
                pass
        else:
            v = varnames[i]
            color = copy.deepcopy(input_color)
            for d in trace.get_values(
                    v, combine=combined, chains=chains, squeeze=False):
                d = transform(d)
                # iterate over columns in case varsize > 1

                for isource, e in enumerate(d.T):
                    e = pmp.utils.make_2d(e)

                    if make_bins_flag:
                        varbin = make_bins(e, nbins=nbins)
                        varbins.append(varbin)
                    else:
                        varbin = varbins[i]

                    if lines:
                        if hypername(v) == 'h_':
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
                        color = scolor('aluminium4')
                    elif isource == 2:
                        color = light(color, 0.3)

                    if plot_style == 'kde':
                        pmp.kdeplot(
                            e, shade=alpha, ax=axs[rowi, coli],
                            color=color, linewidth=1.,
                            kwargs_shade={'color': color})
                        axs[rowi, coli].relim()
                        axs[rowi, coli].autoscale(tight=False)
                        axs[rowi, coli].set_ylim(0)
                        xax = axs[rowi, coli].get_xaxis()
                        #axs[rowi, coli].set_ylim([0, e.max()])
                        xticker = tick.MaxNLocator(nbins=5)
                        xax.set_major_locator(xticker)
                    else:
                        histplot_op(
                            axs[rowi, coli], e, reference=reference,
                            bins=varbin, alpha=alpha, color=color)

                    axs[rowi, coli].set_title(
                        str(v) + ' ' + plot_units[hypername(v)])
                    axs[rowi, coli].grid(grid)
                    axs[rowi, coli].set_yticks([])
                    axs[rowi, coli].set_yticklabels([])
    #                axs[rowi, coli].set_ylabel("Frequency")

                    if lines:
                        try:
                            axs[rowi, coli].axvline(x=lines[v], color="k", lw=1.)
                        except KeyError:
                            pass

                    if posterior:
                        if posterior == 'all':
                            for k, idx in posterior_idxs.iteritems():
                                axs[rowi, coli].axvline(
                                    x=e[idx], color=colors[k], lw=1.)
                        else:
                            idx = posterior_idxs[posterior]
                            axs[rowi, coli].axvline(
                                x=e[idx], color=colors[posterior], lw=1.)

    fig.tight_layout()
    return fig, axs, varbins


def select_transform(sc, n_steps=None):
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
        return x[(n_steps - 1)::n_steps]

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

            return num.vstack(xout)

    def standard(x):
        return x

    if n_steps is None:
        return standard

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

    stage = Stage(homepath=problem.outfolder)

    pc = problem.config.problem_config

    if po.load_stage is not None:
        list_indexes = [po.load_stage]
    else:
        stage_number = stage.handler.highest_sampled_stage()
        if os.path.exists(stage.handler.atmip_path(-1)):
            list_indexes = [i for i in range(-1, stage_number + 1)]
        else:
            list_indexes = [i for i in range(stage_number)]

    if hypers:
        sc = problem.config.hyper_sampler_config
        varnames = pc.hyperparameters.keys() + ['like']
    else:
        sc = problem.config.sampler_config
        varnames = problem.rvs.keys() + pc.hyperparameters.keys() + ['like']

    if len(po.varnames) > 0:
        varnames = po.varnames

    logger.info('Plotting variables: %s' % (', '.join((v for v in varnames))))
    figs = []

    for s in list_indexes:
        if s == 0:
            draws = 1
        elif s == -1 and not hypers and sc.name == 'Metropolis':
            draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
        elif s == -2:    # return summarized trace plot -standard pymc trace
            draws = None
        else:
            draws = sc.parameters.n_steps

        transform = select_transform(sc=sc, n_steps=draws)

        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            'stage_%i_%s.%s' % (s, po.post_llk, po.outformat))

        if not os.path.exists(outpath) or po.force:
            logger.info('plotting stage: %s' % stage.handler.stage_path(s))
            stage.load_results(
                model=problem.model, stage_number=s, load='trace')

            if sc.name == 'Metropolis' and po.post_llk != 'all':
                chains = select_metropolis_chains(
                    problem, stage.mtrace, po.post_llk)
                logger.info('plotting result: %s of Metropolis chain %i' % (
                    po.post_llk, chains))
            else:
                chains = None

            fig, _, _ = traceplot(
                stage.mtrace,
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

    if problem.config.problem_config.n_sources > 1:
        raise NotImplementedError(
            'correlation_hist plot not working (yet) for n_sources > 1')

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
        varnames = problem.config.problem_config.select_variables() + ['like']

    if len(po.varnames) > 0:
        varnames = po.varnames

    logger.info('Plotting variables: %s' % (', '.join((v for v in varnames))))

    if len(varnames) < 2:
        raise Exception('Need at least two parameters to compare!'
                        'Found only %i variables! ' % len(varnames))

    if po.load_stage is None and not hypers and not sc.name == 'SMC':
        draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
    if po.load_stage == -2:
        draws = None
    else:
        draws = sc.parameters.n_steps

    transform = select_transform(sc=sc, n_steps=draws)

    stage = Stage(homepath=problem.outfolder)
    stage.load_results(
        model=problem.model, stage_number=po.load_stage, load='trace')

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


def n_model_plot(models, axes=None, draw_bg=True, highlightidx=[]):
    """
    Plot cake layered earth models.
    """
    if axes is None:
        mpl_init(fontsize=12)
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize('a5', 'portrait'))

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
    axes.set_ylim(ymax + my, ymin - my)
    axes.set_xlim(xmin, xmax + mx)
    return fig, axes


def load_earthmodels(store_superdir, targets, depth_max='cmb'):

    ems = []
    emr = []
    for t in targets:
        path = os.path.join(store_superdir, t.store_id, 'config')
        config = load(filename=path)
        em = config.earthmodel_1d.extract(depth_max=depth_max)
        ems.append(em)

        if config.earthmodel_receiver_1d is not None:
            emr.append(config.earthmodel_receiver_1d)

    return [ems, emr]


def draw_earthmodels(problem, plot_options):

    po = plot_options

    for datatype, composite in problem.composites.iteritems():

        if datatype == 'seismic':
            models_dict = {}
            sc = problem.config.seismic_config

            if sc.gf_config.reference_location is None:
                plot_stations = composite.get_unique_stations()
            else:
                plot_stations = [composite.get_unique_stations()[0]]
                plot_stations[0].station = sc.gf_config.reference_location.station

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
                        crust_inds=range(*sc.gf_config.n_variations),
                        interpolation='multilinear')

                    models = load_earthmodels(
                        composite.engine.store_superdirs[0], targets,
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
                    crust_inds=range(*gc.gf_config.n_variations),
                    sample_rate=gc.gf_config.sample_rate)

                models = load_earthmodels(
                    store_superdir=composite.engine.store_superdirs[0],
                    targets=targets,
                    depth_max=gc.gf_config.source_depth_max * km)
                models_dict[outpath] = models[0]  #select only source site

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
        for path, models in models_dict.iteritems():
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


def fault_slip_distribution(
        fault, mtrace=None, transform=lambda x: x, alpha=0.9, ntickmarks=5,
        ncontours=100, reference=None):
    """
    Draw discretized fault geometry rotated to the 2-d view of the foot-wall
    of the fault.

    Parameters
    ----------
    fault : 

    """

    def draw_quivers(
            ax, uperp, uparr, xgr, ygr, rake, color='black',
            draw_legend=False):

        angles = num.arctan2(uperp, uparr) * \
            (180. / num.pi) + rake

        slips = num.sqrt((uperp ** 2 + uparr ** 2)).ravel()
        slipsx = num.cos(angles * num.pi / 180.) * slips
        slipsy = num.sin(angles * num.pi / 180.) * slips

        # slip arrows of slip on patches
        quivers = ax.quiver(
            xgr.ravel(), ygr.ravel(), slipsx, slipsy,
            units='dots', angles='xy', scale_units='xy', scale=1,
            width=1., color=color)

        if draw_legend:
            quiver_legend_length = num.floor(num.max(slips) * 10.) / 10.

            plt.quiverkey(
                quivers, width / 4., height / 4., quiver_legend_length,
                '%i [m]' % quiver_legend_length)

        return slipsx, slipsy

    reference_slip = num.sqrt(
        reference['uperp'] ** 2 + reference['uparr'] ** 2)

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=mpl_papersize('a4', 'landscape'))

    height = fault.ordering.patch_size_dip
    width = fault.ordering.patch_size_strike

    figs = []
    axs = []
    for i in range(fault.nsubfaults):
        np_h, np_w = fault.get_subfault_discretization(i)
        ext_source = fault.get_subfault(i)

        draw_patches = []
        lls = []
        for patch_dip_ll in range(np_h, 0, -1):
            for patch_strike_ll in range(np_w):
                ll = [patch_strike_ll * width, patch_dip_ll * height - height]
                draw_patches.append(
                    Rectangle(
                        ll, width=width, height=height, edgecolor='black'))
                lls.append(ll)

        llsa = num.vstack(lls)
        lower = llsa.min(axis=0)
        upper = llsa.max(axis=0)
        xlim = [lower[0], upper[0] + width]
        ylim = [lower[1], upper[1] + height]

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xlabel('strike-direction [km]')
        ax.set_ylabel('dip-direction [km]')

        xticker = tick.MaxNLocator(nbins=ntickmarks)
        yticker = tick.MaxNLocator(nbins=ntickmarks)

        ax.get_xaxis().set_major_locator(xticker)
        ax.get_yaxis().set_major_locator(yticker)

        scm = slip_colormap(100)
        pa_col = PatchCollection(
            draw_patches, alpha=alpha, match_original=True)
        pa_col.set(array=reference_slip, cmap=scm)

        ax.add_collection(pa_col)

        # patch central locations
        hpd = fault.ordering.patch_size_dip / 2.
        hps = fault.ordering.patch_size_strike / 2.

        xvec = num.linspace(hps, ext_source.length / km - hps, np_w)
        yvec = num.linspace(ext_source.width / km - hpd, hpd, np_h)

        xgr, ygr = num.meshgrid(xvec, yvec)

        if 'seismic' in fault.datatypes:
            if mtrace is not None:
                nuc_dip = transform(mtrace.get_values(
                    'nucleation_dip', combine=True, squeeze=True))
                nuc_strike = transform(mtrace.get_values(
                    'nucleation_strike', combine=True, squeeze=True))
                velocities = transform(mtrace.get_values(
                    'velocities', combine=True, squeeze=True))

                nchains = nuc_dip.size
                csteps = int(num.floor(nchains / ncontours))
                for i in range(0, nchains, csteps):
                    nuc_dip_idx, nuc_strike_idx = fault.fault_locations2idxs(
                        nuc_dip, nuc_strike, backend='numpy')
                    sts = fault.get_subfault_starttimes(
                        0, velocities[i, :], nuc_dip_idx, nuc_strike_idx)

                    ax.contour(xgr, ygr, sts, colors='gray', alpha=0.1)

            ref_starttimes = fault.point2starttimes(reference)
            contours = ax.contour(xgr, ygr, ref_starttimes, colors='black')
            plt.clabel(contours, inline=True, fontsize=10)

        if mtrace is not None:
            uparr = transform(
                mtrace.get_values('uparr', combine=True, squeeze=True))
            uperp = transform(
                mtrace.get_values('uperp', combine=True, squeeze=True))

            uparrstd = uparr.std(axis=0)
            uperpstd = uperp.std(axis=0)

            uparrmean = uparr.mean(axis=0)
            uperpmean = uperp.mean(axis=0)

            slipsx, slipsy = draw_quivers(
                ax, uperpmean, uparrmean, xgr, ygr,
                ext_source.rake, color='grey', draw_legend=False)

            slipvecrotmat = mt.euler_to_matrix(
                0.0, 0.0, ext_source.rake * mt.d2r)

            circle = num.linspace(0, 2 * num.pi, 100)
            # 2sigma error ellipses
            for i, (upe, upa) in enumerate(zip(uperpstd, uparrstd)):
                ellipse_x = 2 * upa * num.cos(circle)
                ellipse_y = 2 * upe * num.sin(circle)
                ellipse = num.vstack(
                    [ellipse_x, ellipse_y, num.zeros_like(ellipse_x)]).T
                rot_ellipse = ellipse.dot(slipvecrotmat)

                xcoords = xgr.ravel()[i] + rot_ellipse[:, 0] + slipsx[i]
                ycoords = ygr.ravel()[i] + rot_ellipse[:, 1] + slipsy[i]
                ax.plot(xcoords, ycoords, '-k', linewidth=0.5)

        draw_quivers(
            ax, reference['uperp'], reference['uparr'], xgr, ygr,
            ext_source.rake, color='black', draw_legend=True)

        cb = fig.colorbar(pa_col)
        cb.set_label('slip [m]')
        ax.set_aspect('equal', adjustable='box')
        figs.append(fig)
        axs.append(ax)

    return figs, axs


class ModeError(Exception):
    pass


def draw_slip_dist(problem, po):

    mode = problem.config.problem_config.mode

    if mode != 'ffi':
        raise ModeError(
            'Wrong optimization mode: %s! This plot '
            'variant is only valid for "ffi" mode' % mode)

    datatype, gc = problem.composites.items()[0]

    fault = gc.load_fault_geometry()

    sc = problem.config.sampler_config
    if po.load_stage is None and not sc.name == 'SMC':
        draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
    if po.load_stage == -2:
        raise ValueError('Slip distribution plot cannot be made for stage-2')
    else:
        draws = sc.parameters.n_steps

    transform = select_transform(sc=sc, n_steps=draws)

    stage = Stage(homepath=problem.outfolder)
    stage.load_results(
        model=problem.model, stage_number=po.load_stage, load='full')

    if po.reference is None:
        reference = get_result_point(stage, problem.config, po.post_llk)
        llk_str = po.post_llk
        mtrace = stage.mtrace
    else:
        reference = po.reference
        llk_str = 'ref'
        mtrace = None

    figs, axs = fault_slip_distribution(
        fault, mtrace, transform=transform, reference=reference)

    if po.outformat == 'display':
        plt.show()
    else:
        outpath = os.path.join(
            problem.outfolder, po.figure_dir,
            'slip_dist_%s.%s' % (llk_str, po.outformat))

        logger.info('Storing slip-distribution to: %s' % outpath)
        with PdfPages(outpath) as opdf:
            for fig in figs:
                opdf.savefig(fig, dpi=po.dpi)


plots_catalog = {
    'correlation_hist': draw_correlation_hist,
    'stage_posteriors': draw_posteriors,
    'waveform_fits': draw_seismic_fits,
    'scene_fits': draw_geodetic_fits,
    'velocity_models': draw_earthmodels,
    'slip_distribution': draw_slip_dist,
                }


def available_plots():
    return list(plots_catalog.keys())
