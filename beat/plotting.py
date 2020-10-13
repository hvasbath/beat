from pymc3 import plots as pmp
from pymc3 import quantiles

import math
import os
import logging
import copy

from beat import utility
from beat.models import Stage, load_stage
from beat.sampler.metropolis import get_trace_stats
from beat.heart import init_seismic_targets, init_geodetic_targets, \
                       physical_bounds
from beat.config import ffi_mode_str, geometry_mode_str, dist_vars

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick

from scipy.stats import kde
import numpy as num
from pyrocko.guts import (Object, String, Dict, List,
                          Bool, Int, load, StringChoice)
from pyrocko import util, trace
from pyrocko import cake_plot as cp
from pyrocko import orthodrome as otd

from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light
from pyrocko.plot import beachball, nice_value, AutoScaler

import pyrocko.moment_tensor as mt
from pyrocko.plot import mpl_papersize, mpl_init, mpl_graph_color, mpl_margins

logger = logging.getLogger('plotting')

km = 1000.


__all__ = [
    'PlotOptions', 'correlation_plot', 'correlation_plot_hist',
    'get_result_point', 'seismic_fits', 'scene_fits', 'traceplot',
    'select_transform', 'histplot_op']

u_nm = '$[Nm]$'
u_km = '$[km]$'
u_km_s = '$[km/s]$'
u_deg = '$[^{\circ}]$'
u_deg_myr = '$[^{\circ} / myr]$'
u_m = '$[m]$'
u_v = '$[m^3]$'
u_s = '$[s]$'
u_rad = '$[rad]$'
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
    'nucleation_x': u_hyp,
    'nucleation_y': u_hyp,
    'time_shift': u_s,
    'uperp': u_m,
    'uparr': u_m,
    'durations': u_s,
    'velocities': u_km_s,

    'mnn': u_nm,
    'mee': u_nm,
    'mdd': u_nm,
    'mne': u_nm,
    'mnd': u_nm,
    'med': u_nm,
    'magnitude': u_hyp,

    'pole_lat': u_deg,
    'pole_lon': u_deg,
    'omega': u_deg_myr,

    'w': u_rad,
    'v': u_rad,
    'kappa': u_rad,
    'sigma': u_rad,
    'h': u_hyp,

    'distance': u_km,
    'delta_depth': u_km,
    'delta_time': u_s,
    'time': u_s,
    'duration': u_s,
    'peak_ratio': u_hyp,
    'h_': u_hyp,
    'like': u_hyp}


plot_projections = ['latlon', 'local', 'individual']


def hypername(varname):
    if varname in list(plot_units.keys()):
        return varname
    else:
        return 'h_'



class PlotOptions(Object):
    post_llk = String.T(
        default='max',
        help='Which model to plot on the specified plot; Default: "max";'
             ' Options: "max", "min", "mean", "all"')
    plot_projection = StringChoice.T(
        default='local',
        choices=plot_projections,
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
        default={},
        help='Reference point for example from a synthetic test.',
        optional=True)
    outformat = String.T(default='pdf')
    dpi = Int.T(default=300)
    force = Bool.T(default=False)
    varnames = List.T(
        default=[], optional=True, help='Names of variables to plot')
    source_idxs = List.T(
        default=None,
        optional=True,
        help='Indexes to patches of slip distribution to draw marginals for')
    nensemble = Int.T(
        default=1,
        help='Number of draws from the PPD to display fuzzy results.')


def str_unit(quantity):
    """
    Return string representation of waveform unit.
    """
    if quantity == 'displacement':
        return '$m$'
    elif quantity == 'velocity':
        return '$m/s$'
    elif quantity == 'acceleration':
        return '$m/s^2$'
    else:
        raise ValueError('Quantity %s not supported!' % quantity)


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

    if t < 60.0:
        return s + '%.2g s' % t
    elif 60.0 <= t < 3600.:
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


def spherical_kde_op(
        lats0, lons0, lats=None, lons=None, grid_size=(200, 200), sigma=None):

    from scipy.special import logsumexp
    from beat.models.distributions import vonmises_fisher, vonmises_std

    if sigma is None:
        logger.debug('No sigma given, estimating VonMisesStd ...')
        sigmahat = vonmises_std(lats=lats0, lons=lons0)
        sigma = 1.06 * sigmahat * lats0.size ** -0.2
        logger.debug(
            'suggested sigma: %f, sigmahat: %f' % (sigma, sigmahat))

    if lats is None and lons is None:
        lats_vec = num.linspace(-90., 90, grid_size[0])
        lons_vec = num.linspace(-180., 180, grid_size[1])

        lons, lats = num.meshgrid(lons_vec, lats_vec)

    if lats is not None:
        assert lats.size == lons.size

    vmf = vonmises_fisher(
        lats=lats, lons=lons,
        lats0=lats0, lons0=lons0, sigma=sigma)
    kde = num.exp(logsumexp(vmf, axis=-1)).reshape(   # , b=self.weights)
        (grid_size[0], grid_size[1]))
    return kde, lats, lons


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

    fig, axs = plt.subplots(
        sharey='row', sharex='col',
        nrows=nvar - 1, ncols=nvar - 1, figsize=figsize)

    d = dict()
    for var in varnames:
        d[var] = transform(
            mtrace.get_values(
                var, combine=True, squeeze=True))

    for k in range(nvar - 1):
        a = d[varnames[k]]
        for l in range(k + 1, nvar):
            logger.debug('%s, %s' % (varnames[k], varnames[l]))
            b = d[varnames[l]]

            kde2plot(
                a, b, grid=grid, ax=axs[l - 1, k], cmap=cmap, aspect='auto')

            if point is not None:
                axs[l - 1, k].plot(
                    point[varnames[k]], point[varnames[l]],
                    color=point_color, marker=point_style,
                    markersize=point_size)

            axs[l - 1, k].tick_params(direction='in')

            if k == 0:
                axs[l - 1, k].set_ylabel(varnames[l])

        axs[l - 1, k].set_xlabel(varnames[k])

    for k in range(nvar - 1):
        for l in range(k):
            fig.delaxes(axs[l, k])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axs


def correlation_plot_hist(
        mtrace, varnames=None,
        transform=lambda x: x, figsize=None, hist_color='orange', cmap=None,
        grid=50, chains=None, ntickmarks=2, point=None,
        point_style='.', point_color='red', point_size=4, alpha=0.35,
        unify=True):
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
    unify: bool
        If true axis units that belong to one group e.g. [km] will
        have common axis increments

    Returns
    -------
    fig : figure object
    axs : subplot axis handles
    """
    fontsize = 9
    ntickmarks_max = 2
    label_pad = 25
    logger.info('Drawing correlation figure ...')

    if varnames is None:
        varnames = mtrace.varnames

    nvar = len(varnames)

    if figsize is None:
        if nvar < 5:
            figsize = mpl_papersize('a5', 'landscape')
        else:
            figsize = mpl_papersize('a4', 'landscape')

    fig, axs = plt.subplots(nrows=nvar, ncols=nvar, figsize=figsize)

    d = dict()

    for var in varnames:
        d[var] = transform(
            mtrace.get_values(
                var, chains=chains, combine=True, squeeze=True))

    hist_ylims = []
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
                            lw=point_size / 4.)
                    else:
                        reference = None
                else:
                    reference = None

                histplot_op(
                    axs[l, k], pmp.utils.make_2d(a), alpha=alpha,
                    color='orange', tstd=0., reference=reference)

                axs[l, k].get_yaxis().set_visible(False)
                format_axes(axs[l, k])
                xticks = axs[l, k].get_xticks()
                xlim = axs[l, k].get_xlim()
                hist_ylims.append(axs[l, k].get_ylim())
            else:
                b = d[v_nameb]

                kde2plot(
                    a, b, grid=grid, ax=axs[l, k], cmap=cmap, aspect='auto')

                bmin = b.min()
                bmax = b.max()

                if point is not None:
                    if v_namea and v_nameb in point.keys():
                        axs[l, k].plot(
                            point[v_namea], point[v_nameb],
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
                    v_nameb + '\n ' + plot_units[hypername(v_nameb)],
                    fontsize=fontsize)
                if utility.is_odd(l):
                    axs[l, k].tick_params(axis='y', pad=label_pad)
            else:
                axs[l, k].get_yaxis().set_ticklabels([])

            axs[l, k].tick_params(
                axis='both', direction='in', labelsize=fontsize)

            try:    # matplotlib version issue workaround
                axs[l, k].tick_params(
                    axis='both', labelrotation=50.)
            except Exception:
                axs[l, k].set_xticklabels(
                    axs[l, k].get_xticklabels(), rotation=50)
                axs[l, k].set_yticklabels(
                    axs[l, k].get_yticklabels(), rotation=50)

            if utility.is_odd(k):
                axs[l, k].tick_params(axis='x', pad=label_pad)

        axs[l, k].set_xlabel(
            v_namea + '\n ' + plot_units[hypername(v_namea)], fontsize=fontsize)

    if unify:
        varnames_repeat_x = [
            var_reap for varname in varnames for var_reap in (varname,) * nvar]
        varnames_repeat_y = varnames * nvar
        unitiesx = unify_tick_intervals(
            axs, varnames_repeat_x, ntickmarks_max=ntickmarks_max, axis='x')
        apply_unified_axis(
            axs, varnames_repeat_x, unitiesx, axis='x', scale_factor=1.,
            ntickmarks_max=ntickmarks_max)
        apply_unified_axis(
            axs, varnames_repeat_y, unitiesx, axis='y', scale_factor=1.,
            ntickmarks_max=ntickmarks_max)

    for k in range(nvar):
        if unify:
            # reset histogram ylims after unify
            axs[k, k].set_ylim(hist_ylims[k])

        for l in range(k):
            fig.delaxes(axs[l, k])

    fig.tight_layout()
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

    ax = plt.axes()
    im = ax.scatter(
        uwifg.lons, uwifg.lats, point_size, uwifg.displacement,
        edgecolors='none')
    plt.colorbar(im)
    plt.title('Displacements [m] %s' % uwifg.name)
    plt.show()


def plot_cov(target, point_size=20):

    ax = plt.axes()
    im = ax.scatter(
        target.lons, target.lats, point_size,
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
    if point_llk != 'None':
        sampler_name = config.sampler_config.name
        if sampler_name == 'Metropolis':
            if stage.step is None:
                raise AttributeError(
                    'Loading Metropolis results requires'
                    ' sampler parameters to be loaded!')

            sc = config.sampler_config.parameters
            pdict, _ = get_trace_stats(
                stage.mtrace, stage.step, sc.burn, sc.thin)
            point = pdict[point_llk]
        elif sampler_name == 'SMC' or sampler_name == 'PT':
            llk = stage.mtrace.get_values(
                varname='like',
                combine=True)

            posterior_idxs = utility.get_fit_indexes(llk)

            point = stage.mtrace.point(idx=posterior_idxs[point_llk])
        else:
            raise NotImplementedError(
                'Sampler "%s" is not supported!' % config.sampler_config.name)
    else:
        point = None

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


def format_axes(
        ax, remove=['right', 'top', 'left'], linewidth=None, visible=False):
    """
    Removes box top, left and right.
    """
    for rm in remove:
        ax.spines[rm].set_visible(visible)
        if linewidth is not None:
            ax.spines[rm].set_linewidth(linewidth)


def scale_axes(axis, scale, offset=0.):
    from matplotlib.ticker import ScalarFormatter

    class FormatScaled(ScalarFormatter):

        @staticmethod
        def __call__(value, pos):
            return '{:,.1f}'.format(offset + value * scale).replace(',', ' ')

    axis.set_major_formatter(FormatScaled())


def set_anchor(sources, anchor):
    for source in sources:
        source.anchor = anchor


def gnss_fits(problem, stage, plot_options):

    from pyrocko import automap
    from pyrocko.model import gnss

    if len(automap.gmtpy.detect_gmt_installations()) < 1:
        raise automap.gmtpy.GmtPyError(
            'GMT needs to be installed for GNSS plot!')

    gc = problem.config.geodetic_config
    try:
        ds_config = gc.types['GNSS']
    except KeyError:
        raise ImportError('No GNSS data in configuration!')

    logger.info('Trying to load GNSS data from: {}'.format(ds_config.datadir))

    campaigns = ds_config.load_data(campaign=True)
    if not campaigns:
        raise ImportError('Did not fing GNSS data under %s' % ds_config.datadir)

    if len(campaigns) > 1:
        logger.warning(
            'Plotting for more than 1 GNSS dataset needs tp be implemented')

    campaign = campaigns[0]

    datatype = 'geodetic'
    mode = problem.config.problem_config.mode
    problem.init_hierarchicals()

    figsize = 20.  # size in cm

    po = plot_options

    composite = problem.composites[datatype]
    try:
        sources = composite.sources
        ref_sources = None
    except AttributeError:
        logger.info('FFI gnss fit, using reference source ...')
        ref_sources = composite.config.gf_config.reference_sources
        set_anchor(ref_sources, anchor='top')
        fault = composite.load_fault_geometry()
        sources = fault.get_all_subfaults(
            datatype=datatype, component=composite.slip_varnames[0])
        set_anchor(sources, anchor='top')

    if po.reference:
        if mode != ffi_mode_str:
            composite.point2sources(po.reference)
            ref_sources = copy.deepcopy(composite.sources)
        point = po.reference
    else:
        point = get_result_point(stage, problem.config, po.post_llk)

    results = composite.assemble_results(point)

    dataset_to_result = {}
    for dataset, result in zip(composite.datasets, results):
        if dataset.typ == 'GNSS':
            dataset_to_result[dataset] = result

    if po.plot_projection == 'latlon':
        event = problem.config.event
        locations = campaign.stations # + [event]
        #print(locations)
        #lat, lon = otd.geographic_midpoint_locations(locations)

        coords = num.array([loc.effective_latlon for loc in locations])
        lat, lon = num.mean(num.vstack([coords.min(0), coords.max(0)]), axis=0)

    elif po.plot_projection == 'local':
        lat, lon = otd.geographic_midpoint_locations(sources)
        coords = num.hstack(
            [source.outline(cs='latlon').T for source in sources]).T

    else:
        raise NotImplementedError(
            '%s projection not implemented!' % po.plot_projection)

    radius = otd.distance_accurate50m_numpy(
        lat[num.newaxis], lon[num.newaxis],
        coords[:, 0], coords[:, 1]).max()

    radius *= 1.2

    if radius < 30. * km:
        logger.warning(
            'Radius of GNSS campaign %s too small, defaulting'
            ' to 30 km' % campaign.name)
        radius = 30 * km

    model_camp = gnss.GNSSCampaign(
        stations=copy.deepcopy(campaign.stations),
        name='model')

    for dataset, result in dataset_to_result.items():
        for ista, sta in enumerate(model_camp.stations):
            comp = getattr(sta, dataset.component)
            comp.shift = result.processed_syn[ista]
            comp.sigma = 0.

    plot_component_flags = []
    if 'east' in ds_config.components or 'north' in ds_config.components:
        plot_component_flags.append(False)

    if 'up' in ds_config.components:
        plot_component_flags.append(True)

    figs = []
    for vertical in plot_component_flags:
        m = automap.Map(
            width=figsize,
            height=figsize,
            lat=lat,
            lon=lon,
            radius=radius,
            show_topo=False,
            show_grid=True,
            show_rivers=True,
            color_wet=(216, 242, 254),
            color_dry=(238, 236, 230))

        all_stations = campaign.stations + model_camp.stations
        offset_scale = num.zeros(len(all_stations))

        for ista, sta in enumerate(all_stations):
            for comp in sta.components.values():
                offset_scale[ista] += comp.shift

        offset_scale = num.sqrt(offset_scale ** 2).max()

        if len(campaign.stations) > 40:
            logger.warning('More than 40 stations disabling station labels ..')
            labels = False
        else:
            labels = True

        m.add_gnss_campaign(
            campaign,
            psxy_style={
                'G': 'black',
                'W': '0.8p,black',
            },
            offset_scale=offset_scale,
            vertical=vertical,
            labels=labels)

        m.add_gnss_campaign(
            model_camp,
            psxy_style={
                'G': 'red',
                'W': '0.8p,red',
                't': 30,
            },
            offset_scale=offset_scale,
            vertical=vertical,
            labels=False)

        for i, source in enumerate(sources):
            in_rows = source.outline(cs='lonlat')
            if mode != ffi_mode_str:
                color = (num.array(mpl_graph_color(i)) * 255).tolist()
                color_str = utility.list2string(color, '/')
            else:
                color_str = 'black'

            if in_rows.shape[0] > 1:    # finite source
                m.gmt.psxy(
                    in_rows=in_rows,
                    L='+p0.1p,%s' % color_str,
                    W='0.1p,black',
                    G=color_str,
                    t=70,
                    *m.jxyr)
                m.gmt.psxy(
                    in_rows=in_rows[0:2],
                    W='1p,black',
                    *m.jxyr)
            else:                       # point source
                source_scale_factor = 2.
                m.gmt.psxy(
                    in_rows=in_rows,
                    W='0.1p,black',
                    G=color_str,
                    S='c%fp' % float(source.magnitude * source_scale_factor),
                    t=70,
                    *m.jxyr)

        figs.append(m)

    return figs


def map_displacement_grid(displacements, scene):
    arr = num.full_like(scene.displacement, fill_value=num.nan)
    qt = scene.quadtree

    for syn_v, l in zip(displacements, qt.leaves):
        arr[l._slice_rows, l._slice_cols] = syn_v

    arr[scene.displacement_mask] = num.nan
    return arr


def scene_fits(problem, stage, plot_options):
    """
    Plot geodetic data, synthetics and residuals.
    """
    from pyrocko.dataset import gshhg
    from kite.scene import Scene, UserIOWarning
    import gc

    try:
        homepath = problem.config.geodetic_config.types['SAR'].datadir
    except KeyError:
        raise ValueError('SAR data not in geodetic types!')

    datatype = 'geodetic'
    mode = problem.config.problem_config.mode
    problem.init_hierarchicals()

    fontsize = 10
    fontsize_title = 12
    ndmax = 3
    nxmax = 3
    cmap = plt.cm.jet

    po = plot_options

    composite = problem.composites[datatype]
    event = composite.event
    try:
        sources = composite.sources
        ref_sources = None
    except AttributeError:
        logger.info('FFI scene fit, using reference source ...')
        ref_sources = composite.config.gf_config.reference_sources
        set_anchor(ref_sources, anchor='top')
        fault = composite.load_fault_geometry()
        sources = fault.get_all_subfaults(
            datatype=datatype, component=composite.slip_varnames[0])
        set_anchor(sources, anchor='top')

    if po.reference:
        if mode != ffi_mode_str:
            composite.point2sources(po.reference)
            ref_sources = copy.deepcopy(composite.sources)
        point = po.reference
    else:
        point = get_result_point(stage, problem.config, po.post_llk)

    results_tmp = composite.assemble_results(point)
    dataset_to_result = {}
    for dataset, result in zip(composite.datasets, results_tmp):
        if dataset.typ == 'SAR':
            dataset_to_result[dataset] = result

    results = dataset_to_result.values()
    dataset_index = dict(
        (data, i) for (i, data) in enumerate(dataset_to_result.keys()))

    nrmax = len(dataset_to_result.keys())
    fullfig, restfig = utility.mod_i(nrmax, ndmax)
    factors = num.ones(fullfig).tolist()
    if restfig:
        factors.append(float(restfig) / ndmax)

    figures = []
    axes = []
    for f in factors:
        figsize = list(mpl_papersize('a4', 'portrait'))
        figsize[1] *= f

        fig, ax = plt.subplots(
            nrows=int(round(ndmax * f)), ncols=nxmax, figsize=figsize)
        fig.tight_layout()
        fig.subplots_adjust(
            left=0.08,
            right=1.0 - 0.03,
            bottom=0.06,
            top=1.0 - 0.06,
            wspace=0.,
            hspace=0.1)
        figures.append(fig)
        ax_a = num.atleast_2d(ax)
        axes.append(ax_a)

    nfigs = len(figures)

    def axis_config(axes, source, scene, po):

        for i, ax in enumerate(axes):
            if po.plot_projection == 'latlon':
                ystr = 'Latitude [deg]'
                xstr = 'Longitude [deg]'
                if scene.frame.isDegree():
                    scale_x = {'scale': 1.}
                    scale_y = {'scale': 1.}
                else:
                    scale_x = {'scale': otd.m2d}
                    scale_y = {'scale': otd.m2d}

                scale_x['offset'] = source.lon
                scale_y['offset'] = source.lat

            elif po.plot_projection == 'local':
                ystr = 'Distance [km]'
                xstr = 'Distance [km]'
                if scene.frame.isDegree():
                    scale_x = {'scale': otd.d2m / km}
                    scale_y = {'scale': otd.d2m / km}
                else:
                    scale_x = {'scale': 1. / km}
                    scale_y = {'scale': 1. / km}
            else:
                raise TypeError(
                    'Plot projection %s not available' % po.plot_projection)

            ticker = tick.MaxNLocator(nbins=3)
            ax.get_xaxis().set_major_locator(ticker)
            ax.get_yaxis().set_major_locator(ticker)

            if i == 0:
                ax.set_ylabel(ystr, fontsize=fontsize)
                ax.set_xlabel(xstr, fontsize=fontsize)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

            scale_axes(ax.get_xaxis(), **scale_x)
            scale_axes(ax.get_yaxis(), **scale_y)
            ax.set_aspect('equal')

            if i > 0:
                ax.set_yticklabels([])
                ax.set_xticklabels([])

    def draw_coastlines(ax, xlim, ylim, event, scene, po):
        """
        xlim and ylim in Lon/Lat[deg]
        """

        logger.debug('Drawing coastlines ...')
        coasts = gshhg.GSHHG.full()

        if po.plot_projection == 'latlon':
            west, east = xlim
            south, north = ylim

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

                if scene.frame.isMeter():
                    ys, xs = otd.latlon_to_ne_numpy(
                        event.lat, event.lon, p.lats, p.lons)

                elif scene.frame.isDegree():

                    xs = p.lons - event.lon
                    ys = p.lats - event.lat

                ax.plot(xs, ys, '-k', linewidth=0.5)

    def addArrow(ax, scene):
        phi = num.nanmean(scene.phi)
        los_dx = num.cos(phi + num.pi) * .0625
        los_dy = num.sin(phi + num.pi) * .0625

        az_dx = num.cos(phi - num.pi / 2) * .125
        az_dy = num.sin(phi - num.pi / 2) * .125

        anchor_x = .9 if los_dx < 0 else .1
        anchor_y = .85 if los_dx < 0 else .975

        az_arrow = FancyArrow(
            x=anchor_x - az_dx, y=anchor_y - az_dy,
            dx=az_dx, dy=az_dy,
            head_width=.025,
            alpha=.5, fc='k',
            head_starts_at_zero=False,
            length_includes_head=True,
            transform=ax.transAxes)

        los_arrow = FancyArrow(
            x=anchor_x - az_dx / 2, y=anchor_y - az_dy / 2,
            dx=los_dx, dy=los_dy,
            head_width=.02,
            alpha=.5, fc='k',
            head_starts_at_zero=False,
            length_includes_head=True,
            transform=ax.transAxes)

        ax.add_artist(az_arrow)
        ax.add_artist(los_arrow)

    def draw_leaves(ax, scene, offset_e=0, offset_n=0):
        rects = scene.quadtree.getMPLRectangles()
        for r in rects:
            r.set_edgecolor((.4, .4, .4))
            r.set_linewidth(.5)
            r.set_facecolor('none')
            r.set_x(r.get_x() - offset_e)
            r.set_y(r.get_y() - offset_n)
        map(ax.add_artist, rects)

        ax.scatter(scene.quadtree.leaf_coordinates[:, 0] - offset_e,
                   scene.quadtree.leaf_coordinates[:, 1] - offset_n,
                   s=.25, c='black', alpha=.1)

    def draw_sources(ax, sources, scene, po, event, **kwargs):
        bgcolor = kwargs.pop('color', None)

        for i, source in enumerate(sources):

            if scene.frame.isMeter():
                fn, fe = source.outline(cs='xy').T
            elif scene.frame.isDegree():
                fn, fe = source.outline(cs='latlon').T
                fn -= event.lat
                fe -= event.lon

            if not bgcolor:
                color = mpl_graph_color(i)
            else:
                color = bgcolor

            if fn.size > 1:
                alpha = 0.4
                ax.plot(
                    fe, fn, '-',
                    linewidth=0.5, color=color, alpha=alpha, **kwargs)
                ax.fill(
                    fe, fn,
                    edgecolor=color,
                    facecolor=light(color, .5), alpha=alpha)
                ax.plot(
                    fe[0:2], fn[0:2], '-k', alpha=0.7,
                    linewidth=1.0)
            else:
                ax.plot(
                    fe[:, 0], fn[:, 1], marker='*',
                    markersize=10, color=color, **kwargs)

    def cbtick(x):
        rx = math.floor(x * 1000.) / 1000.
        return [-rx, rx]

    colims = [num.max([
        num.max(num.abs(r.processed_obs)),
        num.max(num.abs(r.processed_syn))]) for r in results]
    dcolims = [num.max(num.abs(r.processed_res)) for r in results]

    import string
    for idata, (dataset, result) in enumerate(dataset_to_result.items()):
        subplot_letter = string.ascii_lowercase[idata]
        try:
            scene_path = os.path.join(homepath, dataset.name)
            logger.info(
                'Loading full resolution kite scene: %s' % scene_path)
            scene = Scene.load(scene_path)
        except UserIOWarning:
            logger.warning(
                'Full resolution data could not be loaded! Skipping ...')
            continue

        if scene.frame.isMeter():
            offset_n, offset_e = map(float, otd.latlon_to_ne_numpy(
                scene.frame.llLat, scene.frame.llLon,
                event.lat, event.lon))

        elif scene.frame.isDegree():
            offset_n = event.lat - scene.frame.llLat
            offset_e = event.lon - scene.frame.llLon

        im_extent = (scene.frame.E.min() - offset_e,
                     scene.frame.E.max() - offset_e,
                     scene.frame.N.min() - offset_n,
                     scene.frame.N.max() - offset_n)

        urE, urN, llE, llN = (0., 0., 0., 0.)

        turE, turN, tllE, tllN = zip(
            *[(l.gridE.max() - offset_e,
               l.gridN.max() - offset_n,
               l.gridE.min() - offset_e,
               l.gridN.min() - offset_n)
              for l in scene.quadtree.leaves])

        turE, turN = map(max, (turE, turN))
        tllE, tllN = map(min, (tllE, tllN))
        urE, urN = map(max, ((turE, urE), (urN, turN)))
        llE, llN = map(min, ((tllE, llE), (llN, tllN)))

        lat, lon = otd.ne_to_latlon(
            sources[0].lat, sources[0].lon,
            num.array([llN, urN]), num.array([llE, urE]))

        #result = dataset_to_result[dataset]
        tidx = dataset_index[dataset]

        figidx, rowidx = utility.mod_i(tidx, ndmax)
        axs = axes[figidx][rowidx, :]

        imgs = []
        for ax, data_str in zip(axs, ['obs', 'syn', 'res']):
            logger.info('Plotting %s' % data_str)
            datavec = getattr(result, 'processed_%s' % data_str)

            if data_str == 'res' and po.plot_projection == 'local':
                vmin = -dcolims[tidx]
                vmax = dcolims[tidx]
            else:
                vmin = -colims[tidx]
                vmax = colims[tidx]

            imgs.append(ax.imshow(
                map_displacement_grid(datavec, scene),
                extent=im_extent, cmap=cmap,
                vmin=vmin, vmax=vmax,
                origin='lower'))

            ax.set_xlim(llE, urE)
            ax.set_ylim(llN, urN)

            draw_leaves(ax, scene, offset_e, offset_n)
            draw_coastlines(
                ax, lon, lat, event, scene, po)

        fontdict = {
            'fontsize': fontsize,
            'fontweight': 'bold',
            'verticalalignment': 'top'}

        transform = axes[figidx][rowidx, 0].transAxes

        if dataset.name[-5::] == 'dscxn':
            title = 'descending'
        elif dataset.name[-5::] == 'ascxn':
            title = 'ascending'
        else:
            title = dataset.name

        axes[figidx][rowidx, 0].text(
            .025, 1.025, '({}) {}'.format(subplot_letter, title),
            fontsize=fontsize_title, alpha=1.,
            va='bottom', transform=transform)
        for i, quantity in enumerate(['data', 'model', 'residual']):
            transform = axes[figidx][rowidx, i].transAxes
            axes[figidx][rowidx, i].text(
                0.5, 0.95, quantity, fontdict, transform=transform,
                horizontalalignment='center')

        draw_sources(
            axes[figidx][rowidx, 1], sources, scene, po, event=event)

        if ref_sources:
            ref_color = scolor('aluminium4')
            logger.info('Plotting reference sources')
            draw_sources(
                axes[figidx][rowidx, 1],
                ref_sources, scene, po, event=event, color=ref_color)

        f = factors[figidx]
        if f > 2. / 3:
            cbb = (0.68 - (0.3075 * rowidx))
        elif f > 1. / 2:
            cbb = (0.53 - (0.47 * rowidx))
        elif f > 1. / 4:
            cbb = (0.06)

        cbl = 0.46
        cbw = 0.15
        cbh = 0.01

        cbaxes = figures[figidx].add_axes([cbl, cbb, cbw, cbh])

        cblabel = 'LOS displacement [m]'
        cbs = plt.colorbar(
            imgs[1],
            ax=axes[figidx][rowidx, 0],
            ticks=cbtick(colims[tidx]),
            cax=cbaxes,
            orientation='horizontal',
            cmap=cmap)
        cbs.set_label(cblabel, fontsize=fontsize)

        if po.plot_projection == 'local':
            dcbaxes = figures[figidx].add_axes([cbl + 0.3, cbb, cbw, cbh])
            cbr = plt.colorbar(
                imgs[2],
                ax=axes[figidx][rowidx, 2],
                ticks=cbtick(dcolims[tidx]),
                cax=dcbaxes,
                orientation='horizontal',
                cmap=cmap)
            cbr.set_label(cblabel, fontsize=fontsize)

        axis_config(axes[figidx][rowidx, :], event, scene, po)
        addArrow(axes[figidx][rowidx, 0], scene)

        del scene
        gc.collect()

    return figures


def draw_scene_fits(problem, plot_options):

    if 'geodetic' not in list(problem.composites.keys()):
        raise TypeError('No geodetic composite defined in the problem!')

    if 'SAR' not in problem.config.geodetic_config.types:
        raise TypeError('There is no SAR data in the problem setup!')

    logger.info('Drawing SAR misfits ...')
    po = plot_options

    stage = Stage(homepath=problem.outfolder,
                  backend=problem.config.sampler_config.backend)

    if not po.reference:
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model, stage_number=po.load_stage,
            load='trace', chains=[-1])
        llk_str = po.post_llk
    else:
        llk_str = 'ref'

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'scenes_%s_%s_%s' % (
            stage.number, llk_str, po.plot_projection))

    if not os.path.exists(outpath) or po.force:
        figs = scene_fits(problem, stage, po)
    else:
        logger.info('scene plots exist. Use force=True for replotting!')
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


def draw_gnss_fits(problem, plot_options):

    if 'geodetic' not in list(problem.composites.keys()):
        raise TypeError('No geodetic composite defined in the problem!')

    if 'GNSS' not in problem.config.geodetic_config.types:
        raise TypeError('There is no SAR data in the problem setup!')

    logger.info('Drawing GNSS misfits ...')

    po = plot_options

    stage = Stage(homepath=problem.outfolder,
                  backend=problem.config.sampler_config.backend)

    if not po.reference:
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model, stage_number=po.load_stage,
            load='trace', chains=[-1])
        llk_str = po.post_llk
    else:
        llk_str = 'ref'

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'gnss_%s_%s_%s' % (
            stage.number, llk_str, po.plot_projection))

    if not os.path.exists(outpath) or po.force:
        figs = gnss_fits(problem, stage, po)
    else:
        logger.info('scene plots exist. Use force=True for replotting!')
        return

    if po.outformat == 'display':
        plt.show()
    else:
        logger.info('saving figures to %s' % outpath)
        for component, fig in zip(('horizontal', 'vertical'), figs):
            fig.save(outpath + '_%s.%s' % (
                component, po.outformat), resolution=po.dpi)


def extract_time_shifts(point, wmap):
    try:
        time_shifts = point[wmap.time_shifts_id][
            wmap.station_correction_idxs]
    except KeyError:
        raise ValueError(
            'Sampling results do not contain time-shifts for wmap'
            ' %s!' % wmap.time_shifts_id)
    return time_shifts


def seismic_fits(problem, stage, plot_options):
    """
    Modified from grond. Plot synthetic and data waveforms and the misfit for
    the selected posterior model.
    """

    def plot_trace(axes, tr, **kwargs):
        return axes.plot(tr.get_xdata(), tr.get_ydata(), **kwargs)

    def plot_taper(axes, t, taper, mode='geometry', **kwargs):
        y = num.ones(t.size) * 0.9
        if mode == 'geometry':
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

    def plot_inset_hist(
            axes, data, best_data, bbox_to_anchor,
            cmap=None, cbounds=None, color='orange', alpha=0.4):

        in_ax = inset_axes(
            axes, width="100%", height="100%",
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=axes.transAxes, loc=2, borderpad=0)
        histplot_op(
            in_ax, data,
            alpha=alpha, color=color, cmap=cmap, cbounds=cbounds, tstd=0.)

        format_axes(in_ax)
        linewidth = 0.5
        format_axes(
            in_ax, remove=['bottom'], visible=True,
            linewidth=linewidth)
        in_ax.axvline(
            x=best_data,
            color='red', lw=linewidth)
        in_ax.tick_params(
            axis='both', direction='in', labelsize=5,
            width=linewidth)
        in_ax.get_yaxis().set_visible(False)
        xticker = tick.MaxNLocator(nbins=2)
        in_ax.get_xaxis().set_major_locator(xticker)
        return in_ax

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    composite = problem.composites['seismic']

    fontsize = 8
    fontsize_title = 10

    target_index = dict(
        (target, i) for (i, target) in enumerate(composite.targets))

    po = plot_options

    if not po.reference:
        best_point = get_result_point(stage, problem.config, po.post_llk)
    else:
        best_point = po.reference

    composite.analyse_noise(best_point, chop_bounds=['a', 'd'])
    composite.update_weights(best_point)
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
            results = composite.assemble_results(point)
            ens_results.append(results)
            ens_var_reductions.append(
                composite.get_variance_reductions(
                    point, weights=composite.weights, results=results))

    if best_point:
        bresults = composite.assemble_results(
            best_point, outmode='tapered_data')   # for source individual contributions
    else:
        # get dummy results for data
        bresults = composite.assemble_results(point)
        best_point = point

    bvar_reductions = composite.get_variance_reductions(
        best_point, weights=composite.weights, results=bresults)

    try:
        composite.point2sources(best_point)
        source = composite.sources[0]
    except AttributeError:
        logger.info('FFI waveform fit, using reference source ...')
        source = composite.config.gf_config.reference_sources[0]
        source.time = composite.event.time

    # collecting results for targets
    logger.info('Mapping results to targets ...')
    target_to_results = {}
    all_syn_trs_target = {}
    all_var_reductions = {}
    dtraces = []

    for target in composite.targets:
        target_results = []
        target_synths = []
        target_var_reductions = []

        i = target_index[target]

        nslc_id_str = utility.list2string(target.codes)
        target_results.append(bresults[i])
        target_synths.append(bresults[i].processed_syn)
        target_var_reductions.append(
            bvar_reductions[nslc_id_str])

        dtraces.append(copy.deepcopy(bresults[i].processed_res))
        if plot_options.nensemble > 1:
            for results, var_reductions in zip(
                    ens_results, ens_var_reductions):
                # put all results per target here not only single
                target_results.append(results[i])
                target_synths.append(results[i].processed_syn)
                target_var_reductions.append(
                    var_reductions[nslc_id_str])

        target_to_results[target] = target_results
        all_syn_trs_target[target] = target_synths
        all_var_reductions[target] = num.array(target_var_reductions) * 100.

    # collecting time-shifts:
    station_corr = composite.config.station_corrections
    if station_corr:
        tshifts = problem.config.problem_config.hierarchicals['time_shift']
        time_shift_bounds = [tshifts.lower, tshifts.upper]

        logger.info('Collecting time-shifts ...')
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
        all_time_shifts = {}
        for target in composite.targets:
            target_time_shifts = []
            i = target_index[target]
            target_time_shifts.append(btime_shifts[i])

            if plot_options.nensemble > 1:
                for time_shifts in ens_time_shifts:
                    target_time_shifts.append(time_shifts[i])

            all_time_shifts[target] = num.array(target_time_shifts)

    skey = lambda tr: tr.channel

#    trace_minmaxs = trace.minmax(all_syn_trs, skey)
    dminmaxs = trace.minmax(dtraces, skey)
    for tr in dtraces:
        if tr:
            dmin, dmax = dminmaxs[skey(tr)]
            tr.ydata /= max(abs(dmin), abs(dmax))

    cg_to_targets = utility.gather(
        composite.targets,
        lambda t: t.codes[3],
        filter=lambda t: t in target_to_results)

    cgs = cg_to_targets.keys()

    figs = []
    logger.info('Plotting waveforms ...')
    for cg in cgs:
        targets = cg_to_targets[cg]

        # can keep from here ... until
        nframes = len(targets)

        nx = int(math.ceil(math.sqrt(nframes)))
        ny = (nframes - 1) // nx + 1

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
                        wspace=0.2,
                        hspace=0.2)

                    figs.append(figures[iyy, ixx])

                fig = figures[iyy, ixx]

                target = frame_to_target[iy, ix]

                # get min max of all traces
                key = target.codes[3]
                amin, amax = trace.minmax(
                    all_syn_trs_target[target],
                    key=skey)[key]
                # need target specific minmax
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

                ymin, ymax = - absmax * 1.33 * space_factor, absmax * 1.33
                axes.set_ylim(ymin, ymax)

                itarget = target_index[target]
                result = bresults[itarget]

                traces = all_syn_trs_target[target]

                dtrace = dtraces[itarget]

                if po.nensemble > 1:
                    xmin, xmax = trace.minmaxtime(traces, key=skey)[key]
                    fuzzy_waveforms(
                        axes, traces, linewidth=7, zorder=0,
                        grid_size=(500, 500), alpha=1.0)

                tap_color_annot = (0.35, 0.35, 0.25)
                tap_color_edge = (0.85, 0.85, 0.80)
                #tap_color_fill = (0.95, 0.95, 0.90)

                plot_taper(
                    axes2, result.processed_obs.get_xdata(), result.taper,
                    mode=composite._mode, fc='None', ec=tap_color_edge,
                    zorder=4, alpha=0.6)

                obs_color = scolor('aluminium5')
                syn_color = scolor('scarletred2')
                misfit_color = scolor('scarletred2')

                if best_point:
                    # only draw if highlighted point exists
                    plot_dtrace(
                        axes2, dtrace, space, 0., 1.,
                        fc=light(misfit_color, 0.3),
                        ec=misfit_color, zorder=4)

                    if po.plot_projection == 'individual':
                        for i, tr in enumerate(result.source_contributions):
                            plot_trace(
                                axes, tr,
                                color=mpl_graph_color(i), lw=0.5, zorder=5)
                    else:
                        plot_trace(
                            axes, result.processed_syn,
                            color=syn_color, lw=0.5, zorder=5)

                plot_trace(
                    axes, result.processed_obs,
                    color=obs_color, lw=0.5, zorder=5)

                xdata = result.processed_obs.get_xdata()
                axes.set_xlim(xdata[0], xdata[-1])

                tmarks = [
                    result.processed_obs.tmin,
                    result.processed_obs.tmax]
                tmark_fontsize = fontsize - 1

                # plot variance reductions
                if po.nensemble > 1:
                    nslc_id_str = utility.list2string(target.codes)
                    logger.debug(
                        'Plotting variance reductions for %s' % nslc_id_str)
                    in_ax = plot_inset_hist(
                        axes,
                        data=pmp.utils.make_2d(all_var_reductions[target]),
                        best_data=bvar_reductions[nslc_id_str] * 100.,
                        bbox_to_anchor=(0.9, .75, .2, .2))
                    in_ax.set_title('VR [%]', fontsize=5)

                if station_corr:
                    sidebar_ybounds = [-0.9, -1,3]
                    ytmarks = [-1.3, -1.3]
                    hor_alignment = 'center'
                    if po.nensemble > 1:
                        in_ax = plot_inset_hist(
                            axes,
                            data=pmp.utils.make_2d(all_time_shifts[target]),
                            best_data=btime_shifts[itarget],
                            bbox_to_anchor=(-0.0985, .26, .2, .2),
                            cmap=plt.cm.get_cmap('seismic'),
                            cbounds=time_shift_bounds,
                            color=None,
                            alpha=1.)
                        in_ax.set_xlim(*time_shift_bounds)
                else:
                    sidebar_ybounds = [-1.2, -1.2]
                    ytmarks = [-1.2, -1.2]
                    hor_alignment = 'left'

                for tmark, ybound in zip(tmarks, sidebar_ybounds):
                    axes2.plot(
                        [tmark, tmark], [ybound, 0.1], color=tap_color_annot)

                for xtmark, ytmark, text, ha, va in [
                        (tmarks[0], ytmarks[0],
                         '$\,$ ' + str_duration(tmarks[0] - source.time),
                         hor_alignment,
                         'bottom'),
                        (tmarks[1], ytmarks[1],
                         '$\Delta$ ' + str_duration(tmarks[1] - tmarks[0]),
                         'right',
                         'bottom')]:

                    axes2.annotate(
                        text,
                        xy=(xtmark, ytmark),
                        xycoords='data',
                        xytext=(
                            fontsize * 0.4 * [-1, 1][ha == 'left'],
                            fontsize * 0.2),
                        textcoords='offset points',
                        ha=ha,
                        va=va,
                        color=tap_color_annot,
                        fontsize=tmark_fontsize, zorder=10)

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

                # annotate axis amplitude
                axes.annotate(
                    '%0.3g %s -' % (-absmax, str_unit(target.quantity)),
                    xycoords='data',
                    xy=(tmarks[1], -absmax),
                    xytext=(1., 1.),
                    textcoords='offset points',
                    ha='right',
                    va='center',
                    fontsize=fontsize - 3,
                    color=obs_color,
                    fontstyle='normal')

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

    if not po.reference:
        rpoint = None
        llk_str = po.post_llk
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

        point = get_result_point(stage, problem.config, po.post_llk)
        best_mt = point2array(point, varnames=varnames, rpoint=rpoint)
    else:
        llk_str = 'ref'
        m6s = [point2array(point=po.reference, varnames=varnames)]
        best_mt = None

    return m6s, best_mt, llk_str


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
        'position': (5, 5),
        'color_t': 'black',
        'edgecolor': 'black',
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

        beachball.plot_fuzzy_beachball_mpl_pixmap(
            m6s, axes, best_mt=best_mt, best_color='red', **kwargs)

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

    magnitude_full_max = max( m6s.mean(axis=0)[-1] for (_, m6s, _) in lines)

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
                        label = '{:03.1f}-{:03.1f}%'.format(ratios_qu[2.5], ratios_qu[97.5])
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

    m6s, _, llk_str = extract_mt_components(problem, po, include_magnitude=True)

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


def histplot_op(
        ax, data, reference=None, alpha=.35, color=None, cmap=None, bins=None,
        tstd=None, qlist=[0.01, 99.99], cbounds=None, kwargs={}):
    """
    Modified from pymc3. Additional color argument.
    """
    if color is not None and cmap is not None:
        logger.debug('Using color for histogram edgecolor ...')

    if cmap is not None:
        from matplotlib.colors import Colormap
        if not isinstance(cmap, Colormap):
            raise TypeError(
                'The colormap needs to be a valid matplotlib colormap!')

        histtype = 'bar'
    else:
        histtype = 'stepfilled'

    for i in range(data.shape[1]):
        d = data[:, i]
        quants = quantiles(d, qlist=qlist)
        mind = quants[qlist[0]]
        maxd = quants[qlist[-1]]

        if reference is not None:
            mind = num.minimum(mind, reference)
            maxd = num.maximum(maxd, reference)

        if tstd is None:
            tstd = num.std(d)

        step = (maxd - mind) / 40.

        if bins is None:
            bins = int(num.ceil((maxd - mind) / step))

        major, minor = get_matplotlib_version()
        if major < 3:
            kwargs['normed'] = True
        else:
            kwargs['density'] = True

        n, outbins, patches = ax.hist(
            d, bins=bins, stacked=True, alpha=alpha,
            align='left', histtype=histtype, color=color, edgecolor=color,
            **kwargs)

        if cmap:
            bin_centers = 0.5 * (outbins[:-1] + outbins[1:])
            if cbounds is None:
                col = bin_centers - min(bin_centers)
                col /= max(col)
            else:
                col = (bin_centers - cbounds[0]) / (cbounds[1] - cbounds[0])

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cmap(c))

        left, right = ax.get_xlim()
        leftb = mind - tstd
        rightb = maxd + tstd

        if left != 0.0 or right != 1.0:
            leftb = num.minimum(leftb, left)
            rightb = num.maximum(rightb, right)

        ax.set_xlim(leftb, rightb)


def unify_tick_intervals(axs, varnames, ntickmarks_max=5, axis='x'):
    """
    Take figure axes objects and determine unit ranges between common
    unit classes (see utility.grouped_vars). Assures that the number of
    increments is not larger than ntickmarks_max. Will thus overwrite

    Returns
    -------
    dict : with types_sets keys and (min_range, max_range) as values
    """
    unities = {}
    for setname in utility.unit_sets.keys():
        unities[setname] = [num.inf, -num.inf]

    def extract_type_range(ax, varname, unities):
        for setname, ranges in unities.items():
            if axis == 'x':
                varrange = num.diff(ax.get_xlim())
            elif axis == 'y':
                varrange = num.diff(ax.get_ylim())
            else:
                raise ValueError('Only "x" or "y" allowed!')

            tset = utility.unit_sets[setname]
            min_range, max_range = ranges
            if varname in tset:
                new_ranges = copy.deepcopy(ranges)
                if varrange < min_range:
                    new_ranges[0] = varrange
                if varrange > max_range:
                    new_ranges[1] = varrange

                unities[setname] = new_ranges

    for ax, varname in zip(axs.ravel('F'), varnames):
        extract_type_range(ax, varname, unities)

    for setname, ranges in unities.items():
        min_range, max_range = ranges
        max_range_frac = max_range / ntickmarks_max
        if max_range_frac > min_range:
            logger.debug(
                'Range difference between min and max for %s is large!'
                ' Extending min_range to %f' % (
                setname, max_range_frac))
            unities[setname] = [max_range_frac, max_range]

    return unities


def apply_unified_axis(axs, varnames, unities, axis='x', ntickmarks_max=3,
                       scale_factor=2 / 3):
    for ax, v in zip(axs.ravel('F'), varnames):
        if v in utility.grouped_vars:
            for setname, varrange in unities.items():
                if v in utility.unit_sets[setname]:
                    inc = nice_value(varrange[0] * scale_factor)
                    autos = AutoScaler(
                        inc=inc, snap='on', approx_ticks=ntickmarks_max)
                    if axis == 'x':
                        min, max = ax.get_xlim()
                    elif axis == 'y':
                        min, max = ax.get_ylim()

                    min, max, sinc = autos.make_scale(
                        (min, max), override_mode='min-max')

                    # check physical bounds if passed truncate
                    phys_min, phys_max = physical_bounds[v]
                    if min < phys_min:
                        min = phys_min
                    if max > phys_max:
                        max = phys_max

                    if axis == 'x':
                        ax.set_xlim((min, max))
                    elif axis == 'y':
                        ax.set_ylim((min, max))

                    ticks = num.arange(min, max + inc, inc).tolist()
                    if axis == 'x':
                        ax.xaxis.set_ticks(ticks)
                    elif axis == 'y':
                        ax.yaxis.set_ticks(ticks)
        else:
            ticker = tick.MaxNLocator(nbins=3)
            if axis == 'x':
                ax.get_xaxis().set_major_locator(ticker)
            elif axis == 'y':
                ax.get_yaxis().set_major_locator(ticker)


def traceplot(trace, varnames=None, transform=lambda x: x, figsize=None,
              lines={}, chains=None, combined=False, grid=False,
              varbins=None, nbins=40, color=None, source_idxs=None,
              alpha=0.35, priors=None, prior_alpha=1, prior_style='--',
              axs=None, posterior=None, fig=None, plot_style='kde',
              prior_bounds={}, unify=True, qlist=[0.1, 99.9], kwargs={}):
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
    source_idxs : list
        array like, indexes to sources to plot marginals
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
    unify : bool
        If true axis units that belong to one group e.g. [km] will
        have common axis increments
    kwargs : dict
        for histplot op
    qlist : list
        of quantiles to plot. Default: (all, 0., 100.)

    Returns
    -------

    ax : matplotlib axes
    """
    ntickmarks = 2
    fontsize = 10
    ntickmarks_max = kwargs.pop('ntickmarks_max', 3)
    scale_factor = kwargs.pop('scale_factor', 2 / 3)

    num.set_printoptions(precision=3)

    def make_bins(data, nbins=40, qlist=None):
        d = data.flatten()
        if qlist is not None:
            qu = quantiles(d, qlist=qlist)
            mind = qu[qlist[0]]
            maxd = qu[qlist[-1]]
        else:
            mind = d.min()
            maxd = d.max()
        return num.linspace(mind, maxd, nbins)

    def remove_var(varnames, varname):
        idx = varnames.index(varname)
        varnames.pop(idx)

    if varnames is None:
        varnames = [name for name in trace.varnames if not name.endswith('_')]

    if 'geo_like' in varnames:
        remove_var(varnames, varname='geo_like')

    if 'seis_like' in varnames:
        remove_var(varnames, varname='seis_like')

    if posterior != 'None':
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
            figsize = mpl_papersize('a6', 'landscape')
        elif n < 7:
            figsize = mpl_papersize('a5', 'portrait')
        else:
            figsize = mpl_papersize('a4', 'portrait')

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

                if v in dist_vars:
                    if source_idxs is None:
                        logger.info('No patches defined using 1 every 10!')
                        source_idxs = num.arange(0, d.shape[1], 10).tolist()

                    logger.info(
                        'Plotting patches: %s' % utility.list2string(
                            source_idxs))

                    try:
                        selected = d.T[source_idxs]
                    except IndexError:
                        raise IndexError(
                            'One or several patches do not exist! '
                            'Patch idxs: %s' % utility.list2string(
                                source_idxs))
                else:
                    selected = d.T

                for isource, e in enumerate(selected):
                    e = pmp.utils.make_2d(e)
                    if make_bins_flag:
                        varbin = make_bins(e, nbins=nbins, qlist=qlist)
                        varbins.append(varbin)
                    else:
                        varbin = varbins[i]

                    if lines:
                        if v in lines:
                            reference = lines[v]
                        else:
                            reference = None
                    else:
                        reference = None

                    if color is None:
                        pcolor = mpl_graph_color(isource)
                    else:
                        pcolor = color

                    if plot_style == 'kde':
                        pmp.kdeplot(
                            e, shade=alpha, ax=axs[rowi, coli],
                            color=color, linewidth=1.,
                            kwargs_shade={'color': pcolor})
                        axs[rowi, coli].relim()
                        axs[rowi, coli].autoscale(tight=False)
                        axs[rowi, coli].set_ylim(0)
                        xax = axs[rowi, coli].get_xaxis()
                        # axs[rowi, coli].set_ylim([0, e.max()])
                        xticker = tick.MaxNLocator(nbins=5)
                        xax.set_major_locator(xticker)
                    elif plot_style == 'hist':
                        histplot_op(
                            axs[rowi, coli], e, reference=reference,
                            bins=varbin, alpha=alpha, color=pcolor, qlist=qlist,
                            kwargs=kwargs)
                    else:
                        raise NotImplementedError(
                            'Plot style "%s" not implemented' % plot_style)

                    try:
                        param = prior_bounds[v]

                        if v in dist_vars:
                            try:  # variable bounds
                                lower = param.lower[source_idxs]
                                upper = param.upper[source_idxs]
                            except IndexError:
                                lower, upper = param.lower, param.upper

                            title = '{} {}'.format(v, plot_units[hypername(v)])
                        else:
                            lower = num.array2string(
                                param.lower, separator=',')[1:-1]
                            upper = num.array2string(
                                param.upper, separator=',')[1:-1]

                            title = '{} {} priors: ({}; {})'.format(
                                v, plot_units[hypername(v)], lower, upper)
                    except KeyError:
                        try:
                            title = '{} {}'.format(v, float(lines[v]))
                        except KeyError:
                            title = '{} {}'.format(v, plot_units[hypername(v)])

                    axs[rowi, coli].set_xlabel(title, fontsize=fontsize)
                    axs[rowi, coli].grid(grid)
                    axs[rowi, coli].get_yaxis().set_visible(False)
                    format_axes(axs[rowi, coli])
                    axs[rowi, coli].tick_params(axis='x', labelsize=fontsize)
    #                axs[rowi, coli].set_ylabel("Frequency")

                    if lines:
                        try:
                            axs[rowi, coli].axvline(
                                x=lines[v], color="k", lw=1.)
                        except KeyError:
                            pass

                    if posterior != 'None':
                        if posterior == 'all':
                            for k, idx in posterior_idxs.items():
                                axs[rowi, coli].axvline(
                                    x=e[idx], color=colors[k], lw=1.)
                        else:
                            idx = posterior_idxs[posterior]
                            axs[rowi, coli].axvline(
                                x=e[idx], color=pcolor, lw=1.)

    if unify:
        unities = unify_tick_intervals(
            axs, varnames, ntickmarks_max=ntickmarks_max, axis='x')
        apply_unified_axis(axs, varnames, unities, axis='x',
                           scale_factor=scale_factor)

    if source_idxs:
        axs[0, 0].legend(source_idxs)

    fig.tight_layout()
    return fig, axs, varbins


def get_matplotlib_version():
    from matplotlib import __version__ as mplversion
    return float(mplversion[0]), float(mplversion[2:])


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
            nchains = x.shape[0] // n_steps
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
    elif sc.name == 'Metropolis' or sc.name == 'PT':
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

    stage = Stage(homepath=problem.outfolder,
                  backend=problem.config.sampler_config.backend)

    pc = problem.config.problem_config

    list_indexes = stage.handler.get_stage_indexes(po.load_stage)

    if hypers:
        sc = problem.config.hyper_sampler_config
        varnames = problem.hypernames + ['like']
    else:
        sc = problem.config.sampler_config
        varnames = problem.varnames + problem.hypernames + \
                   problem.hierarchicalnames + ['like']

    if len(po.varnames) > 0:
        varnames = po.varnames

    logger.info('Plotting variables: %s' % (', '.join((v for v in varnames))))
    figs = []

    for s in list_indexes:
        if s == 0:
            draws = 1
        elif s == -1 and not hypers and sc.name == 'Metropolis':
            draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
        else:
            draws = None

        transform = select_transform(sc=sc, n_steps=draws)

        if po.source_idxs:
            sidxs = utility.list2string(po.source_idxs, fill='_')
        else:
            sidxs = ''

        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            'stage_%i_%s_%s.%s' % (s, sidxs, po.post_llk, po.outformat))

        if not os.path.exists(outpath) or po.force:
            logger.info('plotting stage: %s' % stage.handler.stage_path(s))
            stage.load_results(
                varnames=problem.varnames,
                model=problem.model, stage_number=s,
                load='trace', chains=[-1])

            if sc.name == 'Metropolis' and po.post_llk != 'all':
                chains = select_metropolis_chains(
                    problem, stage.mtrace, po.post_llk)
                logger.info('plotting result: %s of Metropolis chain %i' % (
                    po.post_llk, chains))
            else:
                chains = None

            prior_bounds = {}
            prior_bounds.update(**pc.hyperparameters)
            prior_bounds.update(**pc.hierarchicals)
            prior_bounds.update(**pc.priors)

            fig, _, _ = traceplot(
                stage.mtrace,
                varnames=varnames,
                transform=transform,
                chains=chains,
                combined=True,
                source_idxs=po.source_idxs,
                plot_style='hist',
                lines=po.reference,
                posterior=po.post_llk,
                prior_bounds=prior_bounds)

            if not po.outformat == 'display':
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)
            else:
                figs.append(fig)

        else:
            logger.info(
                'plot for stage %s exists. Use force=True for'
                ' replotting!' % s)

    if po.outformat == 'display':
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

    assert mode == geometry_mode_str
    assert po.load_stage != 0

    hypers = utility.check_hyper_flag(problem)

    if hypers:
        sc = problem.config.hyper_sampler_config
        varnames = problem.hypernames
    else:
        sc = problem.config.sampler_config
        varnames = list(problem.varnames) + problem.hypernames + ['like']

    if len(po.varnames) > 0:
        varnames = po.varnames

    logger.info('Plotting variables: %s' % (', '.join((v for v in varnames))))

    if len(varnames) < 2:
        raise TypeError('Need at least two parameters to compare!'
                        'Found only %i variables! ' % len(varnames))

    if po.load_stage is None and not hypers and sc.name == 'Metropolis':
        draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
    else:
        draws = None

    transform = select_transform(sc=sc, n_steps=draws)

    stage = load_stage(problem, stage_number=po.load_stage, load='trace', chains=[-1])

    if sc.name == 'Metropolis' and po.post_llk != 'all':
        chains = select_metropolis_chains(problem, stage.mtrace, po.post_llk)
        logger.info('plotting result: %s of Metropolis chain %i' % (
            po.post_llk, chains))
    else:
        chains = None

    if not po.reference:
        reference = get_result_point(stage, problem.config, po.post_llk)
        llk_str = po.post_llk
    else:
        reference = po.reference
        llk_str = 'ref'

    outpath = os.path.join(
        problem.outfolder, po.figure_dir, 'corr_hist_%s_%s.%s' % (
            stage.number, llk_str, po.outformat))

    if not os.path.exists(outpath) or po.force:
        fig, axs = correlation_plot_hist(
            mtrace=stage.mtrace,
            varnames=varnames,
            transform=transform,
            cmap=plt.cm.gist_earth_r,
            chains=chains,
            point=reference,
            point_size=6,
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
            'dummy', ['white', scolor('chocolate2'), scolor('scarletred2')], N=ncolors)
        #cmap = plt.cm.gist_earth_r

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
    #truncate = len(traces) / 2
    #grid[grid > truncate] = truncate
    ax.imshow(
        grid, extent=extent, origin='lower', cmap=cmap, aspect='auto',
        alpha=alpha, zorder=zorder)


def fuzzy_rupture_fronts(
        ax, rupture_fronts, xgrid, ygrid, alpha=0.6, linewidth=7, zorder=0):
    """
    Fuzzy rupture fronts

    rupture_fronts : list
        of output of cs = pyplot.contour; cs.allsegs
    xgrid : array_like
        of center coordinates of the sub-patches of the fault in
        strike-direction in [km]
    ygrid : array_like
        of center coordinates of the sub-patches of the fault in
        dip-direction in [km]
    """

    from matplotlib.colors import LinearSegmentedColormap

    ncolors = 256
    cmap = LinearSegmentedColormap.from_list(
        'dummy', ['white', 'black'], N=ncolors)

    res_km = 25   # pixel per km

    xmin = xgrid.min()
    xmax = xgrid.max()
    ymin = ygrid.min()
    ymax = ygrid.max()
    extent = (xmin, xmax, ymin, ymax)
    grid = num.zeros(
        (int((num.abs(ymax) - num.abs(ymin)) * res_km),
         int((num.abs(xmax) - num.abs(xmin)) * res_km)),
        dtype='float64')
    for rupture_front in rupture_fronts:
        for level in rupture_front:
            for line in level:
                draw_line_on_array(
                    line[:, 0], line[:, 1],
                    grid=grid,
                    extent=extent,
                    grid_resolution=grid.shape,
                    linewidth=linewidth)

    # increase contrast reduce high intense values
    truncate = len(rupture_fronts) / 2
    grid[grid > truncate] = truncate
    ax.imshow(
        grid, extent=extent, origin='lower', cmap=cmap, aspect='auto',
        alpha=alpha, zorder=zorder)


def fault_slip_distribution(
        fault, mtrace=None, transform=lambda x: x, alpha=0.9, ntickmarks=5,
        reference=None, nensemble=1):
    """
    Draw discretized fault geometry rotated to the 2-d view of the foot-wall
    of the fault.

    Parameters
    ----------
    fault : :class:`ffi.fault.FaultGeometry`
    """

    def draw_quivers(
            ax, uperp, uparr, xgr, ygr, rake, color='black',
            draw_legend=False, normalisation=None, zorder=0):

        # positive uperp is always dip-normal- have to multiply -1
        angles = num.arctan2(-uperp, uparr) * mt.r2d + rake
        slips = num.sqrt((uperp ** 2 + uparr ** 2)).ravel()

        if normalisation is None:
            from beat.models.laplacian import distances
            centers = num.vstack((xgr, ygr)).T
            #interpatch_dists = distances(centers, centers)
            normalisation = slips.max() #/ interpatch_dists.min()

        slips /= normalisation

        slipsx = num.cos(angles * mt.d2r) * slips
        slipsy = num.sin(angles * mt.d2r) * slips

        # slip arrows of slip on patches
        quivers = ax.quiver(
            xgr.ravel(), ygr.ravel(), slipsx, slipsy,
            units='dots', angles='xy', scale_units='xy', scale=1,
            width=1., color=color, zorder=zorder)

        if draw_legend:
            quiver_legend_length = num.ceil(
                num.max(slips * normalisation) * 10.) / 10.

            #ax.quiverkey(
            #    quivers, 0.9, 0.8, quiver_legend_length,
            #    '{} [m]'.format(quiver_legend_length), labelpos='E',
            #    coordinates='figure')

        return quivers, normalisation

    def rotate_coords_plane_normal(coords, sf):

        coords -= sf.bottom_left / km

        rots = utility.get_rotation_matrix()
        rotz = coords.dot(rots['z'](mt.d2r * -sf.strike))
        roty = rotz.dot(rots['y'](mt.d2r * -sf.dip))

        roty[:, 0] *= - 1.
        return roty

    def draw_patches(ax, fault, subfault_idx, patch_values, cmap, alpha):

        lls = fault.get_subfault_patch_attributes(
            subfault_idx, attributes=['bottom_left'])
        widths, lengths = fault.get_subfault_patch_attributes(
            subfault_idx, attributes=['width', 'length'])
        sf = fault.get_subfault(subfault_idx)

        # subtract reference fault lower left and rotate
        rot_lls = rotate_coords_plane_normal(lls, sf)[:, 1::-1]

        d_patches = []
        for ll, width, length in zip(rot_lls, widths, lengths):
            d_patches.append(
                Rectangle(
                    ll, width=length, height=width, edgecolor='black'))

        lower = rot_lls.min(axis=0)
        pad = sf.length / km * 0.05

        xlim = [lower[0] - pad, lower[0] + sf.length / km + pad]
        ylim = [lower[1]- pad, lower[1] + sf.width / km + pad]

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        scale_y = {'scale': 1, 'offset': (-sf.width / km)}
        scale_axes(ax.yaxis, **scale_y)

        ax.set_xlabel('strike-direction [km]', fontsize=fontsize)
        ax.set_ylabel('dip-direction [km]', fontsize=fontsize)

        xticker = tick.MaxNLocator(nbins=ntickmarks)
        yticker = tick.MaxNLocator(nbins=ntickmarks)

        ax.get_xaxis().set_major_locator(xticker)
        ax.get_yaxis().set_major_locator(yticker)

        pa_col = PatchCollection(
            d_patches, alpha=alpha, match_original=True, zorder=0)
        pa_col.set(array=patch_values, cmap=cmap)

        ax.add_collection(pa_col)
        return pa_col

    def draw_colorbar(fig, ax, cb_related, labeltext, ntickmarks=4):
        cbaxes = fig.add_axes([0.88, 0.4, 0.03, 0.3])
        cb = fig.colorbar(cb_related, ax=axs, cax=cbaxes)
        cb.set_label(labeltext, fontsize=fontsize)
        cb.locator = tick.MaxNLocator(nbins=ntickmarks)
        cb.update_ticks()
        ax.set_aspect('equal', adjustable='box')

    def get_values_from_trace(mtrace, varname, reference):
        try:
            u = transform(
                mtrace.get_values(
                    varname, combine=True, squeeze=True))
        except(ValueError, KeyError):
            u = num.atleast_2d(reference[varname])
        return u

    from beat.colormap import slip_colormap
    fontsize = 12

    reference_slip = num.sqrt(
        reference['uperp'] ** 2 + reference['uparr'] ** 2)

    figs = []
    axs = []
    for ns in range(fault.nsubfaults):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize('a5', 'landscape'))

        # alphas = alpha * num.ones(np_h * np_w, dtype='int8')

        ext_source = fault.get_subfault(ns, component='uparr')
        patch_idxs = fault.get_patch_indexes(ns)

        pa_col = draw_patches(
            ax, fault, subfault_idx=ns, patch_values=reference_slip[patch_idxs],
            cmap=slip_colormap(100), alpha=0.65)

        # patch central locations
        centers = fault.get_subfault_patch_attributes(ns, attributes=['center'])
        rot_centers = rotate_coords_plane_normal(centers, ext_source)[:, 1::-1]

        xgr, ygr = rot_centers.T
        if 'seismic' in fault.datatypes:
            shp = fault.ordering.get_subfault_discretization(ns)
            xgr = xgr.reshape(shp)
            ygr = ygr.reshape(shp)

            if mtrace is not None:
                from tqdm import tqdm
                nuc_dip = transform(mtrace.get_values(
                    'nucleation_dip', combine=True, squeeze=True))
                nuc_strike = transform(mtrace.get_values(
                    'nucleation_strike', combine=True, squeeze=True))
                velocities = transform(mtrace.get_values(
                    'velocities', combine=True, squeeze=True))
                nchains = len(mtrace)
                csteps = 6
                rupture_fronts = []
                dummy_fig, dummy_ax = plt.subplots(
                    nrows=1, ncols=1, figsize=mpl_papersize('a5', 'landscape'))
                csteps = float(nchains) / nensemble
                idxs = num.floor(
                    num.arange(0, nchains, csteps)).astype('int32')
                logger.info('Rendering rupture fronts ...')
                for i in tqdm(idxs):
                    nuc_dip_idx, nuc_strike_idx = fault.fault_locations2idxs(
                        ns, nuc_dip[i], nuc_strike[i], backend='numpy')
                    veloc_ns = fault.vector2subfault(
                        index=ns, vector=velocities[i, :])
                    sts = fault.get_subfault_starttimes(
                        ns, veloc_ns, nuc_dip_idx[ns], nuc_strike_idx[ns])
                    
                    contours = dummy_ax.contour(xgr, ygr, sts)
                    rupture_fronts.append(contours.allsegs)

                fuzzy_rupture_fronts(
                    ax, rupture_fronts, xgr, ygr,
                    alpha=1., linewidth=7, zorder=-1)

                durations = transform(mtrace.get_values(
                    'durations', combine=True, squeeze=True))
                std_durations = durations.std(axis=0)
                # alphas = std_durations.min() / std_durations

            # rupture durations
            fig2, ax2 = plt.subplots(
                nrows=1, ncols=1, figsize=mpl_papersize('a5', 'landscape'))

            reference_durations = reference['durations'][patch_idxs]

            pa_col2 = draw_patches(
                ax2, fault, subfault_idx=ns, patch_values=reference_durations,
                cmap=plt.cm.seismic, alpha=alpha)

            draw_colorbar(fig2, ax2, pa_col2, labeltext='durations [s]')
            figs.append(fig2)
            axs.append(ax2)

            ref_starttimes = fault.point2starttimes(reference, index=ns)
            contours = ax.contour(
                xgr, ygr, ref_starttimes,
                colors='black', linewidths=0.5, alpha=0.9)
            ax.plot(
                reference['nucleation_strike'][ns],
                ext_source.width / km - reference['nucleation_dip'][ns],
                marker='*', color='k', markersize=12)
            plt.clabel(contours, inline=True, fontsize=10)

        if mtrace is not None:
            logger.info('Drawing quantiles ...')

            uparr = get_values_from_trace(
                mtrace, 'uparr', reference)[:, patch_idxs]
            uperp = get_values_from_trace(
                mtrace, 'uperp', reference)[:, patch_idxs]

            uparrmean = uparr.mean(axis=0)
            uperpmean = uperp.mean(axis=0)

            quivers, normalisation = draw_quivers(
                ax, uperpmean, uparrmean, xgr, ygr,
                ext_source.rake, color='grey',
                draw_legend=False)

            uparrstd = uparr.std(axis=0) / normalisation
            uperpstd = uperp.std(axis=0) / normalisation

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

                xcoords = xgr.ravel()[i] + rot_ellipse[:, 0] + quivers.U[i]
                ycoords = ygr.ravel()[i] + rot_ellipse[:, 1] + quivers.V[i]
                ax.plot(xcoords, ycoords, '-k', linewidth=0.5, zorder=2)
        else:
            normalisation = None

        uperp = reference['uperp'][patch_idxs]
        uparr = reference['uparr'][patch_idxs]

        logger.info('Drawing slip vectors ...')
        draw_quivers(
            ax, uperp, uparr,
            xgr, ygr, ext_source.rake, color='black', draw_legend=True,
            normalisation=normalisation, zorder=3)

        draw_colorbar(fig, ax, pa_col, labeltext='slip [m]')
        format_axes(ax, remove=['top', 'right'])

        fig.tight_layout()
        figs.append(fig)
        axs.append(ax)

    return figs, axs


class ModeError(Exception):
    pass


def draw_slip_dist(problem, po):

    mode = problem.config.problem_config.mode

    if mode != ffi_mode_str:
        raise ModeError(
            'Wrong optimization mode: %s! This plot '
            'variant is only valid for "%s" mode' % (mode, ffi_mode_str))

    datatype, gc = list(problem.composites.items())[0]

    fault = gc.load_fault_geometry()

    sc = problem.config.sampler_config
    if po.load_stage is None and sc.name == 'Metropolis':
        draws = sc.parameters.n_steps * (sc.parameters.n_stages - 1) + 1
    else:
        draws = None

    transform = select_transform(sc=sc, n_steps=draws)

    stage = load_stage(problem, stage_number=po.load_stage, load='trace', chains=[-1])

    if not po.reference:
        reference = problem.config.problem_config.get_test_point()
        res_point = get_result_point(stage, problem.config, po.post_llk)
        reference.update(res_point)
        llk_str = po.post_llk
        mtrace = stage.mtrace
    else:
        reference = po.reference
        llk_str = 'ref'
        mtrace = None

    figs, axs = fault_slip_distribution(
        fault, mtrace, transform=transform,
        reference=reference, nensemble=po.nensemble)

    if po.outformat == 'display':
        plt.show()
    else:
        outpath = os.path.join(
            problem.outfolder, po.figure_dir,
            'slip_dist_%i_%s_%i' % (stage.number, llk_str, po.nensemble))

        logger.info('Storing slip-distribution to: %s' % outpath)
        if po.outformat == 'pdf':
            with PdfPages(outpath + '.pdf') as opdf:
                for fig in figs:
                    opdf.savefig(fig, dpi=po.dpi)
        else:
            for i, fig in enumerate(figs):
                fig.savefig(outpath + '_%i.%s' % (i, po.outformat), dpi=po.dpi)


def _weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=num.inf, cmin=0, cmax=num.inf):
    """
    Draw weighted lines into array
    Modiefied from:
    https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays

    Parameters
    ----------
    r0 : int
        row index for line end point 0
    c0 : int
        col index for line end point 0
    r1 : int
        row index for line end point 1
    c1 : int
        col index for line end point 1
    w : int
        width in pixels for line
    rmin : int
        min row index for grid to draw on
    rmax : int
        max row index for grid to draw on

    Returns
    -------
    rr : array of row indexes of line
    cc : array of col indexes of line
    w : array of line weights
    """
    def trapez(y, y0, w):
        return num.clip(num.minimum(
            y + 1 + w / 2 - y0,
            - y + 1 + w / 2 + y0), 0, 1)
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = _weighted_line(
            c0, r0, c1, r1, w=w, rmin=cmin, rmax=cmax, cmin=rmin, cmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return _weighted_line(
            r1, c1, r0, c0, w=w, rmin=rmin, rmax=rmax, cmin=cmin, cmax=cmax)

    # The following is now always < 1 in abs
    slope = (r1 - r0) / (c1 - c0)

    # Adjust weight by the slope
    w *= num.sqrt(1 + num.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = num.arange(c0, c1 + 1, dtype=float)
    y = (x * slope) + ((c1 * r0) - (c0 * r1)) / (c1 - c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = num.ceil(w / 2)

    yy = (num.floor(y).reshape(-1, 1) +
          num.arange(-thickness - 1, thickness + 2).reshape(1, -1))
    xx = num.repeat(x, yy.shape[1])

    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask_y = num.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))
    mask_x = num.logical_and.reduce((xx >= cmin, xx < cmax, vals > 0))
    mask = num.logical_and.reduce((mask_y > 0, mask_x > 0))
    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def draw_line_on_array(
        X, Y, grid=None, extent=[], grid_resolution=(400, 400), linewidth=1):
    """
    Draw line on given array by adding 1 to its fields.

    Parameters
    ----------
    X : array_like
        timeseries on xcoordinate (columns of array)
    Y : array_like
        timeseries on ycoordinate (rows of array)
    grid : array_like 2d
        input array that is used for drawing
    extent : array extent
        [xmin, xmax, ymin, ymax] (cols, rows)
    grid_resolution : tuple
        shape of given grid or grid that is being used for allocation
    linewidth : int
        weight (width) of line drawn on grid

    Returns
    -------
    grid, extent
    """

    def check_grid_shape(ngr, naim, axis):
        if ngr != naim:
            raise TypeError(
                'Gridsize of given grid is inconistent for axis %i!'
                ' Expected %i got %i' % (axis, naim, ngr))

    def check_line_in_grid(idxs, axis, nmax, extent):
        imax = idxs.max()
        if imax > nmax:
            raise TypeError(
                'Line endpoint outside of given grid Axis "%s"! %i > %i'
                ' Extent [%s]' % (
                    axis, imax, nmax, utility.list2string(extent)))

    nxs = len(X)
    nys = len(Y)
    if nxs != nys:
        raise TypeError(
            'Length of X and Y have to be identical! %i != %i' % (nxs, nys))

    if len(extent) == 0:
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        extent = [xmin, xmax, ymin, ymax]
    elif len(extent) == 4:
        xmin, xmax, ymin, ymax = extent
    else:
        raise TypeError(
            'extent has to be of length 4! [xmin, xmax, ymin, ymax]')

    if len(grid_resolution) != 2:
        raise TypeError(
            'grid_resolution has to be of length 2! [xstep, ystep]!')

    ynstep, xnstep = grid_resolution

    xvec, xstep = num.linspace(xmin, xmax, xnstep, endpoint=True, retstep=True)
    yvec, ystep = num.linspace(ymin, ymax, ynstep, endpoint=True, retstep=True)

    if grid is not None:
        if grid.ndim != 2:
            raise TypeError('Given grid has to be of dimension 2!')

        for axis, (ngr, naim) in enumerate(
                zip(grid.shape, grid_resolution)):
            check_grid_shape(ngr, naim, axis)
    else:
        grid = num.zeros((ynstep, xnstep), dtype='float64')

    xidxs = utility.positions2idxs(
        X, min_pos=xmin, cell_size=xstep, dtype='int32')
    yidxs = utility.positions2idxs(
        Y, min_pos=ymin, cell_size=ystep, dtype='int32')

    check_line_in_grid(xidxs, 'x', nmax=xnstep - 1, extent=extent)
    check_line_in_grid(yidxs, 'y', nmax=ynstep - 1, extent=extent)
    new_grid = num.zeros_like(grid)
    for i in range(1, nxs):
        c0 = xidxs[i - 1]
        r0 = yidxs[i - 1]
        c1 = xidxs[i]
        r1 = yidxs[i]
        try:
            rr, cc, w = _weighted_line(
                r0=r0, c0=c0, r1=r1, c1=c1, w=linewidth, rmax=ynstep - 1, cmax=xnstep - 1)
            new_grid[rr, cc] = w.astype(grid.dtype)
        except ValueError:
            # line start and end fall in the same grid point cant be drawn
            pass

    grid += new_grid
    return grid, extent


def fuzzy_moment_rate(
        ax, moment_rates, times, cmap=None, grid_size=(500, 500)):
    """
    Plot fuzzy moment rate function into axes.
    """

    if cmap is None:
        # from matplotlib.colors import LinearSegmentedColormap
        # ncolors = 256
        # cmap = LinearSegmentedColormap.from_list(
        #    'dummy', [background_color, rates_color], N=ncolors)
        cmap = plt.cm.hot_r

    nrates = len(moment_rates)
    ntimes = len(times)

    if nrates != ntimes:
        raise TypeError(
            'Number of rates and times have to be identical!'
            ' %i != %i' % (nrates, ntimes))

    max_rates = max(map(num.max, moment_rates))
    max_times = max(map(num.max, times))
    min_rates = min(map(num.min, moment_rates))
    min_times = min(map(num.min, times))

    extent = (min_times, max_times, min_rates, max_rates)
    grid = num.zeros(grid_size, dtype='float64')

    for mr, time in zip(moment_rates, times):
        draw_line_on_array(
            time, mr,
            grid=grid,
            extent=extent,
            grid_resolution=grid.shape,
            linewidth=7)

    # increase contrast reduce high intense values
    truncate = nrates / 2
    grid[grid > truncate] = truncate

    ax.imshow(grid, extent=extent, origin='lower', cmap=cmap, aspect='auto')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Moment rate [$Nm / s$]')


def draw_moment_rate(problem, po):
    """
    Draw moment rate function for the results of a seismic/joint finite fault
    optimization.
    """
    fontsize = 12
    mode = problem.config.problem_config.mode

    if mode != ffi_mode_str:
        raise ModeError(
            'Wrong optimization mode: %s! This plot '
            'variant is only valid for "%s" mode' % (mode, ffi_mode_str))

    if 'seismic' not in problem.config.problem_config.datatypes:
        raise TypeError(
            'Moment rate function only available for optimization results that'
            ' include seismic data.')

    sc = problem.composites['seismic']
    fault = sc.load_fault_geometry()

    stage = load_stage(
        problem, stage_number=po.load_stage, load='trace', chains=[-1])

    if not po.reference:
        reference = get_result_point(stage, problem.config, po.post_llk)
        llk_str = po.post_llk
        mtrace = stage.mtrace
    else:
        reference = po.reference
        llk_str = 'ref'
        mtrace = None

    logger.info(
        'Drawing ensemble of %i moment rate functions ...' % po.nensemble)
    target = sc.wavemaps[0].targets[0]

    if po.plot_projection == 'individual':
        logger.info('Drawing subfault individual rates ...')
        sf_idxs = range(fault.nsubfaults)
    else:
        logger.info('Drawing total rates ...')
        sf_idxs = [list(range(fault.nsubfaults))]

    mpl_init(fontsize=fontsize)
    for i, ns in enumerate(sf_idxs):
        logger.info('Fault %i / %i' % (i + 1, len(sf_idxs)))
        if isinstance(ns, list):
            ns_str = 'total'
        else:
            ns_str = str(ns)

        outpath = os.path.join(
            problem.outfolder, po.figure_dir,
            'moment_rate_%i_%s_%s_%i.%s' % (
                stage.number, ns_str,
                llk_str, po.nensemble, po.outformat))

        ref_mrf_rates, ref_mrf_times = fault.get_moment_rate_function(
            index=ns, point=reference, target=target,
            store=sc.engine.get_store(target.store_id))

        if not os.path.exists(outpath) or po.force:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=mpl_papersize('a7', 'landscape'))
            labelpos = mpl_margins(
                fig, left=5, bottom=4, top=1.5, right=0.5, units=fontsize)
            labelpos(ax, 2., 1.5)
            if mtrace is not None:
                nchains = len(mtrace)
                csteps = float(nchains) / po.nensemble
                idxs = num.floor(
                    num.arange(0, nchains, csteps)).astype('int32')
                mrfs_rate = []
                mrfs_time = []
                for idx in idxs:
                    point = mtrace.point(idx=idx)
                    mrf_rate, mrf_time = \
                        fault.get_moment_rate_function(
                            index=ns, point=point, target=target,
                            store=sc.engine.get_store(target.store_id))
                    mrfs_rate.append(mrf_rate)
                    mrfs_time.append(mrf_time)

                fuzzy_moment_rate(ax, mrfs_rate, mrfs_time)

            ax.plot(
                ref_mrf_times, ref_mrf_rates,
                '-k', alpha=0.8, linewidth=1.)
            format_axes(ax, remove=['top', 'right'])

            if po.outformat == 'display':
                plt.show()
            else:
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)

        else:
            logger.info('Plot exists! Use --force to overwrite!')


def source_geometry(fault, ref_sources, event, datasets=None, values=None,
                    cmap=None, title=None, show=True):
    """
    Plot source geometry in 3d rotatable view

    Parameters
    ----------
    fault: :class:`beat.ffi.fault.FaultGeometry`
    ref_sources: list
        of :class:'beat.sources.RectangularSource'
    """

    from mpl_toolkits.mplot3d import Axes3D
    from beat.utility import RS_center
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    alpha = 0.7

    def plot_subfault(ax, source, color, refloc):
        source.anchor = 'top'
        shift_ne = otd.latlon_to_ne(
            refloc.lat, refloc.lon, source.lat, source.lon)
        coords = source.outline()
        coords[:, 0:2] += shift_ne
        ax.plot(
            coords[:, 1], coords[:, 0], coords[:, 2] * -1.,
            color=color, linewidth=2, alpha=alpha)
        ax.plot(
            coords[0:2, 1], coords[0:2, 0], coords[0:2, 2] * -1.,
            '-k', linewidth=2, alpha=alpha)
        center = RS_center(source)
        center[0:2] += shift_ne
        ax.scatter(
            center[1], center[0], center[2] * -1,
            marker='o', s=20, color=color, alpha=alpha)

    def set_axes_radius(ax, origin, radius, axes=['xyz']):
        if 'x' in axes:
            ax.set_xlim3d([origin[0] - radius, origin[0] + radius])

        if 'y' in axes:
            ax.set_ylim3d([origin[1] - radius, origin[1] + radius])

        if 'z' in axes:
            ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


    def set_axes_equal(ax, axes='xyz'):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        limits = num.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = num.mean(limits, axis=1)
        radius = 0.5 * num.max(num.abs(limits[:, 1] - limits[:, 0]))
        set_axes_radius(ax, origin, radius, axes=axes)


    fig = plt.figure(figsize=mpl_papersize('a4', 'landscape'))
    ax = fig.add_subplot(111, projection='3d')
    extfs = fault.get_all_subfaults()

    arr_coords = []
    for idx, (refs, exts) in enumerate(zip(ref_sources, extfs)):

        plot_subfault(ax, exts, color=mpl_graph_color(idx), refloc=event)
        plot_subfault(ax, refs, color=scolor('aluminium4'), refloc=event)
        for i, patch in enumerate(fault.get_subfault_patches(idx)):
            coords = patch.outline()
            shift_ne = otd.latlon_to_ne(
                event.lat, event.lon, patch.lat, patch.lon)
            coords[:, 0:2] += shift_ne
            coords[:, 2] *= -1.
            coords[:, [0, 1]] = coords[:, [1, 0]]  # swap columns to [E, N, Z] (X, Y, Z)
            arr_coords.append(coords)
            ax.plot(
                coords[:, 0], coords[:, 1], coords[:, 2], zorder=2,
                color=mpl_graph_color(idx), linewidth=0.5, alpha=alpha)
            ax.text(
                patch.east_shift + shift_ne[1],
                patch.north_shift + shift_ne[0], patch.center[2] * -1.,
                str(i + fault.cum_subfault_npatches[idx]), zorder=3,
                fontsize=10)

    if values is not None:

        if cmap is None:
            cmap = plt.cm.get_cmap('jet')

        poly_patches = Poly3DCollection(
            verts=arr_coords, zorder=1, cmap=cmap)
        poly_patches.set_array(values)
        poly_patches.set_clim(values.min(), values.max())
        poly_patches.set_alpha(0.6)
        poly_patches.set_edgecolor('k')
        ax.add_collection(poly_patches)
        plt.colorbar(poly_patches)

    if datasets:
        for dataset in datasets:
            #print(dataset.east_shifts, dataset.north_shifts)
            ax.scatter(
                dataset.east_shifts, dataset.north_shifts, dataset.coords5[:, 4],
                s=10, alpha=0.6, marker='o', color='black')

    scale = {'scale': 1. / km}
    scale_axes(ax.xaxis, **scale)
    scale_axes(ax.yaxis, **scale)
    scale_axes(ax.zaxis, **scale)
    ax.set_zlabel('Depth [km]')
    ax.set_ylabel('North_shift [km]')
    ax.set_xlabel('East_shift [km]')
    set_axes_equal(ax, axes='xy')
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()


def get_gmt_config(gmtpy, h=20., w=20.):

    if gmtpy.is_gmt5(version='newest'):
        gmtconfig = {
            'MAP_GRID_PEN_PRIMARY': '0.1p',
            'MAP_GRID_PEN_SECONDARY': '0.1p',
            'MAP_FRAME_TYPE': 'fancy',
            'FONT_ANNOT_PRIMARY': '14p,Helvetica,black',
            'FONT_ANNOT_SECONDARY': '14p,Helvetica,black',
            'FONT_LABEL': '14p,Helvetica,black',
            'FORMAT_GEO_MAP': 'D',
            'GMT_TRIANGULATE': 'Watson',
            'PS_MEDIA': 'Custom_%ix%i' % (w * gmtpy.cm, h * gmtpy.cm),
        }
    else:
        gmtconfig = {
            'MAP_FRAME_TYPE': 'fancy',
            'GRID_PEN_PRIMARY': '0.01p',
            'ANNOT_FONT_PRIMARY': '1',
            'ANNOT_FONT_SIZE_PRIMARY': '12p',
            'PLOT_DEGREE_FORMAT': 'D',
            'GRID_PEN_SECONDARY': '0.01p',
            'FONT_LABEL': '14p,Helvetica,black',
            'PS_MEDIA': 'Custom_%ix%i' % (w * gmtpy.cm, h * gmtpy.cm),
        }
    return gmtconfig


def draw_station_map_gmt(problem, po):

    from pyrocko import gmtpy

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

    font_size = 12
    font = '1'
    bin_width = 15  # major grid and tick increment in [deg]
    h = 20  # outsize in cm
    w = h - 5

    gmtconfig = get_gmt_config(gmtpy, h=h, w=h)

    def draw_time_shifts_stations(gmt, point, wmap, *args):
        """
        Draw MAP time-shifts at station locations as colored triangles
        """
        time_shifts = extract_time_shifts(point, wmap)

        cptfilepath = '/tmp/tempfile.cpt'
        miny = time_shifts.min()
        maxy = time_shifts.max()
        bound = num.ceil(max(num.abs(miny), maxy))

        gmt.makecpt(
            C='blue,white,red',
            Z=True,
            T='%g/%g' % (-bound, bound),
            out_filename=cptfilepath, suppress_defaults=True)

        for i, station in enumerate(wmap.stations):
            logger.debug('%s, %f' % (station.station, time_shifts[i]))

        gmt.psxy(
            in_columns=(st_lons, st_lats, time_shifts.tolist()),
            C=cptfilepath,
            S='t14p',
            *args)

        # add a colorbar
        gmt.psscale(
            B='xa%s +l time shifts [s]' % num.floor(bound),
            D='x1.5c/0c+w6c/0.5c+jMC+h',
            C=cptfilepath)

    def draw_events(gmt, events, *args):

        ev_lons = [ev.lon for ev in events]
        ev_lats = [ev.lat for ev in events]

        gmt.psxy(
            in_columns=(ev_lons, ev_lats),
            G='yellow',
            S='a14p',
            *args)

    logger.info('Drawing Station Map ...')
    sc = problem.composites['seismic']
    event = problem.config.event

    for wmap in sc.wavemaps:
        outpath = os.path.join(
            problem.outfolder, po.figure_dir, 'station_map_%s_%i_%s.%s' % (
                wmap.name, wmap.mapnumber, value_string, po.outformat))

        dist = max(wmap.config.distances)
        if not os.path.exists(outpath) or po.force:

            st_lons = [station.lon for station in wmap.stations]
            st_lats = [station.lat for station in wmap.stations]

            if dist > 30:
                dist = dist * 1.05 # add interval to have bound
                logger.info(
                    'Using equidistant azimuthal projection for'
                    ' teleseismic setup of wavemap %s.' % wmap._mapid)
                J_basemap = 'E0/-90/%s/%i' % (dist, w)
                J_location = 'E%s/%s/%s/%i' % (event.lon, event.lat, dist, w)
                R_location = '0/360/-90/0'

                gmt = gmtpy.GMT(config=gmtconfig)
                gmt.psbasemap(
                    R=R_location,
                    J='S0/-90/90/%i' % w,
                    B='xa%sf%s' % (bin_width*2, bin_width))
                gmt.pscoast(
                    R='g',
                    J='E%s/%s/%s/%i' % (event.lon, event.lat, dist, w),
                    D='c',
                    G='darkgrey')
                gmt.psbasemap(
                    R='g',
                    J=J_basemap,
                    B='xg%s' % bin_width)
                gmt.psbasemap(
                    R='g',
                    J=J_basemap,
                    B='yg%s' % (2 * bin_width))

                draw_events(
                    gmt, [event], *('-J%s' % J_location, '-R%s' % R_location))

                if point:
                    draw_time_shifts_stations(
                        gmt, point, wmap, *(
                            '-J%s' % J_location, '-R%s' % R_location))
                else:
                    gmt.psxy(
                        R=R_location,
                        J=J_location,
                        in_columns=(st_lons, st_lats),
                        G='red',
                        S='t14p')

                rows = []
                alignment = 'TC'
                for st in wmap.stations:
                    if gmt.is_gmt5():
                        row = (
                            st.lon, st.lat,
                            '%i,%s,%s' % (font_size, font, 'black'),
                            alignment,
                            '{}.{}'.format(st.network, st.station))
                        farg = ['-F+f+j']
                    else:
                        row = (
                            st.lon, st.lat,
                            font_size, 0, font, alignment,
                            '{}.{}'.format(st.network, st.station))
                        farg = []

                    rows.append(row)

                gmt.pstext(
                    in_rows=rows,
                    R=R_location,
                    J=J_location,
                    N=True, *farg)

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

                if point:
                    draw_time_shifts_stations(
                        m.gmt, point, wmap, *m.jxyr)

                    for st in wmap.stations:
                        text = '{}.{}'.format(st.network, st.station)
                        m.add_label(lat=st.lat, lon=st.lon, text=text)
                else:
                    m.add_stations(
                        wmap.stations, psxy_style=dict(S='t14p', G='red'))

                draw_events(m.gmt, [event], *m.jxyr)
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

    result_ensemble = {}
    for varname in ['v', 'w']:
        try:
            result_ensemble[varname] = stage.mtrace.get_values(
                varname, combine=True, squeeze=True).ravel()
        except ValueError:  # if fixed value add that to the ensemble
            n_mts = len(stage.mtrace)
            rpoint = problem.get_random_point()
            result_ensemble[varname] = num.full_like(
                num.empty((n_mts), dtype=num.float64), rpoint[varname])

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        'lune_%i_%s_%i.%s' % (
            po.load_stage, po.post_llk, po.nensemble, po.outformat))

    if not os.path.exists(outpath) or po.force or po.outformat == 'display':
        logger.info('Drawing Lune plot ...')
        gmt = lune_plot(            # this explodes for large sample sizes ...
            v_tape=result_ensemble['v'][::3],
            w_tape=result_ensemble['w'][::3])

        logger.info('saving figure to %s' % outpath)
        gmt.save(outpath, resolution=300, size=10)
    else:
        logger.info('Plot exists! Use --force to overwrite!')


def lune_plot(v_tape=None, w_tape=None):

    from pyrocko import gmtpy

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
            if a.std() == 0:
                logger.info(
                    'Variable "%s" got fixed to %f, adding some jitter to'
                    ' make kde estimate possible' % (varname, a[0]))
                a += num.random.normal(loc=0., scale=0.25, size=a.size)

        from beat.sources import v_to_gamma, w_to_delta

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
        gmt, v_tape=v_tape, w_tape=w_tape, grid_size=(300, 300), R=R, J=J)
    gmt.psbasemap(R=R, J=J, B=B)
    draw_lune_arcs(gmt, R=R, J=J)
    draw_lune_points(gmt, R=R, J=J)
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


plots_catalog = {
    'correlation_hist': draw_correlation_hist,
    'stage_posteriors': draw_posteriors,
    'waveform_fits': draw_seismic_fits,
    'scene_fits': draw_scene_fits,
    'gnss_fits': draw_gnss_fits,
    'velocity_models': draw_earthmodels,
    'slip_distribution': draw_slip_dist,
    'hudson': draw_hudson,
    'lune': draw_lune_plot,
    'fuzzy_beachball': draw_fuzzy_beachball,
    'fuzzy_mt_decomp': draw_fuzzy_mt_decomposition,
    'moment_rate': draw_moment_rate,
    'station_map': draw_station_map_gmt}


common_plots = [
    'stage_posteriors',
    'velocity_models']


seismic_plots = [
    'station_map',
    'waveform_fits']


geodetic_plots = [
    'scene_fits',
    'gnss_fits']


geometry_plots = [
    'correlation_hist',
    'hudson',
    'lune',
    'fuzzy_beachball']


ffi_plots = [
    'moment_rate',
    'slip_distribution']


plots_mode_catalog = {
    'geometry': common_plots + geometry_plots,
    'ffi': common_plots + ffi_plots,
}

plots_datatype_catalog = {
    'seismic': seismic_plots,
    'geodetic': geodetic_plots,
}


def available_plots(mode=None, datatypes=['geodetic', 'seismic']):
    if mode is None:
        return list(plots_catalog.keys())
    else:
        plots = plots_mode_catalog[mode]
        for datatype in datatypes:
            plots.extend(plots_datatype_catalog[datatype])

        return plots
