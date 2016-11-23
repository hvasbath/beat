from pyrocko import cake_plot as cp
from pymc3 import plots as pmp

import math
import os
import logging

from beat import utility, backend
from beat.models import load_stage

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as num
from pyrocko import cake, util
from pyrocko.cake_plot import str_to_mpl_color as scolor

logger = logging.getLogger('plotting')


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
        figsize = (11.7, 8.2)   # A4 landscape

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
        grid=200, point=None,
        point_style='.', point_color='red', point_size='8', alpha=0.35):
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
        figsize = (11.7, 8.2)   # A4 landscape

    fig, axs = plt.subplots(nrows=nvar, ncols=nvar, figsize=figsize,
            subplot_kw={'adjustable': 'box-forced'})

    d = dict()
    for var in varnames:
        d[var] = transform(mtrace.get_values(
                var, combine=True, squeeze=True))

    for k in range(nvar):
        a = d[varnames[k]]

        for l in range(k, nvar):
            logger.debug('%s, %s' % (varnames[k], varnames[l]))
            if l == k:
                histplot_op(
                    axs[l, k], pmp.make_2d(a), alpha=alpha, color='orange')
                axs[l, k].set_xbound(a.min(), a.max())
                axs[l, k].get_yaxis().set_visible(False)

                if point is not None:
                    axs[l, k].axvline(
                        x=point[varnames[k]], color=point_color,
                        lw=int(point_size) / 4.)
            else:
                b = d[varnames[l]]

                pmp.kde2plot(
                    a, b, grid=grid, ax=axs[l, k], cmap=cmap, aspect='auto')

                if point is not None:
                    axs[l, k].plot(point[varnames[k]], point[varnames[l]],
                        color=point_color, marker=point_style,
                        markersize=point_size)

            if l != nvar - 1:
                axs[l, k].get_xaxis().set_ticklabels([])

            if k == 0:
                axs[l, k].set_ylabel(varnames[l])
            else:
                axs[l, k].get_yaxis().set_ticklabels([])

            axs[l, k].tick_params(direction='in')

        axs[l, k].set_xlabel(varnames[k])

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


def plot_matrix(A):
    '''
    Very simple plot of a matrix for fast inspections.
    '''
    ax = plt.axes()
    im = ax.matshow(A)
    plt.colorbar(im)
    plt.show()


def get_fit_indexes(llk):
    """
    Find indexes of various likelihoods in a likelihood distribution.
    """

    mean_idx = (num.abs(llk - llk.mean())).argmin()
    min_idx = (num.abs(llk - llk.min())).argmin()
    max_idx = (num.abs(llk - llk.max())).argmin()

    posterior_idxs = {
        'mean': mean_idx,
        'min': min_idx,
        'max': max_idx}

    return posterior_idxs


def draw_geodetic_misfit_figures(problem, format='pdf', force=False, dpi=450):

    mode = problem.config.problem_config.mode

    stage = load_stage(problem, load='full')

    figure_path = os.path.join(problem.outfolder, 'figures')
    util.ensuredir(figure_path)


    population, _, llk = stage.step.select_end_points(mtrace)

    posterior_idxs = get_fit_indexes(llk)

    out_point = population[idx]

    d = problem.get_synthetics(out_point)

    dd = d[dataset]

    outfigure = os.path.join(figure_path, 'misfits_%s.pdf' % posterior)
    pdfp = PdfPages(outfigure)

    for i, gt in enumerate(problem.gtargets):
        f, axarr = plt.subplots(nrows=1, ncols=3, sharey=True)
        colim = num.max([num.abs(gt.displacement), num.abs(dd[i])])

        im = axarr[0].scatter(
            gt.lons, gt.lats, 20, gt.displacement,
            edgecolors='none', vmin=-colim, vmax=colim)
        plt.title(gt.track)

        im = axarr[1].scatter(
            gt.lons, gt.lats, 20, dd[i],
            edgecolors='none', vmin=-colim, vmax=colim)
        f.colorbar(im, ax=axarr[1])

        im = axarr[2].scatter(
            gt.lons, gt.lats, 20, gt.displacement - dd[i],
            edgecolors='none', vmin=-colim, vmax=colim)
        f.colorbar(im, ax=axarr[2])

        plt.autoscale(enable=True, axis='both', tight=True)
        plt.setp([a.get_xticklabels() for a in axarr], visible=False)
        pdfp.savefig()

    pdfp.close()


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
        (ma-mi) * space - (1.0 + space)
    t2 = num.concatenate((t, t[::-1]))
    axes.fill(
        t2, y2,
        clip_on=False,
        **kwargs)


def plot_dtrace_vline(axes, t, space, **kwargs):
    axes.plot([t, t], [-1.0 - space, -1.0], **kwargs)


def draw_seismic_fits_figures(problem, format='pdf', force=False, dpi=450):
    '''
    Modified from grond plot.
    '''
    fontsize = 8
    fontsize_title = 10

    target_index = dict(
        (target, i) for (i, target) in enumerate(problem.stargets))

    mode = problem.config.problem_config.mode

    po = problem.get_plot_options()
    figure_dir = po['figure_dir']

    stage = load_stage(problem, load='full')

    figure_path = os.path.join(problem.outfolder, figure_dir)
    util.ensuredir(figure_path)

    population, _, llk = stage.step.select_end_points(stage.mtrace)

    posterior_idxs = get_fit_indexes(llk)
    idx = posterior_idxs[problem.plot_options]

    out_point = population[idx]
    # have to get best result based on llk
    target_to_result = {}
    all_syn_trs = []

    dtraces = []
    for target, result in zip(problem.targets, results):
        if isinstance(result, gf.SeismosizerError):
            dtraces.append(None)
            continue

        itarget = target_index[target]
        w = target.get_combined_weight(problem.apply_balancing_weights)

            dtrace = result.processed_syn.copy()
            dtrace.set_ydata(
                (
                    (result.processed_syn.get_ydata() -
                     result.processed_obs.get_ydata())**2))

        target_to_result[target] = result

        dtrace.meta = dict(super_group=target.super_group, group=target.group)
        dtraces.append(dtrace)

        result.processed_syn.meta = dict(
            super_group=target.super_group, group=target.group)

        all_syn_trs.append(result.processed_syn)

    skey = lambda tr: (tr.meta['super_group'], tr.meta['group'])

    trace_minmaxs = trace.minmax(all_syn_trs, skey)

    dminmaxs = trace.minmax([x for x in dtraces if x is not None], skey)

    for tr in dtraces:
        if tr:
            dmin, dmax = dminmaxs[skey(tr)]
            tr.ydata /= max(abs(dmin), abs(dmax))

    figs = []
    # put loop over channels here ...
    for cg in cgs:
        targets = cg_to_targets[cg]

        # can keep from here ... until
        nframes = len(targets)

        nx = int(math.ceil(math.sqrt(nframes)))
        ny = (nframes-1)/nx+1

        nxmax = 4
        nymax = 4

        nxx = (nx-1) / nxmax + 1
        nyy = (ny-1) / nymax + 1

        # nz = nxx * nyy

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
            x = dist*num.sin(num.deg2rad(azi))
            y = dist*num.cos(num.deg2rad(azi))
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
            (fxs[num.newaxis, :] - gxs[:, num.newaxis])**2 +
            (fys[num.newaxis, :] - gys[:, num.newaxis])**2)

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

                ixx = ix/nxmax
                iyy = iy/nymax
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

                # can keep until here ...
                amin, amax = trace_minmaxs[target.super_group, target.group]
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

                if target.misfit_config.domain == 'cc_max_norm':
                    axes.set_ylim(-10. * space_factor, 10.)
                else:
                    axes.set_ylim(-absmax*1.33 * space_factor, absmax*1.33)

                itarget = target_index[target]
                result = target_to_result[target]

                dtrace = dtraces[itarget]

                tap_color_annot = (0.35, 0.35, 0.25)
                tap_color_edge = (0.85, 0.85, 0.80)
                tap_color_fill = (0.95, 0.95, 0.90)

                plot_taper( # trace.Taper object
                    axes2, result.processed_obs.get_xdata(), result.taper,
                    fc=tap_color_fill, ec=tap_color_edge)

                obs_color = scolor('aluminium5')
                obs_color_light = light(obs_color, 0.5)

                syn_color = scolor('scarletred2')
                syn_color_light = light(syn_color, 0.5)

                misfit_color = scolor('scarletred2')
                weight_color = scolor('chocolate2')

                cc_color = scolor('aluminium5')

                # no clue what is dtrace ...
                if target.misfit_config.domain == 'cc_max_norm':
                    tref = (result.filtered_obs.tmin +
                            result.filtered_obs.tmax) * 0.5

                    plot_dtrace(
                        axes2, dtrace, space, -1., 1.,
                        fc=light(cc_color, 0.5),
                        ec=cc_color)

                    plot_dtrace_vline(
                        axes2, tref, space, color=tap_color_annot)

                else:
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

                for tmark, text, ha in [
                        (tmarks[0],
                         '$\,$ ' + str_duration(tmarks[0] - source.time),
                         'right'),
                        (tmarks[1],
                         '$\Delta$ ' + str_duration(tmarks[1] - tmarks[0]),
                         'left')]:

                    axes2.annotate(
                        text,
                        xy=(tmark, -0.9),
                        xycoords='data',
                        xytext=(
                            fontsize*0.4 * [-1, 1][ha == 'left'],
                            fontsize*0.2),
                        textcoords='offset points',
                        ha=ha,
                        va='bottom',
                        color=tap_color_annot,
                        fontsize=fontsize)

                rel_w = ws[itarget] / w_max
                rel_c = gcms[itarget] / gcm_max

                sw = 0.25
                sh = 0.1
                ph = 0.01

                for (ih, rw, facecolor, edgecolor) in [
                        (0, rel_w,  light(weight_color, 0.5), weight_color),
                        (1, rel_c,  light(misfit_color, 0.5), misfit_color)]:

                    bar = patches.Rectangle(
                        (1.0-rw*sw, 1.0-(ih+1)*sh+ph), rw*sw, sh-2*ph,
                        facecolor=facecolor, edgecolor=edgecolor,
                        zorder=10,
                        transform=axes.transAxes, clip_on=False)

                    axes.add_patch(bar)

                scale_string = None

                if target.misfit_config.domain == 'cc_max_norm':
                    scale_string = 'Syn/obs scales differ!'

                infos = []
                if scale_string:
                    infos.append(scale_string)

                infos.append('.'.join(x for x in target.codes if x))
                dist = source.distance_to(target)
                azi = source.azibazi_to(target)[0]
                infos.append(str_dist(dist))
                infos.append(u'%.0f\u00B0' % azi)
                infos.append('%.3g' % ws[itarget])
                infos.append('%.3g' % gcms[itarget])
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
                title += ' (%i/%i, %i/%i)' % (iyy+1, nyy, ixx+1, nxx)

            fig.suptitle(title, fontsize=fontsize_title)

    return figs


def histplot_op(ax, data, alpha=.35, color=None, bins=None):
    """
    Modified from pymc3. Additional color argument.
    """
    for i in range(data.shape[1]):
        d = data[:, i]

        mind = num.min(d)
        maxd = num.max(d)
        step = (maxd - mind) / 40.

        if bins is None:
            bins = int(num.ceil((maxd - mind) / step))

        ax.hist(d, bins=bins, alpha=alpha, align='left', color=color)
        ax.set_xlim(mind - .5, maxd + .5)


def traceplot(trace, varnames=None, transform=lambda x: x, figsize=None,
              lines=None, combined=False, grid=False, varbins=None, nbins=40,
              alpha=0.35, priors=None, prior_alpha=1, prior_style='--',
              axs=None, posterior=None):
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
    alpha : float
        Alpha value for plot line. Defaults to 0.35.
    axs : axes
        Matplotlib axes. Defaults to None.

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

    if varnames is None:
        varnames = [name for name in trace.varnames if not name.endswith('_')]

    if 'geo_like' in varnames:
        remove_var(varnames, varname='geo_like')

    if 'seis_like' in varnames:
        remove_var(varnames, varname='seis_like')

    if posterior:
        llk = trace.get_values('like', combine=combined, squeeze=False)
        llk = num.squeeze(transform(llk[0]))
        llk = pmp.make_2d(llk)

        posterior_idxs = get_fit_indexes(llk)

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
            fig.delaxes(axs[rowi, coli])
        else:
            v = varnames[i]

            for d in trace.get_values(v, combine=combined, squeeze=False):
                d = num.squeeze(transform(d))
                d = pmp.make_2d(d)

                if make_bins_flag:
                    varbin = make_bins(d, nbins=nbins)
                    varbins.append(varbin)
                else:
                    varbin = varbins[i]

                histplot_op(
                    axs[rowi, coli], d, bins=varbin, alpha=alpha,
                    color=scolor('aluminium3'))
                axs[rowi, coli].set_title(str(v))
                axs[rowi, coli].grid(grid)
                axs[rowi, coli].set_ylabel("Frequency")

                if lines:
                    try:
                        axs[rowi, coli].axvline(x=lines[v], color="k", lw=1.5)
                    except KeyError:
                        pass

                if posterior:
                    for k, idx in posterior_idxs.iteritems():
                        axs[rowi, coli].axvline(
                            x=d[idx], color=colors[k], lw=1.5)

    fig.tight_layout()
    return fig, axs, varbins


def stage_posteriors(mtrace, n_steps, posterior=None, lines=None):
    """
    Plot variable posteriors from a certain stage of the ATMIP algorithm.

    Parameters
    ----------
    mtrace : :class:`pymc3.backend.base.MultiTrace`
    n_steps : int
        Number of chains to select last samples of each trace.
    posterior : str
        To mark posterior value in distribution 'max', 'min', 'mean', 'all'
    lines : dict
        :func:pymc3.model.Point to draw vertical lines for reference
    """

    def last_sample(x):
        return x[(n_steps - 1)::n_steps].flatten()

    fig, axs, varbins = traceplot(mtrace, transform=last_sample,
        combined=True, lines=lines, posterior=posterior)

    return fig


def draw_posteriors(problem, format='pdf', force=False, dpi=450):
    """
    Identify which stage is the last complete stage and plot posteriors up to
    format : str
        output format: 'display', 'png' or 'pdf'
    """

    mode = problem.config.problem_config.mode

    po = problem.get_plot_options()

    stage = load_stage(problem, stage_number=po.load_stage, load='params')
    step = stage.step

    if po.load_stage is not None:
        list_indexes = [po.load_stage]
    else:
        list_indexes = [str(i) for i in range(step.stage + 1)]

        if stage.number == 'final':
            list_indexes = list_indexes + ['final']

    figs = []

    for s in list_indexes:
        if s == '0':
            draws = 1
        else:
            draws = step.n_steps

        stage_path = os.path.join(
            problem.config.project_dir, mode, 'stage_%s' % s)

        outpath = os.path.join(
            problem.config.project_dir,
            mode, po.figure_dir, 'stage_%s.%s' % (s, format))

        if not os.path.exists(outpath) or force:
            logger.info('plotting stage: %s' % stage_path)
            mtrace = backend.load(stage_path, model=problem.model)
            fig = stage_posteriors(mtrace, n_steps=draws, posterior='all',
                    lines=po.reference)
            if not format == 'display':
                logger.info('saving figure to %s' % outpath)
                fig.savefig(outpath, format=format, dpi=dpi)
            else:
                figs.append(fig)

        else:
            logger.info('plot for stage %s exists. Use force=True for'
                ' replotting!' % s)

    if format == 'display':
        plt.show()


def draw_correlation_hist(problem, format='pdf', force=False, dpi=450):
    """
    Draw parameter correlation plot and histograms from the final atmip stage.
    Only feasible for 'geometry' problem.
    """

    mode = problem.config.problem_config.mode

    assert mode == 'geometry'

    def last_sample(x):
        return x[(n_steps - 1)::n_steps].flatten()

    n_steps = problem.config.sampler_config.parameters.n_steps

    stage = load_stage(problem, load='trace')

    po = problem.get_plot_options()

    outpath = os.path.join(
        problem.config.project_dir,
        mode, po.figure_dir, 'corr_hist_%s.%s' % (stage.number, format))

    if not os.path.exists(outpath) or force:
        fig, axs = correlation_plot_hist(
            mtrace=stage.mtrace,
            varnames=problem.config.problem_config.select_variables(),
            transform=last_sample,
            cmap=plt.cm.gist_earth_r,
            point=po.reference,
            point_size='8',
            point_color='red')
    else:
        logger.info('correlation plot exists. Use force=True for replotting!')

    if format == 'display':
        plt.show()
    else:
        logger.info('saving figure to %s' % outpath)
        fig.savefig(outpath, format=format, dpi=dpi)


def n_model_plot(models, axes=None):
    '''
    Plot cake layered earth models.
    '''
    if axes is None:
        from matplotlib import pyplot as plt
        cp.mpl_init()
        axes = plt.gca()
    else:
        plt = None

    cp.labelspace(axes)
    cp.labels_model(axes=axes)
    cp.sketch_model(models[0], axes=axes)

    ref = models.pop(0)

    for mod in models:
        z = mod.profile('z')
        vp = mod.profile('vp')
        vs = mod.profile('vs')
        axes.plot(vp, z, color=cp.colors[0], lw=1.)
        axes.plot(vs, z, color=cp.colors[2], lw=1.)

    z = ref.profile('z')
    vp = ref.profile('vp')
    vs = ref.profile('vs')
    axes.plot(vp, z, color=cp.colors[10], lw=2.)
    axes.plot(vs, z, color=cp.colors[12], lw=2.)

    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    xmin = 0.
    my = (ymax - ymin) * 0.05
    mx = (xmax - xmin) * 0.2
    axes.set_ylim(ymax + my, ymin - my)
    axes.set_xlim(xmin, xmax + mx)
    if plt:
        plt.show()


def load_earthmodels(engine, targets, depth_max='cmb'):
    earthmodels = []
    for t in targets:
        store = engine.get_store(t.store_id)
        em = store.config.earthmodel_1d.extract(depth_max=depth_max)
        earthmodels.append(em)

    return earthmodels


plots_catalog = {
    'correlation_hist': draw_correlation_hist,
    'stage_posteriors': draw_posteriors}


def plot_results(problem, plotoptions)


def available_plots():
    return list(plots_catalog.keys())
