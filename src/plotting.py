from pyrocko import cake_plot as cp
from pymc3 import plots as pmp

import os
import logging

from beat import utility, backend

from matplotlib import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as num
from pyrocko import cake, util

logger = logging.getLogger('plotting')


def correlation_plot(mtrace, varnames=None,
        transform=lambda x: x, figsize=None, cmap=None):
    """
    Plot 2d marginals and their correlations of the parameters.
    """

    if varnames is None:
        varnames = mtrace.varnames

    nvar = len(varnames)

    if figsize is None:
        figsize = (8.2, 11.7)   # A4 landscape

    fig, axs = plt.subplots(sharey='row', sharex='col',
        nrows=nvar - 1, ncols=nvar - 1, figsize=figsize)

    d = dict()
    for var in varnames:
        d[var] = transform(mtrace.get_values(
                var, combine=True, squeeze=True))

    for k in range(nvar - 1):
        a = d[varnames[k]]
        for l in range(k + 1, nvar):
            print varnames[k], varnames[l]
            b = d[varnames[l]]

            pmp.kde2plot(a, b, grid=200, ax=axs[l - 1, k], cmap=cmap)

            if k == 0:
                axs[l - 1, k].set_ylabel(varnames[l])

        axs[l - 1, k].set_xlabel(varnames[k])

    for k in range(nvar - 1):
        for l in range(k):
            fig.delaxes(axs[l, k])

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


def plot_misfits(problem, posterior='mean', dataset='geodetic'):

    mode = problem.config.problem_config.mode

    mtrace = backend.load(
        problem.outfolder + '/stage_final', model=problem.model)

    figure_path = os.path.join(problem.outfolder, 'figures')
    util.ensuredir(figure_path)

    step, _ = utility.load_atmip_params(
                problem.config.project_dir, 'final', mode)
    population, _, llk = step.select_end_points(mtrace)

    if posterior == 'mean':
        idx = (num.abs(llk - llk.mean())).argmin()

    if posterior == 'min':
        idx = (num.abs(llk - llk.min())).argmin()

    elif posterior == 'max':
        idx = (num.abs(llk - llk.max())).argmin()

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
              ax=None, posterior=None):
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
        To mark posterior value in distribution 'max', 'min', 'mean'
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
    ax : axes
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

    if posterior is not None:
        llk = trace.get_values('like', combine=combined, squeeze=False)
        llk = num.squeeze(transform(llk[0]))
        llk = pmp.make_2d(llk)

        if posterior == 'mean':
            idx = (num.abs(llk - llk.mean())).argmin()

        if posterior == 'min':
            idx = (num.abs(llk - llk.min())).argmin()

        elif posterior == 'max':
            idx = (num.abs(llk - llk.max())).argmin()

    n = len(varnames)
    nrow = int(num.ceil(n / 2.))
    ncol = 2

    if figsize is None:
        figsize = (12, n * 2)

    if ax is None:
        fig, ax = plt.subplots(nrow, ncol, squeeze=False, figsize=figsize)
    elif ax.shape != (nrow, ncol):
        logger.warn('traceplot requires n*2 subplots')
        return None

    if varbins is None:
        make_bins_flag = True
        varbins = []
    else:
        make_bins_flag = False

    for i, v in enumerate(varnames):
        coli, rowi = utility.mod_i(i, nrow)

        for d in trace.get_values(v, combine=combined, squeeze=False):
            d = num.squeeze(transform(d))
            d = pmp.make_2d(d)

            if make_bins_flag:
                varbin = make_bins(d, nbins=nbins)
                varbins.append(varbin)
            else:
                varbin = varbins[i]

            histplot_op(ax[rowi, coli], d, bins=varbin, alpha=alpha)
            ax[rowi, coli].set_title(str(v))
            ax[rowi, coli].grid(grid)
            ax[rowi, coli].set_ylabel("Frequency")

            if lines:
                try:
                    ax[rowi, coli].axvline(x=lines[v], color="r", lw=1.5)
                except KeyError:
                    pass

            if posterior is not None:
                ax[rowi, coli].axvline(x=d[idx], color="b", lw=1.5)

    plt.tight_layout()
    return ax, varbins


def stage_posteriors(mtrace, n_steps, output='display',
            outpath='./stage_posterior.png', lines=None, style='lines'):
    '''
    Plot variable posteriors from certain stage of the ATMIP algorithm.
    n_steps of chains to select last samples of each trace.
    lines - a point to draw vertical lines for
    '''
    def last_sample(x):
        return x[(n_steps - 1)::n_steps].flatten()

    if style == 'lines' or lines is not None:
        PLT = traceplot(mtrace, transform=last_sample, combined=True,
            lines=lines)
    else:
        PLT = pmp.plot_posterior(mtrace, transform=last_sample)

    if output == 'display':
        plt.show(PLT[0][0])
    elif output == 'png':
        plt.savefig(outpath, dpi=300)

    plt.close()


def plot_all_posteriors(problem):
    '''
    Loop through all stages and plot the pdfs of the variables.
    Inputs: problem Object
    '''
    mode = problem.config.problem_config.mode

    step, _ = utility.load_atmip_params(
        problem.config.project_dir, 'final', mode=mode)

    for i in range(step.stage + 1):
        if i == 0:
            draws = 1
        else:
            draws = step.n_steps

        stage_path = os.path.join(
            problem.config.project_dir, mode, 'stage_%i' % i)
        mtrace = backend.load(stage_path, model=problem.model)

        outpath = os.path.join(
            problem.config.project_dir, mode, 'figures', 'stage_%i' % i)
        print('plotting stage path: %s' % stage_path)
        stage_posteriors(
            mtrace, n_steps=draws, output='png', outpath=outpath)

    stage_path = os.path.join(
            problem.config.project_dir, mode, 'stage_final')
    mtrace = backend.load(stage_path, model=problem.model)

    out_path = os.path.join(
        problem.config.project_dir, mode, 'figures', 'stage_final')
    print('plotting stage path: %s' % stage_path)
    stage_posteriors(mtrace, n_steps=draws, output='png', outpath=out_path)


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
