from pyrocko import cake_plot as cp
import pymc3 as pm

import os
from beat import utility, backend
from matplotlib import pylab as plt

import numpy as num
from pyrocko import cake, util

from matplotlib.backends.backend_pdf import PdfPages


def plot(lons, lats, disp):
    '''
    Very simple scatter plot of displacements for fast inspections.
    '''
    #colim = num.max([disp.max(), num.abs(disp.min())])
    ax = plt.axes()
    im = ax.scatter(lons, lats, 15, disp, edgecolors='none')
    plt.colorbar(im)
    plt.title('Displacements [m]')
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


def stage_posteriors(mtrace, n_steps, output='display',
                        outpath='./stage_posterior.png', lines=None):
    '''
    Plot variable posteriors from certain stage of the ATMIP algorithm.
    n_steps of chains to select last samples of each trace.
    lines - a point to draw vertical lines for
    '''
    def last_sample(x):
        return x[(n_steps - 1)::n_steps]

    PLT = pm.plots.traceplot(mtrace, transform=last_sample, combined=True,
        lines=lines)
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
        problem.config.project_dir, mode, 'figures', 'stage_final')
    mtrace = backend.load(stage_path, model=problem.model)
    os.chdir(stage_path)
    print('plotting stage path: %s' % stage_path)
    stage_posteriors(mtrace, n_steps=draws, output='png', outpath=stage_path)


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
