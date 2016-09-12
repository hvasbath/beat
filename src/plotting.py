from pyrocko import cake_plot as cp
import pymc3 as pm

import os
from beat import utility, models, backend
from matplotlib import pylab as plt

import numpy as num
from pyrocko import cake

from matplotlib.backends.backend_pdf import PdfPages


def plot_misfits(problem, mtrace, mode='geometry', posterior='mean'):

    step, _ = utility.load_atmip_params(
                problem.config.project_dir, 'final', mode)
    _, step.array_population, step.likelihoods = \
                                    step.select_end_points(mtrace)

    if posterior == 'mean':
        out_point = step.mean_end_points()

    seis_synths, geo_synths = problem.get_synthetics(out_point)

    return seis_synths, geo_synths


def stage_posteriors(mtrace, output='display'):
    '''
    Plot variable posteriors from certain stage of the ATMIP algorithm.
    '''
    PLT = pm.plots.traceplot(mtrace, combined=True)
    if output == 'display':
        plt.show(PLT[0][0])
    elif output == 'png':
        plt.savefig('stage_posterior.png', dpi=300)


def plot_all_posteriors(project_dir, mode='geometry'):
    '''
    Loop through all stages and plot the pdfs of the variables.
    '''
    problem = models.load_model(project_dir)

    step, _ = utility.load_atmip_params(project_dir, 'final', mode=mode)

    for i in range(step.stage + 1):
        stage_path = os.path.join(project_dir, mode, 'stage_%i' % i )
        mtrace = backend.load(stage_path, model=problem.model)
        os.chdir(stage_path)
        print('plotting stage path: %s' %stage_path)
        stage_posteriors(mtrace, output='png')

    stage_path = os.path.join(project_dir, mode, 'stage_final')
    mtrace = backend.load(stage_path, model=problem.model)
    os.chdir(stage_path)
    stage_posteriors(mtrace, output='png')


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
    my = (ymax-ymin)*0.05
    mx = (xmax-xmin)*0.2
    axes.set_ylim(ymax+my, ymin-my)
    axes.set_xlim(xmin, xmax+mx)
    if plt:
        plt.show()

def load_earthmodels(engine, targets, depth_max='cmb'):
    earthmodels = []
    for t in targets:
        store = engine.get_store(t.store_id)
        em = store.config.earthmodel_1d.extract(depth_max=depth_max)
        earthmodels.append(em)

    return earthmodels
