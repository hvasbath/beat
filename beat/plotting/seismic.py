import logging
import os

from scipy import stats

import numpy as num
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from pymc3.plots.utils import make_2d
from pyrocko import gmtpy, trace
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.guts import load
from pyrocko.moment_tensor import to6
from pyrocko.plot import (
    beachball,
    mpl_graph_color,
    mpl_init,
    mpl_margins,
    mpl_papersize,
)

from beat import utility
from beat.heart import calculate_radiation_weights
from beat.models import Stage, load_stage

from .common import (
    draw_line_on_array,
    format_axes,
    get_gmt_config,
    get_result_point,
    plot_inset_hist,
    spherical_kde_op,
    str_dist,
    str_duration,
    str_unit,
    get_weights_point,
)

km = 1000.0
SQRT2 = num.sqrt(2.0)
PI = num.pi

logger = logging.getLogger("plotting.seismic")


def n_model_plot(models, axes=None, draw_bg=True, highlightidx=[]):
    """
    Plot cake layered earth models.
    """
    from pyrocko import cake_plot as cp

    fontsize = 10
    if axes is None:
        mpl_init(fontsize=fontsize)
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize("a6", "portrait")
        )
        labelpos = mpl_margins(
            fig, left=6, bottom=4, top=1.5, right=0.5, units=fontsize
        )
        labelpos(axes, 2.0, 1.5)

    def plot_profile(mod, axes, vp_c, vs_c, lw=0.5):
        z = mod.profile("z")
        vp = mod.profile("vp")
        vs = mod.profile("vs")
        axes.plot(vp, z, color=vp_c, lw=lw)
        axes.plot(vs, z, color=vs_c, lw=lw)

    cp.labelspace(axes)
    cp.labels_model(axes=axes)
    if draw_bg:
        cp.sketch_model(models[0], axes=axes)
    else:
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)

    ref_vp_c = scolor("aluminium5")
    ref_vs_c = scolor("aluminium5")
    vp_c = scolor("scarletred2")
    vs_c = scolor("skyblue2")

    for i, mod in enumerate(models):
        plot_profile(
            mod, axes, vp_c=cp.light(vp_c, 0.3), vs_c=cp.light(vs_c, 0.3), lw=1.0
        )

    for count, i in enumerate(sorted(highlightidx)):
        if count == 0:
            vpcolor = ref_vp_c
            vscolor = ref_vs_c
        else:
            vpcolor = vp_c
            vscolor = vs_c

        plot_profile(models[i], axes, vp_c=vpcolor, vs_c=vscolor, lw=2.0)

    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    xmin = 0.0
    my = (ymax - ymin) * 0.05
    mx = (xmax - xmin) * 0.2
    axes.set_ylim(ymax, ymin - my)
    axes.set_xlim(xmin, xmax + mx)
    return fig, axes


def load_earthmodels(store_superdir, store_ids, depth_max="cmb"):

    ems = []
    emr = []
    for store_id in store_ids:
        path = os.path.join(store_superdir, store_id, "config")
        config = load(filename=path)
        em = config.earthmodel_1d.extract(depth_max=depth_max)
        ems.append(em)

        if config.earthmodel_receiver_1d is not None:
            emr.append(config.earthmodel_receiver_1d)

    return [ems, emr]


def draw_earthmodels(problem, plot_options):

    from beat.heart import init_geodetic_targets, init_seismic_targets

    po = plot_options

    for datatype, composite in problem.composites.items():

        if datatype == "seismic":
            models_dict = {}
            sc = problem.config.seismic_config

            if sc.gf_config.reference_location is None:
                plot_stations = composite.datahandlers[0].stations
            else:
                plot_stations = [composite.datahandlers[0].stations[0]]
                plot_stations[0].station = sc.gf_config.reference_location.station

            for station in plot_stations:
                outbasepath = os.path.join(
                    problem.outfolder,
                    po.figure_dir,
                    "%s_%s_velocity_model" % (datatype, station.station),
                )

                if not os.path.exists(outbasepath) or po.force:
                    targets = init_seismic_targets(
                        [station],
                        earth_model_name=sc.gf_config.earth_model_name,
                        channels=sc.get_unique_channels()[0],
                        sample_rate=sc.gf_config.sample_rate,
                        crust_inds=list(range(*sc.gf_config.n_variations)),
                        interpolation="multilinear",
                    )
                    store_ids = [t.store_id for t in targets]

                    models = load_earthmodels(
                        composite.engine.store_superdirs[0],
                        store_ids,
                        depth_max=sc.gf_config.depth_limit_variation * km,
                    )

                    for i, mods in enumerate(models):
                        if i == 0:
                            site = "source"
                        elif i == 1:
                            site = "receiver"

                        outpath = outbasepath + "_%s.%s" % (site, po.outformat)

                        models_dict[outpath] = mods

                else:
                    logger.info(
                        "%s earthmodel plot for station %s exists. Use "
                        "force=True for replotting!" % (datatype, station.station)
                    )

        elif datatype == "geodetic":
            gc = problem.config.geodetic_config

            models_dict = {}
            outpath = os.path.join(
                problem.outfolder,
                po.figure_dir,
                "%s_%s_velocity_model.%s" % (datatype, "psgrn", po.outformat),
            )

            if not os.path.exists(outpath) or po.force:
                targets = init_geodetic_targets(
                    datasets=composite.datasets,
                    earth_model_name=gc.gf_config.earth_model_name,
                    interpolation="multilinear",
                    crust_inds=list(range(*gc.gf_config.n_variations)),
                    sample_rate=gc.gf_config.sample_rate,
                )

                models = load_earthmodels(
                    store_superdir=composite.engine.store_superdirs[0],
                    targets=targets,
                    depth_max=gc.gf_config.source_depth_max * km,
                )
                models_dict[outpath] = models[0]  # select only source site

            else:
                logger.info(
                    "%s earthmodel plot exists. Use force=True for"
                    " replotting!" % datatype
                )
                return

        else:
            raise TypeError("Plot for datatype %s not (yet) supported" % datatype)

        figs = []
        axes = []
        tobepopped = []
        for path, models in models_dict.items():
            if len(models) > 0:
                fig, axs = n_model_plot(
                    models, axes=None, draw_bg=po.reference, highlightidx=[0]
                )
                figs.append(fig)
                axes.append(axs)
            else:
                tobepopped.append(path)

        for entry in tobepopped:
            models_dict.pop(entry)

        if po.outformat == "display":
            plt.show()
        else:
            for fig, outpath in zip(figs, models_dict.keys()):
                logger.info("saving figure to %s" % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)


def get_fuzzy_cmap(ncolors=256):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "dummy", ["white", scolor("chocolate2"), scolor("scarletred2")], N=ncolors
    )


def fuzzy_waveforms(
    ax,
    traces,
    linewidth,
    zorder=0,
    extent=None,
    grid_size=(500, 500),
    cmap=None,
    alpha=0.6,
):
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
        cmap = get_fuzzy_cmap()
        # cmap = plt.cm.gist_earth_r

    if extent is None:
        key = traces[0].channel
        skey = lambda tr: tr.channel

        ymin, ymax = trace.minmax(traces, key=skey)[key]
        xmin, xmax = trace.minmaxtime(traces, key=skey)[key]

        ymax = max(abs(ymin), abs(ymax))
        ymin = -ymax

        extent = [xmin, xmax, ymin, ymax]

    grid = num.zeros(grid_size, dtype="float64")

    for tr in traces:

        draw_line_on_array(
            tr.get_xdata(),
            tr.ydata,
            grid=grid,
            extent=extent,
            grid_resolution=grid.shape,
            linewidth=linewidth,
        )

    # increase contrast reduce high intense values
    # truncate = len(traces) / 2
    # grid[grid > truncate] = truncate
    ax.imshow(
        grid,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        alpha=alpha,
        zorder=zorder,
    )


def zero_pad_spectrum(trace):
    ydata = trace.get_ydata()  # [lower_idx:upper_idx]
    ydata[[0, -1]] = 0.0
    return ydata


def fuzzy_spectrum(
    ax,
    traces,
    taper_frequencies=(0, 1.0),
    ypad_factor=1.2,
    zorder=0,
    extent=None,
    linewidth=7.0,
    grid_size=(500, 500),
    cmap=None,
    alpha=0.5,
):

    if cmap is None:
        cmap = get_fuzzy_cmap()

    grid = num.zeros(grid_size, dtype="float64")
    fxdata = traces[0].get_xdata()

    if extent is None:
        key = traces[0].channel
        skey = lambda tr: tr.channel

        ymin, ymax = trace.minmax(traces, key=skey)[key]

        lower_idx, upper_idx = utility.get_valid_spectrum_data(
            deltaf=fxdata[1] - fxdata[0], taper_frequencies=taper_frequencies
        )

        extent = [*taper_frequencies, 0, ypad_factor * ymax]
    else:
        lower_idx, upper_idx = 0, -1

    # fxdata = fxdata[lower_idx:upper_idx]
    for tr in traces:
        ydata = zero_pad_spectrum(tr)
        draw_line_on_array(
            fxdata,
            ydata,
            grid=grid,
            extent=extent,
            grid_resolution=grid.shape,
            linewidth=linewidth,
        )

    ax.imshow(
        grid,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        alpha=alpha,
        zorder=zorder,
    )


def extract_time_shifts(point, hierarchicals, wmap):
    if wmap.config.domain == "time":
        try:
            time_shifts = point[wmap.time_shifts_id][wmap.station_correction_idxs]
        except KeyError:
            if wmap.time_shifts_id in hierarchicals:
                time_shifts = hierarchicals[wmap.time_shifts_id][
                    wmap.station_correction_idxs
                ]
            else:
                raise ValueError(
                    "Sampling results do not contain time-shifts for wmap"
                    " %s!" % wmap.time_shifts_id
                )
    else:
        time_shifts = [0] * wmap.n_t
    return time_shifts


def subplot_waveforms(
    axes,
    axes2,
    po,
    target,
    source,
    traces,
    result,
    stdz_residual,
    var_reductions,
    time_shifts,
    time_shift_bounds,
    synth_plot_flag,
    absmax,
    mode,
    fontsize,
    tap_color_edge,
    syn_color,
    obs_color,
    time_shift_color,
    tap_color_annot,
):
    def plot_trace(axes, tr, **kwargs):
        return axes.plot(tr.get_xdata(), tr.get_ydata(), **kwargs)

    def plot_taper(axes, t, taper, mode="geometry", **kwargs):
        y = num.ones(t.size) * 0.9
        if mode == "geometry":
            taper(y, t[0], t[1] - t[0])
        y2 = num.concatenate((y, -y[::-1]))
        t2 = num.concatenate((t, t[::-1]))
        axes.fill(t2, y2, **kwargs)

    skey = lambda tr: tr.channel

    if po.nensemble > 1:
        xmin, xmax = trace.minmaxtime(traces, key=skey)[target.codes[3]]
        fuzzy_waveforms(
            axes, traces, linewidth=7, zorder=0, grid_size=(500, 500), alpha=1.0
        )

        logger.debug("Plotting variance reductions for %s" % target.nslcd_id_str)

        best_data = var_reductions[0]

        in_ax = plot_inset_hist(
            axes,
            data=make_2d(var_reductions),
            best_data=best_data,
            bbox_to_anchor=(0.9, 0.75, 0.2, 0.2),
            background_alpha=0.7,
        )
        in_ax.set_title("VR [%]", fontsize=5)

    # histogram of stdz residual
    in_ax_res = plot_inset_hist(
        axes,
        data=make_2d(stdz_residual),
        best_data=None,
        bbox_to_anchor=(0.65, 0.75, 0.2, 0.2),
        color="grey",
        background_alpha=0.7,
    )
    # reference gaussian
    x = num.linspace(*stats.norm.ppf((0.001, 0.999)), 100)
    gauss = stats.norm.pdf(x)
    in_ax_res.plot(x, gauss, "k-", lw=0.5, alpha=0.8)
    in_ax_res.set_title("std. res. [$\sigma$]", fontsize=5)

    plot_taper(
        axes2,
        result.processed_obs.get_xdata(),
        result.taper,
        mode=mode,
        fc="None",
        ec=tap_color_edge,
        zorder=4,
        alpha=0.6,
    )

    if synth_plot_flag:
        # only plot if highlighted point exists
        if po.plot_projection == "individual":
            for i, tr in enumerate(result.source_contributions):
                plot_trace(axes, tr, color=mpl_graph_color(i), lw=0.5, zorder=5)
        else:
            plot_trace(axes, result.processed_syn, color=syn_color, lw=0.5, zorder=5)

    plot_trace(axes, result.processed_obs, color=obs_color, lw=0.5, zorder=5)

    xdata = result.processed_obs.get_xdata()
    axes.set_xlim(xdata[0], xdata[-1])

    tmarks = [result.processed_obs.tmin, result.processed_obs.tmax]
    tmark_fontsize = fontsize - 1

    if time_shifts is not None:
        sidebar_ybounds = [-0.3, -0.4]
        ytmarks = [-1.15, -0.7]
        hor_alignment = "center"

        if synth_plot_flag:
            best_data = time_shifts[0]
        else:  # for None post_llk
            best_data = None

        if po.nensemble > 1:
            in_ax = plot_inset_hist(
                axes,
                data=make_2d(time_shifts),
                best_data=best_data,
                bbox_to_anchor=(-0.0985, 0.16, 0.2, 0.2),
                # cmap=plt.cm.get_cmap('seismic'),
                # cbounds=time_shift_bounds,
                color=time_shift_color,
                alpha=0.7,
                background_alpha=0.7,
            )
            in_ax.set_xlim(*time_shift_bounds)
    else:
        sidebar_ybounds = [-0.6, -0.4]
        ytmarks = [-0.9, -0.7]
        hor_alignment = "center"

    for tmark, ybound in zip(tmarks, sidebar_ybounds):
        axes2.plot([tmark, tmark], [ybound, 0.1], color=tap_color_annot)

    for xtmark, ytmark, text, ha, va in [
        (
            tmarks[0],
            ytmarks[0],
            "$\,$ " + str_duration(tmarks[0] - source.time),
            hor_alignment,
            "bottom",
        ),
        (
            tmarks[1],
            ytmarks[1],
            "$\Delta$ " + str_duration(tmarks[1] - tmarks[0]),
            "center",
            "bottom",
        ),
    ]:

        axes2.annotate(
            text,
            xy=(xtmark, ytmark),
            xycoords="data",
            xytext=(fontsize * 0.4 * [-1, 1][ha == "left"], fontsize * 0.2),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=tap_color_annot,
            fontsize=tmark_fontsize,
            zorder=10,
        )

    # annotate axis amplitude
    axes.annotate(
        "%0.3g %s -" % (-absmax, str_unit(target.quantity)),
        xycoords="data",
        xy=(tmarks[1], -absmax / 2),
        xytext=(1.0, 1.0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=fontsize - 3,
        color=obs_color,
        fontstyle="normal",
    )

    axes2.set_zorder(10)


def subplot_spectrum(
    axes,
    axes2,
    po,
    target,
    traces,
    result,
    stdz_residual,
    synth_plot_flag,
    only_spectrum,
    var_reductions,
    fontsize,
    syn_color,
    obs_color,
    misfit_color,
    tap_color_annot,
    ypad_factor,
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if not only_spectrum:
        axes = inset_axes(
            axes2,
            width="100%",
            height="100%",
            bbox_to_anchor=(-0.05, -0.15, 0.65, 0.24),
            bbox_transform=axes.transAxes,
            loc=2,
            borderpad=0,
        )
        bbox_y = -0.15
    else:
        bbox_y = 0.75

    taper_frequencies = result.get_taper_frequencies()
    if po.nensemble > 1:
        fuzzy_spectrum(
            axes,
            traces=traces,
            taper_frequencies=taper_frequencies,
            ypad_factor=ypad_factor,
            zorder=0,
            extent=None,
            linewidth=7.0,
            grid_size=(500, 500),
            cmap=None,
            alpha=1.0,
        )

        if synth_plot_flag:
            best_data = var_reductions[0]
        else:  # for None post_llk
            best_data = None

        in_ax = plot_inset_hist(
            axes2,
            data=make_2d(var_reductions),
            best_data=best_data,
            bbox_to_anchor=(0.9, bbox_y, 0.2, 0.2),
        )
        in_ax.set_title("SPC_VR [%]", fontsize=5)

    # histogram of stdz residual
    in_ax_res = plot_inset_hist(
        axes2,
        data=make_2d(stdz_residual),
        best_data=None,
        bbox_to_anchor=(0.65, bbox_y, 0.2, 0.2),
        color="grey",
        background_alpha=0.7,
    )
    # reference gaussian
    x = num.linspace(*stats.norm.ppf((0.001, 0.999)), 100)
    gauss = stats.norm.pdf(x)
    in_ax_res.plot(x, gauss, "k-", lw=0.5, alpha=0.8)
    in_ax_res.set_title("spc. std. res. [$\sigma$]", fontsize=5)

    fxdata = result.processed_syn.get_xdata()

    linewidths = [1.0, 0.5, 0.5]
    colors = [obs_color, syn_color, misfit_color]
    ymaxs = []
    for attr_suffix, lw, color in zip(["obs", "syn", "res"], linewidths, colors):

        tr = getattr(result, "processed_{}".format(attr_suffix))
        ydata = zero_pad_spectrum(tr)
        ymaxs.append(ydata.max())

        if attr_suffix == "res":
            axes.fill(fxdata, ydata, clip_on=False, color=color, lw=lw, alpha=0.15)
        else:
            axes.plot(fxdata, ydata, color=color, lw=lw)

    ymax = num.max(ymaxs)

    format_axes(axes, remove=["right", "top", "left", "bottom"])
    axes.yaxis.set_visible(False)
    axes.xaxis.set_visible(False)
    axes.set_xlim([fxdata.min(), fxdata.max()])
    axes.set_ylim([0, ypad_factor * ymax])

    if only_spectrum:
        ybounds = [0.6 * ymax, 0.6 * ymax]
        ymax_factor_amp = 0.45
        ymax_factor_f = 0.2
    else:
        ybounds = [0.5 * ymax, ymax]
        ymax_factor_amp = 0.9
        ymax_factor_f = 0.4

    for tmark, ybound in zip([fxdata[0], fxdata[-1]], ybounds):
        axes.plot([tmark, tmark], [0.0, ybound], color=tap_color_annot, lw=0.75)

    # annotate axis amplitude
    xpos = fxdata[-1]
    axes.annotate(
        "%0.3g -" % (ymax),
        xycoords="data",
        xy=(xpos, ymax_factor_amp * ymax),
        xytext=(1.0, 1.0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=fontsize - 3,
        color=obs_color,
        fontstyle="normal",
    )

    axes.annotate(
        "$ f \ |\ ^{%0.2g}_{%0.2g} \ $" % (fxdata[0], xpos),
        xycoords="data",
        xy=(xpos, ymax_factor_f * ymax),
        xytext=(1.0, 1.0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=fontsize + 1,
        color=obs_color,
        fontstyle="normal",
    )


def seismic_fits(problem, stage, plot_options):
    """
    Modified from grond. Plot synthetic and data waveforms and the misfit for
    the selected posterior model.
    """

    time_shift_color = scolor("aluminium3")
    obs_color = scolor("aluminium5")
    syn_color = scolor("scarletred2")
    misfit_color = scolor("scarletred2")

    tap_color_annot = (0.35, 0.35, 0.25)
    tap_color_edge = (0.85, 0.85, 0.80)
    # tap_color_fill = (0.95, 0.95, 0.90)

    problem.init_hierarchicals()
    composite = problem.composites["seismic"]

    fontsize = 8
    fontsize_title = 10

    target_index = dict((target, i) for (i, target) in enumerate(composite.targets))

    po = plot_options

    if not po.reference:
        best_point = get_result_point(stage.mtrace, po.post_llk)
    else:
        best_point = po.reference

    try:
        composite.point2sources(best_point)
        source = composite.sources[0]
        chop_bounds = ["a", "d"]
    except AttributeError:
        logger.info("FFI waveform fit, using reference source ...")
        source = composite.config.gf_config.reference_sources[0]
        source.time = composite.event.time
        chop_bounds = ["b", "c"]

    if best_point:  # for source individual contributions
        bresults = composite.assemble_results(
            best_point, outmode="tapered_data", chop_bounds=chop_bounds
        )
        synth_plot_flag = True
    else:
        # get dummy results for data
        logger.warning('Got "None" post_llk, still loading MAP for VR calculation')
        best_point = get_result_point(stage.mtrace, "max")
        bresults = composite.assemble_results(best_point, chop_bounds=chop_bounds)
        synth_plot_flag = False

    tpoint = get_weights_point(composite, best_point, problem.config)

    composite.analyse_noise(tpoint, chop_bounds=chop_bounds)
    composite.update_weights(tpoint, chop_bounds=chop_bounds)
    if plot_options.nensemble > 1:
        from tqdm import tqdm

        logger.info("Collecting ensemble of %i synthetic waveforms ..." % po.nensemble)
        nchains = len(stage.mtrace)
        csteps = float(nchains) / po.nensemble
        idxs = num.floor(num.arange(0, nchains, csteps)).astype("int32")
        ens_results = []
        points = []
        ens_var_reductions = []
        for idx in tqdm(idxs):
            point = stage.mtrace.point(idx=idx)
            points.append(point)
            results = composite.assemble_results(
                point, chop_bounds=chop_bounds, force=False
            )
            ens_results.append(results)
            ens_var_reductions.append(
                composite.get_variance_reductions(
                    point,
                    weights=composite.weights,
                    results=results,
                    chop_bounds=chop_bounds,
                )
            )

    bvar_reductions = composite.get_variance_reductions(
        best_point, weights=composite.weights, results=bresults, chop_bounds=chop_bounds
    )

    stdz_residuals = composite.get_standardized_residuals(
        best_point,
        chop_bounds=chop_bounds,
        results=bresults,
        weights=composite.weights,
    )

    # collecting results for targets
    logger.info("Mapping results to targets ...")
    target_to_results = {}
    all_syn_trs_target = {}
    all_var_reductions = {}

    for target in composite.targets:
        target_results = []
        target_synths = []
        target_var_reductions = []

        i = target_index[target]

        nslcd_id_str = target.nslcd_id_str
        target_results.append(bresults[i])
        target_synths.append(bresults[i].processed_syn)
        target_var_reductions.append(bvar_reductions[nslcd_id_str])

        if plot_options.nensemble > 1:
            for results, var_reductions in zip(ens_results, ens_var_reductions):
                # put all results per target here not only single

                target_results.append(results[i])
                target_synths.append(results[i].processed_syn)
                target_var_reductions.append(var_reductions[nslcd_id_str])

        target_to_results[target] = target_results
        all_syn_trs_target[target] = target_synths
        all_var_reductions[target] = num.array(target_var_reductions) * 100.0

    # collecting time-shifts:
    station_corr = composite.config.station_corrections
    time_shift_bounds = [0, 0]
    if station_corr:
        tshifts = problem.config.problem_config.hierarchicals["time_shift"]
        time_shift_bounds = [tshifts.lower.squeeze(), tshifts.upper.squeeze()]

        logger.info("Collecting time-shifts ...")
        if plot_options.nensemble > 1:
            ens_time_shifts = []
            for point in points:
                comp_time_shifts = []
                for wmap in composite.wavemaps:
                    comp_time_shifts.append(
                        extract_time_shifts(point, composite.hierarchicals, wmap)
                    )

                ens_time_shifts.append(num.hstack(comp_time_shifts))

        btime_shifts = num.hstack(
            [
                extract_time_shifts(best_point, composite.hierarchicals, wmap)
                for wmap in composite.wavemaps
            ]
        )

        logger.info("Mapping time-shifts to targets ...")
        all_time_shifts = {}
        for target in composite.targets:
            target_time_shifts = []
            i = target_index[target]
            target_time_shifts.append(btime_shifts[i])

            if plot_options.nensemble > 1:
                for time_shifts in ens_time_shifts:
                    target_time_shifts.append(time_shifts[i])

            all_time_shifts[target] = num.array(target_time_shifts)
    else:
        all_time_shifts = {target: None for target in composite.targets}

    event_figs = []
    for event_idx, event in enumerate(composite.events):
        # gather event related targets
        event_targets = []
        for wmap in composite.wavemaps:
            if event_idx == wmap.config.event_idx:
                event_targets.extend(wmap.targets)

        target_codes_to_targets = utility.gather(event_targets, lambda t: t.codes)

        # gather unique target codes
        unique_target_codes = list(target_codes_to_targets.keys())
        cg_to_target_codes = utility.gather(unique_target_codes, lambda t: t[3])

        skey = lambda tr: tr.channel
        cgs = cg_to_target_codes.keys()

        figs = []
        logger.info("Plotting waveforms ... for event number: %i" % event_idx)
        logger.info(event.__str__())
        for cg in cgs:
            target_codes = cg_to_target_codes[cg]

            nframes = len(target_codes)

            nx = int(num.ceil(num.sqrt(nframes)))
            ny = (nframes - 1) // nx + 1

            logger.debug("nx %i, ny %i" % (nx, ny))

            nxmax = 4
            nymax = 4

            nxx = (nx - 1) // nxmax + 1
            nyy = (ny - 1) // nymax + 1

            xs = num.arange(nx) // ((max(2, nx) - 1.0) / 2.0)
            ys = num.arange(ny) // ((max(2, ny) - 1.0) / 2.0)

            xs -= num.mean(xs)
            ys -= num.mean(ys)

            fxs = num.tile(xs, ny)
            fys = num.repeat(ys, nx)

            data = []
            for target_code in target_codes:
                targets = target_codes_to_targets[target_code]
                target = targets[0]
                azi = source.azibazi_to(target)[0]
                dist = source.distance_to(target)
                x = dist * num.sin(num.deg2rad(azi))
                y = dist * num.cos(num.deg2rad(azi))
                data.append((x, y, dist))

            gxs, gys, dists = num.array(data, dtype=num.float).T

            iorder = num.argsort(dists)

            gxs = gxs[iorder]
            gys = gys[iorder]
            target_codes_sorted = [target_codes[ii] for ii in iorder]

            gxs -= num.mean(gxs)
            gys -= num.mean(gys)

            gmax = max(num.max(num.abs(gys)), num.max(num.abs(gxs)))
            if gmax == 0.0:
                gmax = 1.0

            gxs /= gmax
            gys /= gmax

            dists = num.sqrt(
                (fxs[num.newaxis, :] - gxs[:, num.newaxis]) ** 2
                + (fys[num.newaxis, :] - gys[:, num.newaxis]) ** 2
            )

            distmax = num.max(dists)

            availmask = num.ones(dists.shape[1], dtype=num.bool)
            frame_to_target_code = {}
            for itarget, target_code in enumerate(target_codes_sorted):
                iframe = num.argmin(num.where(availmask, dists[itarget], distmax + 1.0))
                availmask[iframe] = False
                iy, ix = num.unravel_index(iframe, (ny, nx))
                frame_to_target_code[iy, ix] = target_code

            figures = {}
            for iy in range(ny):
                for ix in range(nx):
                    if (iy, ix) not in frame_to_target_code:
                        continue

                    ixx = ix // nxmax
                    iyy = iy // nymax
                    if (iyy, ixx) not in figures:
                        figures[iyy, ixx] = plt.figure(
                            figsize=mpl_papersize("a4", "landscape")
                        )

                        figures[iyy, ixx].subplots_adjust(
                            left=0.03,
                            right=1.0 - 0.03,
                            bottom=0.06,
                            top=0.96,
                            wspace=0.20,
                            hspace=0.30,
                        )

                        figs.append(figures[iyy, ixx])

                    logger.debug("iyy %i, ixx %i" % (iyy, ixx))
                    logger.debug("iy %i, ix %i" % (iy, ix))
                    fig = figures[iyy, ixx]

                    target_code = frame_to_target_code[iy, ix]
                    domain_targets = target_codes_to_targets[target_code]
                    if len(domain_targets) > 1:
                        only_spectrum = False
                    else:
                        only_spectrum = True

                    for k_subf, target in enumerate(domain_targets):

                        syn_traces = all_syn_trs_target[target]
                        itarget = target_index[target]
                        result = bresults[itarget]

                        # get min max of all traces
                        key = target.codes[3]
                        amin, amax = trace.minmax(syn_traces, key=skey)[key]
                        # need target specific minmax
                        absmax = max(abs(amin), abs(amax))

                        ny_this = nymax  # min(ny, nymax)
                        nx_this = nxmax  # min(nx, nxmax)
                        i_this = (iy % ny_this) * nx_this + (ix % nx_this) + 1
                        logger.debug("i_this %i" % i_this)
                        logger.debug(
                            "Station {}".format(utility.list2string(target.codes))
                        )

                        if k_subf == 0:
                            # only create axes instances for first target
                            axes2 = fig.add_subplot(ny_this, nx_this, i_this)

                            space = 0.4
                            space_factor = 0.7 + space
                            axes2.set_axis_off()
                            axes2.set_ylim(-1.05 * space_factor, 1.05)

                            axes = axes2.twinx()
                            axes.set_axis_off()

                        if target.domain == "time":
                            ymin, ymax = -absmax * 1.5 * space_factor, absmax * 1.5
                            try:
                                axes.set_ylim(ymin, ymax)
                            except ValueError:
                                logger.debug(
                                    "These traces contain NaN or Inf open in snuffler?"
                                )
                                input("Press enter! Otherwise Ctrl + C")
                                from pyrocko.trace import snuffle

                                snuffle(syn_traces)

                            subplot_waveforms(
                                axes=axes,
                                axes2=axes2,
                                po=po,
                                result=result,
                                stdz_residual=stdz_residuals[target.nslcd_id_str],
                                target=target,
                                traces=syn_traces,
                                source=source,
                                var_reductions=all_var_reductions[target],
                                time_shifts=all_time_shifts[target],
                                time_shift_bounds=time_shift_bounds,
                                synth_plot_flag=synth_plot_flag,
                                absmax=absmax,
                                mode=composite._mode,
                                fontsize=fontsize,
                                syn_color=syn_color,
                                obs_color=obs_color,
                                time_shift_color=time_shift_color,
                                tap_color_edge=tap_color_edge,
                                tap_color_annot=tap_color_annot,
                            )

                        if target.domain == "spectrum":
                            subplot_spectrum(
                                axes=axes,
                                axes2=axes2,
                                po=po,
                                target=target,
                                traces=syn_traces,
                                result=result,
                                stdz_residual=stdz_residuals[target.nslcd_id_str],
                                synth_plot_flag=synth_plot_flag,
                                only_spectrum=only_spectrum,
                                var_reductions=all_var_reductions[target],
                                fontsize=fontsize,
                                syn_color=syn_color,
                                obs_color=obs_color,
                                misfit_color=misfit_color,
                                tap_color_annot=tap_color_annot,
                                ypad_factor=1.2,
                            )

                    scale_string = None

                    infos = []
                    if scale_string:
                        infos.append(scale_string)

                    infos.append(".".join(x for x in target.codes if x))
                    dist = source.distance_to(target)
                    azi = source.azibazi_to(target)[0]
                    infos.append(str_dist(dist))
                    infos.append("%.0f\u00B0" % azi)
                    # infos.append('%.3f' % gcms[itarget])
                    axes2.annotate(
                        "\n".join(infos),
                        xy=(0.0, 1.0),
                        xycoords="axes fraction",
                        xytext=(1.0, 1.0),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=fontsize,
                        fontstyle="normal",
                        zorder=10,
                    )

                    axes2.set_zorder(10)

            for (iyy, ixx), fig in figures.items():
                title = ".".join(x for x in cg if x)
                if len(figures) > 1:
                    title += " (%i/%i, %i/%i)" % (iyy + 1, nyy, ixx + 1, nxx)

                fig.suptitle(title, fontsize=fontsize_title)

        event_figs.append((event_idx, figs))

    return event_figs


def draw_seismic_fits(problem, po):

    if "seismic" not in list(problem.composites.keys()):
        raise TypeError("No seismic composite defined for this problem!")

    logger.info("Drawing Waveform fits ...")

    stage = Stage(
        homepath=problem.outfolder, backend=problem.config.sampler_config.backend
    )

    mode = problem.config.problem_config.mode

    if not po.reference:
        llk_str = po.post_llk
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model,
            stage_number=po.load_stage,
            load="trace",
            chains=[-1],
        )
    else:
        llk_str = "ref"

    outpath = os.path.join(
        problem.config.project_dir,
        mode,
        po.figure_dir,
        "waveforms_%s_%s_%i" % (stage.number, llk_str, po.nensemble),
    )

    if not os.path.exists(outpath) or po.force:
        event_figs = seismic_fits(problem, stage, po)
    else:
        logger.info("waveform plots exist. Use force=True for replotting!")
        return

    if po.outformat == "display":
        plt.show()
    else:
        for event_idx, figs in event_figs:
            event_outpath = "{}_{}".format(outpath, event_idx)
            logger.info("saving figures to %s" % event_outpath)
            if po.outformat == "pdf":
                with PdfPages(event_outpath + ".pdf") as opdf:
                    for fig in figs:
                        opdf.savefig(fig)
            else:
                for i, fig in enumerate(figs):
                    fig.savefig(
                        event_outpath + "_%i.%s" % (i, po.outformat), dpi=po.dpi
                    )


def point2array(point, varnames, idx_source=1, rpoint=None):
    """
    Concatenate values of point according to order of given varnames.
    """
    if point is not None:
        array = num.empty((len(varnames)), dtype="float64")
        for i, varname in enumerate(varnames):
            try:
                array[i] = point[varname][idx_source].ravel()
            except KeyError:  # in case fixed variable
                if rpoint:
                    array[i] = rpoint[varname][idx_source].ravel()
                else:
                    raise ValueError(
                        'Fixed Component "%s" no fixed value given!' % varname
                    )

        return array
    else:
        return None


def extract_mt_components(problem, po, include_magnitude=False):
    """
    Extract Moment Tensor components from problem results for plotting.
    """
    source_type = problem.config.problem_config.source_type
    n_sources = problem.config.problem_config.n_sources

    if source_type in ["MTSource", "MTQTSource"]:
        varnames = ["mnn", "mee", "mdd", "mne", "mnd", "med"]
    elif source_type in ["DCSource", "RectangularSource"]:
        varnames = ["strike", "dip", "rake"]
    else:
        raise ValueError('Plot is only supported for point "MTSource" and "DCSource"')

    if include_magnitude:
        varnames += ["magnitude"]

    if not po.reference:
        rpoint = None
        llk_str = po.post_llk
        stage = load_stage(
            problem, stage_number=po.load_stage, load="trace", chains=[-1]
        )

        list_m6s = []
        list_best_mts = []
        for idx_source in range(n_sources):
            n_mts = len(stage.mtrace)
            m6s = num.empty((n_mts, len(varnames)), dtype="float64")
            for i, varname in enumerate(varnames):
                try:
                    m6s[:, i] = (
                        stage.mtrace.get_values(varname, combine=True, squeeze=True)
                        .T[idx_source]
                        .ravel()
                    )

                except ValueError:  # if fixed value add that to the ensemble
                    rpoint = problem.get_random_point()
                    mtfield = num.full_like(
                        num.empty((n_mts), dtype=num.float64),
                        rpoint[varname][idx_source],
                    )
                    m6s[:, i] = mtfield

            if po.nensemble:
                logger.info("Drawing %i solutions from ensemble ..." % po.nensemble)
                csteps = float(n_mts) / po.nensemble
                idxs = num.floor(num.arange(0, n_mts, csteps)).astype("int32")
                m6s = m6s[idxs, :]
            else:
                logger.info("Drawing full ensemble ...")

            point = get_result_point(stage.mtrace, po.post_llk)
            best_mt = point2array(
                point, varnames=varnames, rpoint=rpoint, idx_source=idx_source
            )

            list_m6s.append(m6s)
            list_best_mts.append(best_mt)
    else:
        llk_str = "ref"
        point = po.reference
        list_best_mts = []
        list_m6s = []
        if source_type == "MTQTSource":
            composite = problem.composites[problem.config.problem_config.datatypes[0]]
            composite.point2sources(po.reference)
            for source in composite.sources:
                list_m6s.append([source.get_derived_parameters()[0:6]])
                list_best_mts.append(None)

        else:
            for idx_source in range(n_sources):
                list_m6s.append(
                    [
                        point2array(
                            point=po.reference, varnames=varnames, idx_source=idx_source
                        )
                    ]
                )
                list_best_mts.append(None)

    return list_m6s, list_best_mts, llk_str, point


def draw_ray_piercing_points_bb(
    ax,
    takeoff_angles_rad,
    azimuths_rad,
    polarities,
    nomask=False,
    markersize=5,
    size=1,
    position=(0, 0),
    transform=None,
    stations=None,
    projection="lambert",
):
    # overturn takeoff-angles above 90 deg
    toa_idx = takeoff_angles_rad >= (num.pi / 2.0)
    takeoff_angles_rad[toa_idx] = num.pi - takeoff_angles_rad[toa_idx]

    # project stations to coordinate system of beachball
    rtp = num.vstack(
        [num.ones_like(takeoff_angles_rad), takeoff_angles_rad, azimuths_rad]
    ).T
    points = beachball.numpy_rtp2xyz(rtp)
    x, y = beachball.project(points, projection=projection).T
    x = size * x + position[1]
    y = size * y + position[0]

    if not nomask:
        xp, yp = x[polarities >= 0], y[polarities >= 0]
        xt, yt = x[polarities < 0], y[polarities < 0]
        ax.plot(
            yp,
            xp,
            "D",
            ms=markersize,
            mew=0.5,
            mec="black",
            mfc="white",
            transform=transform,
        )
        ax.plot(
            yt,
            xt,
            "s",
            ms=markersize,
            mew=0.5,
            mec="white",
            mfc="black",
            transform=transform,
        )
    else:
        ax.scatter(x, y, markersize, polarities, transform=transform)

    if stations is not None:
        if len(stations) != x.size:
            raise ValueError("Number of stations is inconsistent with polarity data!")

        for i_s, station in enumerate(stations):

            ax.text(
                y[i_s],
                x[i_s],
                "{}.{}".format(
                    station.network,
                    station.station,
                    # takeoff_angles_rad[i_s] * 180 / num.pi,
                    # azimuths_rad[i_s] * 180 / num.pi,
                ),  # polarities[i_s]),
                color="red",
                fontsize=5,
                va="top",
                ha="right",
                rotation=45,
            )


def lower_focalsphere_angles(grid_resolution, projection):

    nx = grid_resolution
    ny = grid_resolution

    x = num.linspace(-1.0, 1.0, nx)
    y = num.linspace(-1.0, 1.0, ny)

    vecs2 = num.zeros((nx * ny, 2), dtype=num.float64)
    vecs2[:, 0] = num.tile(x, ny)
    vecs2[:, 1] = num.repeat(y, nx)

    ii_ok = vecs2[:, 0] ** 2 + vecs2[:, 1] ** 2 <= 1.0
    amps = num.full(nx * ny, num.nan, dtype=num.float64)

    amps[ii_ok] = 0.0

    vp = num.array([1, 0, 0])
    vt = num.array([0, 1, 0])
    vn = num.array([0, 0, 1])

    vecs3_ok = beachball.inverse_project(vecs2[ii_ok, :], projection)

    to_e = num.vstack((vp, vt, vn))

    vecs_e = num.dot(to_e, vecs3_ok.T).T
    rtp = beachball.numpy_xyz2rtp(vecs_e)

    atheta, aphi = rtp[:, 1], rtp[:, 2]

    if 0:
        atheta_re = num.zeros_like(amps)
        atheta_re[ii_ok] = atheta

        aphi_re = num.zeros_like(amps)
        aphi_re[ii_ok] = aphi
        atheta_re = num.reshape(atheta_re * 180 / num.pi, (ny, nx))
        aphi_re = num.reshape(aphi_re * 180 / num.pi, (ny, nx)).T

        print("theta", atheta_re.min(), atheta_re.max())
        print("phi", aphi_re.min(), aphi_re.max())

        fig, axs = plt.subplots(
            nrows=1, ncols=2, figsize=mpl_papersize("a6", "landscape")
        )

        im1 = axs[0].imshow(atheta_re)
        plt.colorbar(im1)
        im2 = axs[1].imshow(aphi_re, origin="lower")
        plt.colorbar(im2)
        plt.show()
    return amps, atheta, aphi, ii_ok, x, y


def mts2amps(
    mts,
    projection,
    beachball_type,
    grid_resolution=200,
    mask=True,
    view="top",
    wavename="any_P",
):

    n_balls = len(mts)
    nx = ny = grid_resolution

    amps, takeoff_angles_rad, azimuths_rad, ii_ok, x, y = lower_focalsphere_angles(
        grid_resolution, projection
    )

    for mt in mts:

        mt = beachball.deco_part(mt, mt_type=beachball_type, view=view)

        radiation_weights = calculate_radiation_weights(
            takeoff_angles_rad, azimuths_rad, wavename=wavename
        )

        m9 = mt.m()

        if isinstance(m9, num.matrix):
            m9 = m9.A

        m0_unscaled = num.sqrt(num.sum(m9**2)) / SQRT2
        m9 /= m0_unscaled
        amps_ok = radiation_weights.T.dot(to6(m9))

        if mask:
            amps_ok[amps_ok > 0] = 1.0
            amps_ok[amps_ok < 0] = 0.0

        amps[ii_ok] += amps_ok

    return num.reshape(amps, (ny, nx)) / n_balls, x, y


def plot_fuzzy_beachball_mpl_pixmap(
    mts,
    axes,
    best_mt=None,
    beachball_type="deviatoric",
    wavename="any_P",
    position=(0.0, 0.0),
    size=None,
    zorder=0,
    color_t="red",
    color_p="white",
    edgecolor="black",
    best_color="red",
    linewidth=2,
    alpha=1.0,
    projection="lambert",
    size_units="data",
    grid_resolution=100,
    method="imshow",
    view="top",
):
    """
    Plot fuzzy beachball from a list of given MomentTensors

    :param mts: list of
        :py:class:`pyrocko.moment_tensor.MomentTensor` object or an
        array or sequence which can be converted into an MT object
    :param best_mt: :py:class:`pyrocko.moment_tensor.MomentTensor` object or
        an array or sequence which can be converted into an MT object
        of most likely or minimum misfit solution to extra highlight
    :param best_color: mpl color for best MomentTensor edges,
        polygons are not plotted

    See plot_beachball_mpl for other arguments
    """
    from matplotlib.colors import LinearSegmentedColormap

    if size_units == "points":
        raise beachball.BeachballError(
            'size_units="points" not supported in ' "plot_fuzzy_beachball_mpl_pixmap"
        )

    transform, position, size = beachball.choose_transform(
        axes, size_units, position, size
    )

    amps, x, y = mts2amps(
        mts,
        grid_resolution=grid_resolution,
        projection=projection,
        beachball_type=beachball_type,
        mask=True,
        wavename=wavename,
        view=view,
    )

    ncolors = 256
    cmap = LinearSegmentedColormap.from_list("dummy", [color_p, color_t], N=ncolors)

    levels = num.linspace(0, 1.0, ncolors)
    if method == "contourf":
        axes.contourf(
            position[0] + y * size,
            position[1] + x * size,
            amps.T,
            levels=levels,
            cmap=cmap,
            transform=transform,
            zorder=zorder,
            alpha=alpha,
        )

    elif method == "imshow":
        axes.imshow(
            amps.T,
            extent=(
                position[0] + y[0] * size,
                position[0] + y[-1] * size,
                position[1] - x[0] * size,
                position[1] - x[-1] * size,
            ),
            cmap=cmap,
            transform=transform,
            zorder=zorder - 0.1,
            alpha=alpha,
        )
    else:
        assert False, "invalid `method` argument"

    # draw optimum edges
    if best_mt is not None:
        best_amps, bx, by = mts2amps(
            [best_mt],
            grid_resolution=grid_resolution,
            projection=projection,
            wavename=wavename,
            beachball_type=beachball_type,
            mask=False,
        )

        axes.contour(
            position[0] + by * size,
            position[1] + bx * size,
            best_amps.T,
            levels=[0.0],
            colors=[best_color],
            linewidths=linewidth,
            transform=transform,
            zorder=zorder,
            alpha=alpha,
        )

    phi = num.linspace(0.0, 2 * PI, 361)
    x = num.cos(phi)
    y = num.sin(phi)
    axes.plot(
        position[0] + y * size,
        position[1] + x * size,
        linewidth=linewidth,
        color=edgecolor,
        transform=transform,
        zorder=zorder,
        alpha=alpha,
    )


def draw_fuzzy_beachball(problem, po):

    if po.load_stage is None:
        po.load_stage = -1

    list_m6s, list_best_mt, llk_str, point = extract_mt_components(problem, po)

    logger.info("Drawing Fuzzy Beachball ...")

    kwargs = {
        "beachball_type": "full",
        "size": 8,
        "size_units": "data",
        "linewidth": 2.0,
        "alpha": 1.0,
        "position": (5, 5),
        "color_t": "black",
        "edgecolor": "black",
        "projection": "lambert",
        "zorder": 0,
        "grid_resolution": 400,
    }

    if "polarity" in problem.config.problem_config.datatypes:
        composite = problem.composites["polarity"]
        composite.point2sources(point)
        wavenames = [pmap.config.name for pmap in composite.wavemaps]
    else:
        wavenames = ["any_P"]

    for k_pamp, wavename in enumerate(wavenames):

        for idx_source, (m6s, best_mt) in enumerate(zip(list_m6s, list_best_mt)):
            outpath = os.path.join(
                problem.outfolder,
                po.figure_dir,
                "fuzzy_beachball_%i_%s_%i_%s_%i.%s"
                % (
                    po.load_stage,
                    llk_str,
                    po.nensemble,
                    wavename,
                    idx_source,
                    po.outformat,
                ),
            )

            if not os.path.exists(outpath) or po.force or po.outformat == "display":
                fig = plt.figure(figsize=(4.0, 4.0))
                fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
                axes = fig.add_subplot(1, 1, 1)

                transform, position, size = beachball.choose_transform(
                    axes, kwargs["size_units"], kwargs["position"], kwargs["size"]
                )

                plot_fuzzy_beachball_mpl_pixmap(
                    m6s,
                    axes,
                    best_mt=best_mt,
                    best_color="white",
                    wavename=wavename,
                    **kwargs
                )

                if best_mt is not None:
                    best_amps, bx, by = mts2amps(
                        [best_mt],
                        grid_resolution=kwargs["grid_resolution"],
                        projection=kwargs["projection"],
                        beachball_type=kwargs["beachball_type"],
                        wavename=wavename,
                        mask=False,
                    )

                    axes.contour(
                        position[0] + by * size,
                        position[1] + bx * size,
                        best_amps.T,
                        levels=[0.0],
                        colors=["black"],
                        linestyles="dashed",
                        linewidths=kwargs["linewidth"],
                        transform=transform,
                        zorder=kwargs["zorder"],
                        alpha=kwargs["alpha"],
                    )

                if "polarity" in problem.config.problem_config.datatypes:
                    pmap = composite.wavemaps[k_pamp]
                    source = composite.sources[pmap.config.event_idx]
                    pmap.update_targets(
                        composite.engine,
                        source,
                        always_raytrace=composite.config.gf_config.always_raytrace,
                    )
                    draw_ray_piercing_points_bb(
                        axes,
                        pmap.get_takeoff_angles_rad(),
                        pmap.get_azimuths_rad(),
                        pmap._prepared_data,
                        stations=pmap.stations,
                        size=size,
                        position=position,
                        transform=transform,
                    )

                axes.set_xlim(0.0, 10.0)
                axes.set_ylim(0.0, 10.0)
                axes.set_axis_off()

                if not po.outformat == "display":
                    logger.info("saving figure to %s" % outpath)
                    fig.savefig(outpath, dpi=po.dpi)
                else:
                    plt.show()

            else:
                logger.info("Plot already exists! Please use --force to overwrite!")


def fuzzy_mt_decomposition(axes, list_m6s, labels=None, colors=None, fontsize=12):
    """
    Plot fuzzy moment tensor decompositions for list of mt ensembles.
    """
    from pymc3 import quantiles
    from pyrocko.moment_tensor import MomentTensor

    logger.info("Drawing Fuzzy MT Decomposition ...")

    # beachball kwargs
    kwargs = {
        "beachball_type": "full",
        "size": 1.0,
        "size_units": "data",
        "edgecolor": "black",
        "linewidth": 1,
        "grid_resolution": 200,
    }

    def get_decomps(source_vals):

        isos = []
        dcs = []
        clvds = []
        devs = []
        tots = []
        for val in source_vals:
            m = MomentTensor.from_values(val)
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
        labels = ["Ensemble"] + ([None] * (nlines - 1))

    lines = []
    for i, (label, m6s, color) in enumerate(zip(labels, list_m6s, colors)):
        if color is None:
            color = mpl_graph_color(i)

        lines.append((label, m6s, color))

    magnitude_full_max = max(m6s.mean(axis=0)[-1] for (_, m6s, _) in lines)

    for xpos, label in [
        (0.0, "Full"),
        (2.0, "Isotropic"),
        (4.0, "Deviatoric"),
        (6.0, "CLVD"),
        (8.0, "DC"),
    ]:

        axes.annotate(
            label,
            xy=(1 + xpos, nlines_max),
            xycoords="data",
            xytext=(0.0, 0.0),
            textcoords="offset points",
            ha="center",
            va="center",
            color="black",
            fontsize=fontsize,
        )

    for i, (label, m6s, color_t) in enumerate(lines):
        ypos = nlines_max - (i * yscale) - 1.0
        mean_magnitude = m6s.mean(0)[-1]
        size0 = mean_magnitude / magnitude_full_max

        isos, dcs, clvds, devs, tots = get_decomps(m6s)
        axes.annotate(
            label,
            xy=(-2.0, ypos),
            xycoords="data",
            xytext=(0.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="black",
            fontsize=fontsize,
        )

        for xpos, decomp, ops in [
            (0.0, tots, "-"),
            (2.0, isos, "="),
            (4.0, devs, "="),
            (6.0, clvds, "+"),
            (8.0, dcs, None),
        ]:

            ratios = num.array([comp[1] for comp in decomp])
            ratio = ratios.mean()
            ratios_diff = ratios.max() - ratios.min()

            ratios_qu = quantiles(ratios * 100.0)
            mt_parts = [comp[2] for comp in decomp]

            if ratio > 1e-4:
                try:
                    size = num.sqrt(ratio) * 0.95 * size0
                    kwargs["position"] = (1.0 + xpos, ypos)
                    kwargs["size"] = size
                    kwargs["color_t"] = color_t
                    beachball.plot_fuzzy_beachball_mpl_pixmap(
                        mt_parts, axes, best_mt=None, **kwargs
                    )

                    if ratios_diff > 0.0:
                        label = "{:03.1f}-{:03.1f}%".format(
                            ratios_qu[2.5], ratios_qu[97.5]
                        )
                    else:
                        label = "{:03.1f}%".format(ratios_qu[2.5])

                    axes.annotate(
                        label,
                        xy=(1.0 + xpos, ypos - 0.65),
                        xycoords="data",
                        xytext=(0.0, 0.0),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=fontsize - 2,
                    )

                except beachball.BeachballError as e:
                    logger.warn(str(e))

                    axes.annotate(
                        "ERROR",
                        xy=(1.0 + xpos, ypos),
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=fontsize,
                    )

            else:
                axes.annotate(
                    "N/A",
                    xy=(1.0 + xpos, ypos),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize,
                )

                label = "{:03.1f}%".format(0.0)
                axes.annotate(
                    label,
                    xy=(1.0 + xpos, ypos - 0.65),
                    xycoords="data",
                    xytext=(0.0, 0.0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize - 2,
                )

            if ops is not None:
                axes.annotate(
                    ops,
                    xy=(2.0 + xpos, ypos),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fontsize,
                )

    axes.axison = False
    axes.set_xlim(-2.25, 9.75)
    axes.set_ylim(-0.7, nlines_max + 0.5)
    axes.set_axis_off()


def draw_fuzzy_mt_decomposition(problem, po):

    fontsize = 10

    n_sources = problem.config.problem_config.n_sources

    if po.load_stage is None:
        po.load_stage = -1

    list_m6s, _, llk_str, _ = extract_mt_components(problem, po, include_magnitude=True)

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        "fuzzy_mt_decomposition_%i_%s_%i.%s"
        % (po.load_stage, llk_str, po.nensemble, po.outformat),
    )

    if not os.path.exists(outpath) or po.force or po.outformat == "display":

        height = 1.5 + (n_sources - 1) * 0.65
        fig = plt.figure(figsize=(6.0, height))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97)
        axes = fig.add_subplot(1, 1, 1)

        fuzzy_mt_decomposition(axes, list_m6s=list_m6s, fontsize=fontsize)

        if not po.outformat == "display":
            logger.info("saving figure to %s" % outpath)
            fig.savefig(outpath, dpi=po.dpi)
        else:
            plt.show()

    else:
        logger.info("Plot already exists! Please use --force to overwrite!")


def draw_hudson(problem, po):
    """
    Modified from grond. Plot the hudson graph for the reference event(grey)
    and the best solution (red beachball).
    Also a random number of models from the
    selected stage are plotted as smaller beachballs on the hudson graph.
    """

    from numpy import random
    from pyrocko import moment_tensor as mtm
    from pyrocko.plot import beachball, hudson

    if po.load_stage is None:
        po.load_stage = -1

    list_m6s, list_best_mts, llk_str, _ = extract_mt_components(problem, po)

    logger.info("Drawing Hudson plot ...")

    fontsize = 12
    beachball_type = "full"
    color = "red"
    markersize = fontsize * 1.5
    markersize_small = markersize * 0.2
    beachballsize = markersize
    beachballsize_small = beachballsize * 0.5

    for idx_source, (m6s, best_mt) in enumerate(zip(list_m6s, list_best_mts)):
        fig = plt.figure(figsize=(4.0, 4.0))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        axes = fig.add_subplot(1, 1, 1)
        hudson.draw_axes(axes)

        data = []
        for m6 in m6s:
            mt = mtm.as_mt(m6)
            u, v = hudson.project(mt)

            if random.random() < 0.05:
                try:
                    beachball.plot_beachball_mpl(
                        mt,
                        axes,
                        beachball_type=beachball_type,
                        position=(u, v),
                        size=beachballsize_small,
                        color_t="black",
                        alpha=0.5,
                        zorder=1,
                        linewidth=0.25,
                    )
                except beachball.BeachballError as e:
                    logger.warn(str(e))

            else:
                data.append((u, v))

        if data:
            u, v = num.array(data).T
            axes.plot(
                u,
                v,
                "o",
                color=color,
                ms=markersize_small,
                mec="none",
                mew=0,
                alpha=0.25,
                zorder=0,
            )

        if best_mt is not None:
            mt = mtm.as_mt(best_mt)
            u, v = hudson.project(mt)

            try:
                beachball.plot_beachball_mpl(
                    mt,
                    axes,
                    beachball_type=beachball_type,
                    position=(u, v),
                    size=beachballsize,
                    color_t=color,
                    alpha=0.5,
                    zorder=2,
                    linewidth=0.25,
                )
            except beachball.BeachballError as e:
                logger.warn(str(e))

        if isinstance(problem.event.moment_tensor, mtm.MomentTensor):
            mt = problem.event.moment_tensor
            u, v = hudson.project(mt)

            if not po.reference:
                try:
                    beachball.plot_beachball_mpl(
                        mt,
                        axes,
                        beachball_type=beachball_type,
                        position=(u, v),
                        size=beachballsize,
                        color_t="grey",
                        alpha=0.5,
                        zorder=2,
                        linewidth=0.25,
                    )
                    logger.info("drawing reference event in grey ...")
                except beachball.BeachballError as e:
                    logger.warn(str(e))
        else:
            logger.info(
                "No reference event moment tensor information given, "
                "skipping drawing ..."
            )

        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            "hudson_%i_%s_%i_%i.%s"
            % (po.load_stage, llk_str, po.nensemble, idx_source, po.outformat),
        )

        if not os.path.exists(outpath) or po.force or po.outformat == "display":

            if not po.outformat == "display":
                logger.info("saving figure to %s" % outpath)
                fig.savefig(outpath, dpi=po.dpi)
            else:
                plt.show()

        else:
            logger.info("Plot already exists! Please use --force to overwrite!")


def draw_data_stations(
    gmt, stations, data, dist, data_cpt=None, scale_label=None, *args
):
    """
    Draw MAP time-shifts at station locations as colored triangles
    """
    miny = data.min()
    maxy = data.max()
    bound = num.ceil(max(num.abs(miny), maxy))

    if data_cpt is None:
        data_cpt = "/tmp/tempfile.cpt"

        gmt.makecpt(
            C="blue,white,red",
            Z=True,
            T="%g/%g" % (-bound, bound),
            out_filename=data_cpt,
            suppress_defaults=True,
        )

    for i, station in enumerate(stations):
        logger.debug("%s, %f" % (station.station, data[i]))

    st_lons = [station.lon for station in stations]
    st_lats = [station.lat for station in stations]

    gmt.psxy(in_columns=(st_lons, st_lats, data.tolist()), C=data_cpt, *args)

    if dist > 30.0:
        D = "x1.25c/0c+w5c/0.5c+jMC+h"
        F = False
    else:
        D = "x5.5c/4.1c+w5c/0.5c+jMC+h"
        F = "+gwhite"

    if scale_label:
        # add a colorbar
        gmt.psscale(
            B="xa%s +l %s" % (num.floor(bound), scale_label), D=D, F=F, C=data_cpt
        )
    else:
        logger.info('Not plotting scale as "scale_label" is None')


def draw_events(gmt, events, *args, **kwargs):

    ev_lons = [ev.lon for ev in events]
    ev_lats = [ev.lat for ev in events]

    gmt.psxy(in_columns=(ev_lons, ev_lats), *args, **kwargs)


def gmt_station_map_azimuthal(
    gmt,
    stations,
    event,
    data_cpt=None,
    data=None,
    max_distance=90,
    width=20,
    bin_width=15,
    fontsize=12,
    font="1",
    plot_names=True,
    scale_label="time-shifts [s]",
):
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
        path to gmt `*.cpt` file for coloring
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

    J_basemap = "E0/-90/%s/%i" % (max_distance, width)
    J_location = "E%s/%s/%s/%i" % (event.lon, event.lat, max_distance, width)
    R_location = "0/360/-90/0"

    gmt.psbasemap(
        R=R_location, J="S0/-90/90/%i" % width, B="xa%sf%s" % (bin_width * 2, bin_width)
    )
    gmt.pscoast(R="g", J=J_location, D="c", G="darkgrey")

    # plotting equal distance circles
    bargs = ["-Bxg%f" % bin_width, "-Byg%f" % (2 * bin_width)]
    gmt.psbasemap(R="g", J=J_basemap, *bargs)

    if data is not None:
        draw_data_stations(
            gmt,
            stations,
            data,
            max_distance,
            data_cpt,
            scale_label,
            *("-J%s" % J_location, "-R%s" % R_location, "-St14p")
        )
    else:
        st_lons = [station.lon for station in stations]
        st_lats = [station.lat for station in stations]

        gmt.psxy(
            R=R_location, J=J_location, in_columns=(st_lons, st_lats), G="red", S="t14p"
        )

    if plot_names:
        rows = []
        alignment = "TC"
        for st in stations:
            if gmt.is_gmt5():
                row = (
                    st.lon,
                    st.lat,
                    "%i,%s,%s" % (fontsize, font, "black"),
                    alignment,
                    "{}.{}".format(st.network, st.station),
                )
                farg = ["-F+f+j"]
            else:
                raise gmtpy.GmtPyError("Only GMT version 5.x supported!")

            rows.append(row)

        gmt.pstext(in_rows=rows, R=R_location, J=J_location, N=True, *farg)

    draw_events(
        gmt,
        [event],
        *("-J%s" % J_location, "-R%s" % R_location),
        **dict(G="orange", S="a14p")
    )


def draw_station_map_gmt(problem, po):
    """
    Draws distance dependent for teleseismic vs regional/local setups
    """

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError("GMT needs to be installed for station_map plot!")

    if po.outformat == "svg":
        raise NotImplementedError("SVG format is not supported for this plot!")

    ts = "time_shift"
    if ts in po.varnames:
        logger.info("Plotting time-shifts on station locations")
        stage = load_stage(
            problem, stage_number=po.load_stage, load="trace", chains=[-1]
        )

        point = get_result_point(stage.mtrace, po.post_llk)
        value_string = "%i" % po.load_stage
    else:
        point = None
        value_string = "0"
        if len(po.varnames) > 0:
            raise ValueError(
                "Requested variables %s is not supported for plotting!"
                "Supported: %s" % (utility.list2string(po.varnames), ts)
            )

    fontsize = 12
    font = "1"
    bin_width = 15  # major grid and tick increment in [deg]
    h = 15  # outsize in cm
    w = h - 5

    logger.info("Drawing Station Map ...")
    for datatype in problem.config.problem_config.datatypes:
        sc = problem.composites[datatype]
        if datatype != "geodetic":
            wmaps = sc.wavemaps
        else:
            wmaps = []

        event = problem.config.event

        gmtconfig = get_gmt_config(gmtpy, h=h, w=h, fontsize=fontsize)
        gmtconfig["MAP_LABEL_OFFSET"] = "4p"
        for wmap in wmaps:
            outpath = os.path.join(
                problem.outfolder,
                po.figure_dir,
                "station_map_%s_%i_%s.%s"
                % (wmap.name, wmap.mapnumber, value_string, po.outformat),
            )

            dist = max(wmap.get_distances_deg())
            if not os.path.exists(outpath) or po.force:

                if point:
                    time_shifts = extract_time_shifts(point, sc.hierarchicals, wmap)
                else:
                    time_shifts = None

                if dist > 30:
                    logger.info(
                        "Using equidistant azimuthal projection for"
                        " teleseismic setup of wavemap %s." % wmap._mapid
                    )

                    gmt = gmtpy.GMT(config=gmtconfig)
                    gmt_station_map_azimuthal(
                        gmt,
                        wmap.stations,
                        event,
                        data=time_shifts,
                        max_distance=dist,
                        width=w,
                        bin_width=bin_width,
                        fontsize=fontsize,
                        font=font,
                    )

                    gmt.save(outpath, resolution=po.dpi, size=w)

                else:
                    logger.info(
                        "Using equidistant projection for regional setup "
                        "of wavemap %s." % wmap._mapid
                    )
                    from pyrocko import orthodrome as otd
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
                        gmt_config=gmtconfig,
                    )

                    if time_shifts:
                        sargs = m.jxyr + ["-St14p"]
                        draw_data_stations(
                            m.gmt,
                            wmap.stations,
                            time_shifts,
                            dist,
                            data_cpt=None,
                            scale_label="time shifts [s]",
                            *sargs
                        )

                        for st in wmap.stations:
                            text = "{}.{}".format(st.network, st.station)
                            m.add_label(lat=st.lat, lon=st.lon, text=text)
                    else:
                        m.add_stations(
                            wmap.stations, psxy_style=dict(S="t14p", G="red")
                        )

                    draw_events(m.gmt, [event], *m.jxyr, **dict(G="yellow", S="a14p"))
                    m.save(outpath, resolution=po.dpi, oversample=2.0)

                logger.info("saving figure to %s" % outpath)
            else:
                logger.info("Plot exists! Use --force to overwrite!")


def draw_lune_plot(problem, po):

    if po.outformat == "svg":
        raise NotImplementedError("SVG format is not supported for this plot!")

    if problem.config.problem_config.source_type != "MTQTSource":
        TypeError("Lune plot is only supported for the MTQTSource!")

    if po.load_stage is None:
        po.load_stage = -1

    stage = load_stage(problem, stage_number=po.load_stage, load="trace", chains=[-1])
    n_mts = len(stage.mtrace)

    n_sources = problem.config.problem_config.n_sources

    for idx_source in range(n_sources):
        result_ensemble = {}
        for varname in ["v", "w"]:
            try:
                result_ensemble[varname] = (
                    stage.mtrace.get_values(varname, combine=True, squeeze=True)
                    .T[idx_source]
                    .ravel()
                )
            except ValueError:  # if fixed value add that to the ensemble
                rpoint = problem.get_random_point()
                result_ensemble[varname] = num.full_like(
                    num.empty((n_mts), dtype=num.float64), rpoint[varname][idx_source]
                )

        if po.reference:
            reference_v_tape = po.reference["v"][idx_source]
            reference_w_tape = po.reference["w"][idx_source]
            llk_str = "ref"
        else:
            reference_v_tape = None
            reference_w_tape = None
            llk_str = po.post_llk

        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            "lune_%i_%s_%i_%i.%s"
            % (po.load_stage, llk_str, po.nensemble, idx_source, po.outformat),
        )

        if po.nensemble > 1:
            logger.info("Plotting selected ensemble as nensemble > 1 ...")
            selected = num.linspace(0, n_mts, po.nensemble, dtype="int", endpoint=False)
            v_tape = result_ensemble["v"][selected]
            w_tape = result_ensemble["w"][selected]
        else:
            logger.info("Plotting whole posterior ...")
            v_tape = result_ensemble["v"]
            w_tape = result_ensemble["w"]

        if not os.path.exists(outpath) or po.force or po.outformat == "display":
            logger.info("Drawing Lune plot ...")
            gmt = lune_plot(
                v_tape=v_tape,
                w_tape=w_tape,
                reference_v_tape=reference_v_tape,
                reference_w_tape=reference_w_tape,
            )

            logger.info("saving figure to %s" % outpath)
            gmt.save(outpath, resolution=300, size=10)
        else:
            logger.info("Plot exists! Use --force to overwrite!")


def lune_plot(v_tape=None, w_tape=None, reference_v_tape=None, reference_w_tape=None):

    from beat.sources import v_to_gamma, w_to_delta

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError("GMT needs to be installed for lune_plot!")

    fontsize = 14
    font = "1"

    def draw_lune_arcs(gmt, R, J):

        lons = [30.0, -30.0, 30.0, -30.0]
        lats = [54.7356, 35.2644, -35.2644, -54.7356]

        gmt.psxy(in_columns=(lons, lats), N=True, W="1p,black", R=R, J=J)

    def draw_lune_points(gmt, R, J, labels=True):

        lons = [0.0, -30.0, -30.0, -30.0, 0.0, 30.0, 30.0, 30.0, 0.0]
        lats = [-90.0, -54.7356, 0.0, 35.2644, 90.0, 54.7356, 0.0, -35.2644, 0.0]
        annotations = ["-ISO", "", "+CLVD", "+LVD", "+ISO", "", "-CLVD", "-LVD", "DC"]
        alignments = ["TC", "TC", "RM", "RM", "BC", "BC", "LM", "LM", "TC"]

        gmt.psxy(in_columns=(lons, lats), N=True, S="p6p", W="1p,0", R=R, J=J)

        rows = []
        if labels:
            farg = ["-F+f+j"]
            for lon, lat, text, align in zip(lons, lats, annotations, alignments):

                rows.append(
                    (lon, lat, "%i,%s,%s" % (fontsize, font, "black"), align, text)
                )

            gmt.pstext(in_rows=rows, N=True, R=R, J=J, D="j5p", *farg)

    def draw_lune_kde(gmt, v_tape, w_tape, grid_size=(200, 200), R=None, J=None):
        def check_fixed(a, varname):
            if a.std() < 0.1:
                logger.info(
                    'Spread of variable "%s" is %f, which is below necessary'
                    " width to estimate a spherical kde, adding some jitter to"
                    " make kde estimate possible" % (varname, a.std())
                )
                a += num.random.normal(loc=0.0, scale=0.05, size=a.size)

        gamma = num.rad2deg(v_to_gamma(v_tape))  # lune longitude [rad]
        delta = num.rad2deg(w_to_delta(w_tape))  # lune latitude [rad]

        check_fixed(gamma, varname="v")
        check_fixed(delta, varname="w")

        lats_vec, lats_inc = num.linspace(-90.0, 90.0, grid_size[0], retstep=True)
        lons_vec, lons_inc = num.linspace(-30.0, 30.0, grid_size[1], retstep=True)
        lons, lats = num.meshgrid(lons_vec, lats_vec)

        kde_vals, _, _ = spherical_kde_op(
            lats0=delta, lons0=gamma, lons=lons, lats=lats, grid_size=grid_size
        )
        Tmin = num.min([0.0, kde_vals.min()])
        Tmax = num.max([0.0, kde_vals.max()])

        cptfilepath = "/tmp/tempfile.cpt"
        gmt.makecpt(
            C="white,yellow,orange,red,magenta,violet",
            Z=True,
            D=True,
            T="%f/%f" % (Tmin, Tmax),
            out_filename=cptfilepath,
            suppress_defaults=True,
        )

        grdfile = gmt.tempfilename()
        gmt.xyz2grd(
            G=grdfile,
            R=R,
            I="%f/%f" % (lons_inc, lats_inc),
            in_columns=(lons.ravel(), lats.ravel(), kde_vals.ravel()),  # noqa
            out_discard=True,
        )

        gmt.grdimage(grdfile, R=R, J=J, C=cptfilepath)

        # gmt.pscontour(
        #    in_columns=(lons.ravel(), lats.ravel(),  kde_vals.ravel()),
        #    R=R, J=J, I=True, N=True, A=True, C=cptfilepath)
        # -Ctmp_$out.cpt -I -N -A- -O -K >> $ps

    def draw_reference_lune(gmt, R, J, reference_v_tape, reference_w_tape):

        gamma = num.rad2deg(v_to_gamma(reference_v_tape))  # lune longitude [rad]
        delta = num.rad2deg(w_to_delta(reference_w_tape))  # lune latitude [rad]

        gmt.psxy(
            in_rows=[(float(gamma), float(delta))],
            N=True,
            G="blue",
            W="1p,black",
            S="p3p",
            R=R,
            J=J,
        )

    h = 20.0
    w = h / 1.9

    gmtconfig = get_gmt_config(gmtpy, h=h, w=w)
    bin_width = 15  # tick increment

    J = "H0/%f" % (w - 5.0)
    R = "-30/30/-90/90"
    B = "f%ig%i/f%ig%i" % (bin_width, bin_width, bin_width, bin_width)
    # range_arg="-T${zmin}/${zmax}/${dz}"

    gmt = gmtpy.GMT(config=gmtconfig)

    draw_lune_kde(gmt, v_tape=v_tape, w_tape=w_tape, grid_size=(701, 301), R=R, J=J)
    gmt.psbasemap(R=R, J=J, B=B)
    draw_lune_arcs(gmt, R=R, J=J)
    draw_lune_points(gmt, R=R, J=J)

    if reference_v_tape is not None:
        draw_reference_lune(
            gmt,
            R=R,
            J=J,
            reference_v_tape=reference_v_tape,
            reference_w_tape=reference_w_tape,
        )

    return gmt
