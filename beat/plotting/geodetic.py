import copy
import logging
import math
import os

from scipy import stats

import numpy as num
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MaxNLocator
from pymc3.plots.utils import make_2d
from pyrocko import gmtpy
from pyrocko import orthodrome as otd
from pyrocko.cake_plot import light
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.plot import AutoScaler, mpl_graph_color, mpl_papersize, nice_value

from beat import utility
from beat.config import ffi_mode_str
from beat.models import Stage

from .common import (
    format_axes,
    get_gmt_colorstring_from_mpl,
    get_latlon_ratio,
    get_nice_plot_bounds,
    get_result_point,
    km,
    cbtick,
    plot_inset_hist,
    scale_axes,
    set_anchor,
    plot_covariances,
    get_weights_point,
)

logger = logging.getLogger("plotting.geodetic")


def map_displacement_grid(displacements, scene):
    arr = num.full_like(scene.displacement, fill_value=num.nan)
    qt = scene.quadtree

    for syn_v, l in zip(displacements, qt.leaves):
        arr[l._slice_rows, l._slice_cols] = syn_v

    arr[scene.displacement_mask] = num.nan
    return arr


def shaded_displacements(
    displacement,
    shad_data,
    cmap="RdBu",
    shad_lim=(0.4, 0.98),
    tick_step=0.01,
    contrast=1.0,
    mask=None,
    data_limits=(-0.5, 0.5),
):
    """
    Map color data (displacement) on shaded relief.
    """

    from matplotlib.cm import ScalarMappable
    from scipy.ndimage import convolve as im_conv

    # Light source from somewhere above - psychologically the best choice
    # from upper left
    ramp = num.array([[1, 0], [0, -1.0]]) * contrast

    # convolution of two 2D arrays
    shad = im_conv(shad_data * km, ramp.T)
    shad *= -1.0

    # if there are strong artificial edges in the data, shades get
    # dominated by them. Cutting off the largest and smallest 2% of
    # shades helps
    percentile2 = num.quantile(shad, 0.02)
    percentile98 = num.quantile(shad, 0.98)
    shad[shad > percentile98] = percentile98
    shad[shad < percentile2] = percentile2

    # normalize shading
    shad -= num.nanmin(shad)
    shad /= num.nanmax(shad)

    if mask is not None:
        shad[mask] = num.nan

    # reduce range to balance gray color
    shad *= shad_lim[1] - shad_lim[0]
    shad += shad_lim[0]

    # create ticks for plotting - real values for the labels
    # and their position in normed data for the ticks
    if data_limits is None:
        data_max = num.nanmax(num.abs(displacement))
        data_limits = (-data_max, data_max)
    displ_min, displ_max = data_limits

    # Combine color and shading
    color_map = ScalarMappable(cmap=cmap)
    color_map.set_clim(displ_min, displ_max)

    rgb_map = color_map.to_rgba(displacement)
    rgb_map[num.isnan(displacement)] = 1.0
    rgb_map *= shad[:, :, num.newaxis]

    return rgb_map


def gnss_fits(problem, stage, plot_options):

    from pyrocko import automap
    from pyrocko.model import gnss

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError("GMT needs to be installed for GNSS plot!")

    gc = problem.config.geodetic_config
    try:
        ds_config = gc.types["GNSS"]
    except KeyError:
        raise ImportError("No GNSS data in configuration!")

    logger.info("Trying to load GNSS data from: {}".format(ds_config.datadir))

    campaigns = ds_config.load_data(campaign=True)
    if not campaigns:
        raise ImportError("Did not fing GNSS data under %s" % ds_config.datadir)

    if len(campaigns) > 1:
        logger.warning("Plotting for more than 1 GNSS dataset needs tp be implemented")

    campaign = campaigns[0]

    datatype = "geodetic"
    mode = problem.config.problem_config.mode
    problem.init_hierarchicals()

    figsize = 20.0  # size in cm

    po = plot_options

    composite = problem.composites[datatype]
    try:
        sources = composite.sources
        ref_sources = None
    except AttributeError:
        logger.info("FFI gnss fit, using reference source ...")
        ref_sources = composite.config.gf_config.reference_sources
        set_anchor(ref_sources, anchor="top")
        fault = composite.load_fault_geometry()
        sources = fault.get_all_subfaults(
            datatype=datatype, component=composite.slip_varnames[0]
        )
        set_anchor(sources, anchor="top")

    if po.reference:
        if mode != ffi_mode_str:
            composite.point2sources(po.reference)
            ref_sources = copy.deepcopy(composite.sources)
        bpoint = po.reference
    else:
        bpoint = get_result_point(stage.mtrace, po.post_llk)

    results = composite.assemble_results(bpoint)

    bvar_reductions = composite.get_variance_reductions(
        bpoint, weights=composite.weights, results=results
    )

    dataset_to_result = {}
    for dataset, result in zip(composite.datasets, results):
        if dataset.typ == "GNSS":
            dataset_to_result[dataset] = result

    if po.plot_projection == "latlon":
        event = problem.config.event
        locations = campaign.stations  # + [event]
        # print(locations)
        # lat, lon = otd.geographic_midpoint_locations(locations)

        coords = num.array([loc.effective_latlon for loc in locations])
        lat, lon = num.mean(num.vstack([coords.min(0), coords.max(0)]), axis=0)

    elif po.plot_projection == "local":
        lat, lon = otd.geographic_midpoint_locations(sources)
        coords = num.hstack([source.outline(cs="latlon").T for source in sources]).T

    else:
        raise NotImplementedError("%s projection not implemented!" % po.plot_projection)

    if po.nensemble > 1:
        from tqdm import tqdm

        logger.info(
            "Collecting ensemble of %i " "synthetic displacements ..." % po.nensemble
        )
        nchains = len(stage.mtrace)
        csteps = float(nchains) / po.nensemble
        idxs = num.floor(num.arange(0, nchains, csteps)).astype("int32")
        ens_results = []
        # points = []
        ens_var_reductions = []
        for idx in tqdm(idxs):
            point = stage.mtrace.point(idx=idx)
            # points.append(point)
            e_results = composite.assemble_results(point)
            ens_results.append(e_results)
            ens_var_reductions.append(
                composite.get_variance_reductions(
                    point, weights=composite.weights, results=e_results
                )
            )

        all_var_reductions = {}
        bvar_reductions_comp = {}
        for dataset in dataset_to_result.keys():
            target_var_reds = []
            target_bvar_red = bvar_reductions[dataset.name]
            target_var_reds.append(target_bvar_red)
            bvar_reductions_comp[dataset.component] = target_bvar_red * 100.0
            for var_reds in ens_var_reductions:
                target_var_reds.append(var_reds[dataset.name])

            all_var_reductions[dataset.component] = num.array(target_var_reds) * 100.0

    radius = otd.distance_accurate50m_numpy(
        lat[num.newaxis], lon[num.newaxis], coords[:, 0], coords[:, 1]
    ).max()

    radius *= 1.2

    if radius < 30.0 * km:
        logger.warning(
            "Radius of GNSS campaign %s too small, defaulting"
            " to 30 km" % campaign.name
        )
        radius = 30 * km

    model_camp = gnss.GNSSCampaign(
        stations=copy.deepcopy(campaign.stations), name="model"
    )

    for dataset, result in dataset_to_result.items():
        for ista, sta in enumerate(model_camp.stations):
            comp = getattr(sta, dataset.component)
            comp.shift = result.processed_syn[ista]
            comp.sigma = 0.0

    plot_component_flags = []
    if "east" in ds_config.components or "north" in ds_config.components:
        plot_component_flags.append(False)

    if "up" in ds_config.components:
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
            color_dry=(238, 236, 230),
        )

        all_stations = campaign.stations + model_camp.stations
        offset_scale = num.zeros(len(all_stations))

        for ista, sta in enumerate(all_stations):
            for comp in sta.components.values():
                offset_scale[ista] += comp.shift

        offset_scale = num.sqrt(offset_scale**2).max()

        if len(campaign.stations) > 40:
            logger.warning("More than 40 stations disabling station labels ..")
            labels = False
        else:
            labels = True

        m.add_gnss_campaign(
            campaign,
            psxy_style={"G": "black", "W": "0.8p,black"},
            offset_scale=offset_scale,
            vertical=vertical,
            labels=labels,
        )

        m.add_gnss_campaign(
            model_camp,
            psxy_style={"G": "red", "W": "0.8p,red", "t": 30},
            offset_scale=offset_scale,
            vertical=vertical,
            labels=False,
        )

        for i, source in enumerate(sources):
            in_rows = source.outline(cs="lonlat")
            if mode != ffi_mode_str:
                color = (num.array(mpl_graph_color(i)) * 255).tolist()
                color_str = utility.list2string(color, "/")
            else:
                color_str = "black"

            if in_rows.shape[0] > 1:  # finite source
                m.gmt.psxy(
                    in_rows=in_rows,
                    L="+p0.1p,%s" % color_str,
                    W="0.1p,black",
                    G=color_str,
                    t=70,
                    *m.jxyr
                )
                m.gmt.psxy(in_rows=in_rows[0:2], W="1p,black", *m.jxyr)
            else:  # point source
                source_scale_factor = 2.0
                m.gmt.psxy(
                    in_rows=in_rows,
                    W="0.1p,black",
                    G=color_str,
                    S="c%fp" % float(source.magnitude * source_scale_factor),
                    t=70,
                    *m.jxyr
                )

        if dataset:
            # plot strain rate tensor
            if dataset.has_correction:
                from beat.heart import StrainRateTensor
                from beat.models.corrections import StrainRateCorrection

                for i, corr in enumerate(dataset.corrections):

                    if isinstance(corr, StrainRateCorrection):
                        lats, lons = corr.get_station_coordinates()
                        mid_lat, mid_lon = otd.geographic_midpoint(lats, lons)
                        corr_point = corr.get_point_rvs(bpoint)
                        srt = StrainRateTensor.from_point(corr_point)
                        in_rows = [(mid_lon, mid_lat, srt.eps1, srt.eps2, srt.azimuth)]

                        color_str = get_gmt_colorstring_from_mpl(i)
                        m.gmt.psvelo(
                            in_rows=in_rows,
                            S="x%f" % offset_scale,
                            A="9p+g%s+p1p" % color_str,
                            W=color_str,
                            *m.jxyr
                        )

        m.draw_axes()
        if po.nensemble > 1:
            if vertical:
                var_reductions_ens = all_var_reductions["up"]
            else:
                var_reductions_tmp = []
                if "east" in all_var_reductions:
                    var_reductions_tmp.append(all_var_reductions["east"])

                if "north" in all_var_reductions:
                    var_reductions_tmp.append(all_var_reductions["north"])

                var_reductions_ens = num.mean(var_reductions_tmp, axis=0)

            # draw white background box for histogram
            m.gmt.psbasemap(D="n0.722/0.716+w4c/4c", F="+gwhite+p0.25p", *m.jxyr)

            # get resulting bounds no plotting
            vmin, vmax = var_reductions_ens.min(), var_reductions_ens.max()
            imin, imax, sinc = get_nice_plot_bounds(vmin, vmax)
            nbins = 50
            Z = 0

            out_filename = "/tmp/histbounds.txt"
            m.gmt.pshistogram(
                in_rows=make_2d(all_var_reductions[dataset.component]),
                W=(imax - imin) / nbins,
                out_filename=out_filename,
                Z=Z,
                I="o",
            )
            bin_bounds = num.loadtxt(out_filename).max(0)
            bmin, bmax, binc = get_nice_plot_bounds(0, bin_bounds[1])

            # set data region
            jxyr = [
                "-JX4c/4c",
                "-Xf13.5c",
                "-Yf13.4c",
                "-R{}/{}/{}/{}".format(imin, imax, bmin, bmax),
            ]

            hist_args = [
                "-Bxa%ff%f+lVR [%s]" % (sinc, sinc, "%"),
                "-Bya%i" % binc,
                "-BwSne",
            ] + jxyr

            m.gmt.pshistogram(
                in_rows=make_2d(all_var_reductions[dataset.component]),
                W=(imax - imin) / nbins,
                G="lightorange",
                Z=Z,
                F=True,
                L="0.5p,orange",
                *hist_args
            )

            # plot vertical line on hist with best solution
            m.gmt.psxy(
                in_rows=(
                    [bvar_reductions_comp[dataset.component], 0],
                    [bvar_reductions_comp[dataset.component], po.nensemble],
                ),
                W="1.5p,red",
                *jxyr
            )

        figs.append(m)

    return figs


def geodetic_covariances(problem, stage, plot_options):

    datatype = "geodetic"
    mode = problem.config.problem_config.mode
    problem.init_hierarchicals()

    po = plot_options

    composite = problem.composites[datatype]
    event = composite.event
    try:
        sources = composite.sources
        ref_sources = None
    except AttributeError:
        logger.info("FFI scene fit, using reference source ...")
        ref_sources = composite.config.gf_config.reference_sources
        set_anchor(ref_sources, anchor="top")
        fault = composite.load_fault_geometry()
        sources = fault.get_all_subfaults(
            datatype=datatype, component=composite.slip_varnames[0]
        )
        set_anchor(sources, anchor="top")

    if po.reference:
        if mode != ffi_mode_str:
            composite.point2sources(po.reference)
            ref_sources = copy.deepcopy(composite.sources)
        bpoint = po.reference
    else:
        bpoint = get_result_point(stage.mtrace, po.post_llk)

    tpoint = get_weights_point(composite, bpoint, problem.config)

    bresults_tmp = composite.assemble_results(bpoint)
    composite.analyse_noise(tpoint)

    covariances = [dataset.covariance for dataset in composite.datasets]

    figs, axs = plot_covariances(composite.datasets, covariances)

    return figs


def scene_fits(problem, stage, plot_options):
    """
    Plot geodetic data, synthetics and residuals.
    """
    import gc

    from kite.scene import Scene, UserIOWarning
    from pyrocko.dataset import gshhg

    from beat.colormap import roma_colormap

    try:
        homepath = problem.config.geodetic_config.types["SAR"].datadir
    except KeyError:
        raise ValueError("SAR data not in geodetic types!")

    datatype = "geodetic"
    mode = problem.config.problem_config.mode
    problem.init_hierarchicals()

    fontsize = 10
    fontsize_title = 12
    ndmax = 3
    nxmax = 3
    # cmap = plt.cm.jet
    # cmap = roma_colormap(256)
    cmap = plt.cm.RdYlBu_r

    po = plot_options

    composite = problem.composites[datatype]
    event = composite.event
    try:
        sources = composite.sources
        ref_sources = None
    except AttributeError:
        logger.info("FFI scene fit, using reference source ...")
        ref_sources = composite.config.gf_config.reference_sources
        set_anchor(ref_sources, anchor="top")
        fault = composite.load_fault_geometry()
        sources = fault.get_all_subfaults(
            datatype=datatype, component=composite.slip_varnames[0]
        )
        set_anchor(sources, anchor="top")

    if po.reference:
        if mode != ffi_mode_str:
            composite.point2sources(po.reference)
            ref_sources = copy.deepcopy(composite.sources)
        bpoint = po.reference
    else:
        bpoint = get_result_point(stage.mtrace, po.post_llk)

    bresults_tmp = composite.assemble_results(bpoint)

    tpoint = get_weights_point(composite, bpoint, problem.config)

    composite.analyse_noise(tpoint)
    composite.update_weights(tpoint)

    # to display standardized residuals
    stdz_residuals = composite.get_standardized_residuals(
        bpoint, results=bresults_tmp, weights=composite.weights
    )

    if po.plot_projection == "individual":
        for result, dataset in zip(bresults_tmp, composite.datasets):
            result.processed_res = stdz_residuals[dataset.name]

    bvar_reductions = composite.get_variance_reductions(
        bpoint, weights=composite.weights, results=bresults_tmp
    )

    dataset_to_result = {}
    for dataset, bresult in zip(composite.datasets, bresults_tmp):
        if dataset.typ == "SAR":
            dataset_to_result[dataset] = bresult

    results = dataset_to_result.values()
    dataset_index = dict((data, i) for (i, data) in enumerate(dataset_to_result.keys()))

    nrmax = len(dataset_to_result.keys())
    fullfig, restfig = utility.mod_i(nrmax, ndmax)
    factors = num.ones(fullfig).tolist()
    if restfig:
        factors.append(float(restfig) / ndmax)

    topo_plot_thresh = 300
    if plot_options.nensemble > topo_plot_thresh:
        logger.info("Plotting shaded relief as nensemble > %i." % topo_plot_thresh)
        show_topo = True
    else:
        logger.info(
            "Not plotting shaded relief for nensemble < %i." % (topo_plot_thresh + 1)
        )
        show_topo = False

    if po.nensemble > 1:
        from tqdm import tqdm

        logger.info(
            "Collecting ensemble of %i " "synthetic displacements ..." % po.nensemble
        )
        nchains = len(stage.mtrace)
        csteps = float(nchains) / po.nensemble
        idxs = num.floor(num.arange(0, nchains, csteps)).astype("int32")
        ens_results = []
        points = []
        ens_var_reductions = []
        for idx in tqdm(idxs):
            point = stage.mtrace.point(idx=idx)
            points.append(point)
            e_results = composite.assemble_results(point)
            ens_results.append(e_results)
            ens_var_reductions.append(
                composite.get_variance_reductions(
                    point, weights=composite.weights, results=e_results
                )
            )

        all_var_reductions = {}
        for dataset in dataset_to_result.keys():
            target_var_reds = []
            target_var_reds.append(bvar_reductions[dataset.name])
            for var_reds in ens_var_reductions:
                target_var_reds.append(var_reds[dataset.name])

            all_var_reductions[dataset.name] = num.array(target_var_reds) * 100.0

    figures = []
    axes = []
    for f in factors:
        figsize = list(mpl_papersize("a4", "portrait"))
        figsize[1] *= f

        fig, ax = plt.subplots(
            nrows=int(round(ndmax * f)), ncols=nxmax, figsize=figsize
        )
        fig.tight_layout()
        fig.subplots_adjust(
            left=0.08,
            right=1.0 - 0.03,
            bottom=0.06,
            top=1.0 - 0.06,
            wspace=0.0,
            hspace=0.1,
        )
        figures.append(fig)
        ax_a = num.atleast_2d(ax)
        axes.append(ax_a)

    nfigs = len(figures)

    def axis_config(axes, source, scene, po):

        latlon_ratio = get_latlon_ratio(source.lat, source.lon)
        for i, ax in enumerate(axes):
            if po.plot_projection == "latlon":
                ystr = "Latitude [deg]"
                xstr = "Longitude [deg]"
                if scene.frame.isDegree():
                    scale_x = {"scale": 1.0}
                    scale_y = {"scale": 1.0}
                    ax.set_aspect(latlon_ratio)
                else:
                    scale_x = {"scale": otd.m2d}
                    scale_y = {"scale": otd.m2d}
                    ax.set_aspect("equal")

                scale_x["offset"] = source.lon
                scale_y["offset"] = source.lat

            elif po.plot_projection in ["local", "individual"]:
                ystr = "Distance [km]"
                xstr = "Distance [km]"
                if scene.frame.isDegree():
                    scale_x = {"scale": otd.d2m / km / latlon_ratio}
                    scale_y = {"scale": otd.d2m / km}
                    ax.set_aspect(latlon_ratio)
                else:
                    scale_x = {"scale": 1.0 / km}
                    scale_y = {"scale": 1.0 / km}
                    ax.set_aspect("equal")
            else:
                raise TypeError("Plot projection %s not available" % po.plot_projection)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

            if i == 0:
                ax.set_ylabel(ystr, fontsize=fontsize)
                ax.set_xlabel(xstr, fontsize=fontsize)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

            ax.scale_x = scale_x
            ax.scale_y = scale_y

            scale_axes(ax.get_xaxis(), **scale_x)
            scale_axes(ax.get_yaxis(), **scale_y)

            if i > 0:
                ax.set_yticklabels([])
                ax.set_xticklabels([])

    def draw_coastlines(ax, xlim, ylim, event, scene, po):
        """
        xlim and ylim in Lon/Lat[deg]
        """

        logger.debug("Drawing coastlines ...")
        coasts = gshhg.GSHHG.full()

        if po.plot_projection == "latlon":
            west, east = xlim
            south, north = ylim

        elif po.plot_projection in ["local", "individual"]:
            lats, lons = otd.ne_to_latlon(
                event.lat,
                event.lon,
                north_m=num.array(ylim) * km,
                east_m=num.array(xlim) * km,
            )
            south, north = lats
            west, east = lons

        polygons = coasts.get_polygons_within(
            west=west, east=east, south=south, north=north
        )

        for p in polygons:
            if p.is_land() or p.is_antarctic_grounding_line() or p.is_island_in_lake():

                if scene.frame.isMeter():
                    ys, xs = otd.latlon_to_ne_numpy(
                        event.lat, event.lon, p.lats, p.lons
                    )

                elif scene.frame.isDegree():

                    xs = p.lons - event.lon
                    ys = p.lats - event.lat

                ax.plot(xs, ys, "-k", linewidth=0.5)

    def add_arrow(ax, scene):
        phi = num.nanmean(scene.phi)
        theta = num.nanmean(scene.theta)
        if theta == 0.0:  # MAI / az offsets
            phi -= num.pi

        los_dx = num.cos(phi + num.pi) * 0.0625
        los_dy = num.sin(phi + num.pi) * 0.0625

        az_dx = num.cos(phi - num.pi / 2) * 0.125
        az_dy = num.sin(phi - num.pi / 2) * 0.125

        anchor_x = 0.9 if los_dx < 0 else 0.1
        anchor_y = 0.85 if los_dx < 0 else 0.975

        if theta > 0.0:  # MAI / az offsets
            az_arrow = FancyArrow(
                x=anchor_x - az_dx,
                y=anchor_y - az_dy,
                dx=az_dx,
                dy=az_dy,
                head_width=0.025,
                alpha=0.5,
                fc="k",
                head_starts_at_zero=False,
                length_includes_head=True,
                transform=ax.transAxes,
            )
            ax.add_artist(az_arrow)

        los_arrow = FancyArrow(
            x=anchor_x - az_dx / 2,
            y=anchor_y - az_dy / 2,
            dx=los_dx,
            dy=los_dy,
            head_width=0.02,
            alpha=0.5,
            fc="k",
            head_starts_at_zero=False,
            length_includes_head=True,
            transform=ax.transAxes,
        )

        ax.add_artist(los_arrow)

    def draw_leaves(ax, scene, offset_e=0, offset_n=0):
        rects = scene.quadtree.getMPLRectangles()
        for r in rects:
            r.set_edgecolor((0.4, 0.4, 0.4))
            r.set_linewidth(0.5)
            r.set_facecolor("none")
            r.set_x(r.get_x() - offset_e)
            r.set_y(r.get_y() - offset_n)
        map(ax.add_artist, rects)

        ax.scatter(
            scene.quadtree.leaf_coordinates[:, 0] - offset_e,
            scene.quadtree.leaf_coordinates[:, 1] - offset_n,
            s=0.25,
            c="black",
            alpha=0.1,
        )

    def draw_sources(ax, sources, scene, po, event, **kwargs):
        bgcolor = kwargs.pop("color", None)

        for i, source in enumerate(sources):

            if scene.frame.isMeter():
                fn, fe = source.outline(cs="xy").T
            elif scene.frame.isDegree():
                fn, fe = source.outline(cs="latlon").T
                fn -= event.lat
                fe -= event.lon

            if not bgcolor:
                color = mpl_graph_color(i)
            else:
                color = bgcolor

            if fn.size > 1:
                alpha = 0.4
                ax.plot(fe, fn, "-", linewidth=0.5, color=color, alpha=alpha, **kwargs)
                ax.fill(
                    fe, fn, edgecolor=color, facecolor=light(color, 0.5), alpha=alpha
                )
                ax.plot(fe[0:2], fn[0:2], "-k", alpha=0.7, linewidth=1.0)
            else:
                ax.plot(fe, fn, marker="*", markersize=10, color=color, **kwargs)

    colims = [
        num.max([num.max(num.abs(r.processed_obs)), num.max(num.abs(r.processed_syn))])
        for r in results
    ]
    dcolims = [num.max(num.abs(r.processed_res)) for r in results]

    import string

    for idata, (dataset, result) in enumerate(dataset_to_result.items()):
        subplot_letter = string.ascii_lowercase[idata]
        try:
            scene_path = os.path.join(homepath, dataset.name)
            logger.info("Loading full resolution kite scene: %s" % scene_path)
            scene = Scene.load(scene_path)
        except UserIOWarning:
            logger.warning("Full resolution data could not be loaded! Skipping ...")
            continue

        if scene.frame.isMeter():
            offset_n, offset_e = map(
                float,
                otd.latlon_to_ne_numpy(
                    scene.frame.llLat, scene.frame.llLon, event.lat, event.lon
                ),
            )

        elif scene.frame.isDegree():
            offset_n = event.lat - scene.frame.llLat
            offset_e = event.lon - scene.frame.llLon

        im_extent = (
            scene.frame.E.min() - offset_e,
            scene.frame.E.max() - offset_e,
            scene.frame.N.min() - offset_n,
            scene.frame.N.max() - offset_n,
        )

        urE, urN, llE, llN = (0.0, 0.0, 0.0, 0.0)

        true, turN, tllE, tllN = zip(
            *[
                (
                    l.gridE.max() - offset_e,
                    l.gridN.max() - offset_n,
                    l.gridE.min() - offset_e,
                    l.gridN.min() - offset_n,
                )
                for l in scene.quadtree.leaves
            ]
        )

        true, turN = map(max, (true, turN))
        tllE, tllN = map(min, (tllE, tllN))
        urE, urN = map(max, ((true, urE), (urN, turN)))
        llE, llN = map(min, ((tllE, llE), (llN, tllN)))

        lat, lon = otd.ne_to_latlon(
            sources[0].lat, sources[0].lon, num.array([llN, urN]), num.array([llE, urE])
        )

        # result = dataset_to_result[dataset]
        tidx = dataset_index[dataset]

        figidx, rowidx = utility.mod_i(tidx, ndmax)
        axs = axes[figidx][rowidx, :]

        imgs = []
        for ax, data_str in zip(axs, ["obs", "syn", "res"]):
            logger.info("Plotting %s" % data_str)
            datavec = getattr(result, "processed_%s" % data_str)

            if data_str == "res" and po.plot_projection in ["local", "individual"]:
                vmin = -dcolims[tidx]
                vmax = dcolims[tidx]
                logger.debug(
                    "Variance of residual for %s is: %f", dataset.name, datavec.var()
                )
            else:
                vmin = -colims[tidx]
                vmax = colims[tidx]

            data = map_displacement_grid(datavec, scene)

            if show_topo:
                elevation = scene.get_elevation()
                elevation_mask = num.where(elevation == 0.0, True, False)

                data = shaded_displacements(
                    data,
                    elevation,
                    cmap,
                    shad_lim=(0.4, 0.99),
                    contrast=1.0,
                    mask=elevation_mask,
                    data_limits=(vmin, vmax),
                )

            imgs.append(
                ax.imshow(
                    data,
                    extent=im_extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    interpolation="nearest",
                )
            )

            ax.set_xlim(llE, urE)
            ax.set_ylim(llN, urN)

            draw_leaves(ax, scene, offset_e, offset_n)
            draw_coastlines(ax, lon, lat, event, scene, po)

        # histogram of stdz residual
        in_ax_res = plot_inset_hist(
            axs[2],
            data=make_2d(stdz_residuals[dataset.name]),
            best_data=None,
            linewidth=1.0,
            bbox_to_anchor=(0.0, 0.775, 0.25, 0.225),
            labelsize=6,
            color="grey",
        )
        # reference gaussian
        x = num.linspace(*stats.norm.ppf((0.001, 0.999)), 100)
        gauss = stats.norm.pdf(x)
        in_ax_res.plot(x, gauss, "k-", lw=0.5, alpha=0.8)

        format_axes(in_ax_res, remove=["right", "bottom"], visible=True, linewidth=0.75)
        in_ax_res.set_xlabel("std. res. [$\sigma$]", fontsize=fontsize - 3)

        if po.nensemble > 1:
            in_ax = plot_inset_hist(
                axs[2],
                data=make_2d(all_var_reductions[dataset.name]),
                best_data=bvar_reductions[dataset.name] * 100.0,
                linewidth=1.0,
                bbox_to_anchor=(0.75, 0.775, 0.25, 0.225),
                labelsize=6,
            )

            format_axes(in_ax, remove=["left", "bottom"], visible=True, linewidth=0.75)
            in_ax.set_xlabel("VR [%]", fontsize=fontsize - 3)

        fontdict = {
            "fontsize": fontsize,
            "fontweight": "bold",
            "verticalalignment": "top",
        }

        transform = axes[figidx][rowidx, 0].transAxes

        if dataset.name[-5::] == "dscxn":
            title = "descending"
        elif dataset.name[-5::] == "ascxn":
            title = "ascending"
        else:
            title = dataset.name

        axes[figidx][rowidx, 0].text(
            0.025,
            1.025,
            "({}) {}".format(subplot_letter, title),
            fontsize=fontsize_title,
            alpha=1.0,
            va="bottom",
            transform=transform,
        )
        for i, quantity in enumerate(["data", "model", "residual"]):
            transform = axes[figidx][rowidx, i].transAxes
            axes[figidx][rowidx, i].text(
                0.5,
                0.95,
                quantity,
                fontdict,
                transform=transform,
                horizontalalignment="center",
            )

        draw_sources(axes[figidx][rowidx, 1], sources, scene, po, event=event)

        if ref_sources:
            ref_color = scolor("aluminium4")
            logger.info("Plotting reference sources")
            draw_sources(
                axes[figidx][rowidx, 1],
                ref_sources,
                scene,
                po,
                event=event,
                color=ref_color,
            )

        f = factors[figidx]
        if f > 2.0 / 3:
            cbb = 0.68 - (0.3075 * rowidx)
        elif f > 1.0 / 2:
            cbb = 0.53 - (0.47 * rowidx)
        elif f > 1.0 / 4:
            cbb = 0.06

        cbl = 0.46
        cbw = 0.15
        cbh = 0.01

        cbaxes = figures[figidx].add_axes([cbl, cbb, cbw, cbh])

        cblabel = "LOS displacement [m]"
        cbs = plt.colorbar(
            imgs[1],
            ax=axes[figidx][rowidx, 0],
            ticks=cbtick(colims[tidx]),
            cax=cbaxes,
            orientation="horizontal",
        )
        cbs.set_label(cblabel, fontsize=fontsize)

        if po.plot_projection in ["local", "individual"]:
            dcbaxes = figures[figidx].add_axes([cbl + 0.3, cbb, cbw, cbh])
            cbr = plt.colorbar(
                imgs[2],
                ax=axes[figidx][rowidx, 2],
                ticks=cbtick(dcolims[tidx]),
                cax=dcbaxes,
                orientation="horizontal",
            )
            if po.plot_projection == "individual":
                cblabel = "standard dev [$\sigma$]"

            cbr.set_label(cblabel, fontsize=fontsize)

        axis_config(axes[figidx][rowidx, :], event, scene, po)
        add_arrow(axes[figidx][rowidx, 0], scene)

        del scene
        gc.collect()

    return figures


def draw_geodetic_covariances(problem, plot_options):

    if "geodetic" not in list(problem.composites.keys()):
        raise TypeError("No geodetic composite defined in the problem!")

    logger.info("Drawing geodetic covariances ...")
    po = plot_options

    stage = Stage(
        homepath=problem.outfolder, backend=problem.config.sampler_config.backend
    )

    if not po.reference:
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model,
            stage_number=po.load_stage,
            load="trace",
            chains=[-1],
        )
        llk_str = po.post_llk
    else:
        llk_str = "ref"

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode,
        po.figure_dir,
        "geodetic_covs_%s_%s" % (stage.number, llk_str),
    )

    if not os.path.exists(outpath + ".%s" % po.outformat) or po.force:
        figs = geodetic_covariances(problem, stage, po)
    else:
        logger.info("geodetic covariances plots exist. Use force=True for replotting!")
        return

    if po.outformat == "display":
        plt.show()
    else:
        logger.info("saving figures to %s" % outpath)
        if po.outformat == "pdf":
            with PdfPages(outpath + ".pdf") as opdf:
                for fig in figs:
                    opdf.savefig(fig)
        else:
            for i, fig in enumerate(figs):
                fig.savefig("%s_%i.%s" % (outpath, i, po.outformat), dpi=po.dpi)


def draw_scene_fits(problem, plot_options):

    if "geodetic" not in list(problem.composites.keys()):
        raise TypeError("No geodetic composite defined in the problem!")

    if "SAR" not in problem.config.geodetic_config.types:
        raise TypeError("There is no SAR data in the problem setup!")

    logger.info("Drawing SAR misfits ...")
    po = plot_options

    stage = Stage(
        homepath=problem.outfolder, backend=problem.config.sampler_config.backend
    )

    if not po.reference:
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model,
            stage_number=po.load_stage,
            load="trace",
            chains=[-1],
        )
        llk_str = po.post_llk
    else:
        llk_str = "ref"

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode,
        po.figure_dir,
        "scenes_%s_%s_%s_%i"
        % (stage.number, llk_str, po.plot_projection, po.nensemble),
    )

    if not os.path.exists(outpath + ".%s" % po.outformat) or po.force:
        figs = scene_fits(problem, stage, po)
    else:
        logger.info("scene plots exist. Use force=True for replotting!")
        return

    if po.outformat == "display":
        plt.show()
    else:
        logger.info("saving figures to %s" % outpath)
        if po.outformat == "pdf":
            with PdfPages(outpath + ".pdf") as opdf:
                for fig in figs:
                    opdf.savefig(fig)
        else:
            for i, fig in enumerate(figs):
                fig.savefig("%s_%i.%s" % (outpath, i, po.outformat), dpi=po.dpi)


def draw_gnss_fits(problem, plot_options):

    if "geodetic" not in list(problem.composites.keys()):
        raise TypeError("No geodetic composite defined in the problem!")

    if "GNSS" not in problem.config.geodetic_config.types:
        raise TypeError("There is no GNSS data in the problem setup!")

    logger.info("Drawing GNSS misfits ...")

    po = plot_options

    stage = Stage(
        homepath=problem.outfolder, backend=problem.config.sampler_config.backend
    )

    if not po.reference:
        stage.load_results(
            varnames=problem.varnames,
            model=problem.model,
            stage_number=po.load_stage,
            load="trace",
            chains=[-1],
        )
        llk_str = po.post_llk
    else:
        llk_str = "ref"

    mode = problem.config.problem_config.mode

    outpath = os.path.join(
        problem.config.project_dir,
        mode,
        po.figure_dir,
        "gnss_%s_%s_%i_%s" % (stage.number, llk_str, po.nensemble, po.plot_projection),
    )

    if not os.path.exists(outpath) or po.force:
        figs = gnss_fits(problem, stage, po)
    else:
        logger.info("scene plots exist. Use force=True for replotting!")
        return

    if po.outformat == "display":
        plt.show()
    else:
        logger.info("saving figures to %s" % outpath)
        for component, fig in zip(("horizontal", "vertical"), figs):
            fig.save(outpath + "_%s.%s" % (component, po.outformat), resolution=po.dpi)
