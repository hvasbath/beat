import logging
import os

import numpy as num
import pyrocko.moment_tensor as mt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from pyrocko import gmtpy
from pyrocko import orthodrome as otd
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.plot import (
    AutoScaler,
    mpl_graph_color,
    mpl_init,
    mpl_margins,
    mpl_papersize,
)

from beat import utility
from beat.config import ffi_mode_str
from beat.models import Stage, load_stage

from .common import (
    draw_line_on_array,
    format_axes,
    get_gmt_config,
    get_result_point,
    km,
    scale_axes,
)

logger = logging.getLogger("plotting.ffi")


def fuzzy_moment_rate(ax, moment_rates, times, cmap=None, grid_size=(500, 500)):
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
            "Number of rates and times have to be identical!"
            " %i != %i" % (nrates, ntimes)
        )

    max_rates = max(map(num.max, moment_rates))
    max_times = max(map(num.max, times))
    min_rates = min(map(num.min, moment_rates))
    min_times = min(map(num.min, times))

    extent = (min_times, max_times, min_rates, max_rates)
    grid = num.zeros(grid_size, dtype="float64")

    for mr, time in zip(moment_rates, times):
        draw_line_on_array(
            time, mr, grid=grid, extent=extent, grid_resolution=grid.shape, linewidth=7
        )

    # increase contrast reduce high intense values
    truncate = nrates / 2
    grid[grid > truncate] = truncate

    ax.imshow(grid, extent=extent, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Moment rate [$Nm / s$]")


def draw_moment_rate(problem, po):
    """
    Draw moment rate function for the results of a seismic/joint finite fault
    optimization.
    """
    fontsize = 12
    mode = problem.config.problem_config.mode

    if mode != ffi_mode_str:
        raise ModeError(
            "Wrong optimization mode: %s! This plot "
            'variant is only valid for "%s" mode' % (mode, ffi_mode_str)
        )

    if "seismic" not in problem.config.problem_config.datatypes:
        raise TypeError(
            "Moment rate function only available for optimization results that"
            " include seismic data."
        )

    sc = problem.composites["seismic"]
    fault = sc.load_fault_geometry()

    stage = load_stage(problem, stage_number=po.load_stage, load="trace", chains=[-1])

    if not po.reference:
        reference = get_result_point(stage.mtrace, po.post_llk)
        llk_str = po.post_llk
        mtrace = stage.mtrace
    else:
        reference = po.reference
        llk_str = "ref"
        mtrace = None

    logger.info("Drawing ensemble of %i moment rate functions ..." % po.nensemble)
    target = sc.wavemaps[0].targets[0]

    if po.plot_projection == "individual":
        logger.info("Drawing subfault individual rates ...")
        sf_idxs = range(fault.nsubfaults)
    else:
        logger.info("Drawing total rates ...")
        sf_idxs = [list(range(fault.nsubfaults))]

    mpl_init(fontsize=fontsize)
    for i, ns in enumerate(sf_idxs):
        logger.info("Fault %i / %i" % (i + 1, len(sf_idxs)))
        if isinstance(ns, list):
            ns_str = "total"
        else:
            ns_str = str(ns)

        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            "moment_rate_%i_%s_%s_%i.%s"
            % (stage.number, ns_str, llk_str, po.nensemble, po.outformat),
        )

        ref_mrf_rates, ref_mrf_times = fault.get_moment_rate_function(
            index=ns,
            point=reference,
            target=target,
            store=sc.engine.get_store(target.store_id),
        )

        if not os.path.exists(outpath) or po.force:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=mpl_papersize("a7", "landscape")
            )
            labelpos = mpl_margins(
                fig, left=5, bottom=4, top=1.5, right=0.5, units=fontsize
            )
            labelpos(ax, 2.0, 1.5)
            if mtrace is not None:
                nchains = len(mtrace)
                csteps = float(nchains) / po.nensemble
                idxs = num.floor(num.arange(0, nchains, csteps)).astype("int32")
                mrfs_rate = []
                mrfs_time = []
                for idx in idxs:
                    point = mtrace.point(idx=idx)
                    mrf_rate, mrf_time = fault.get_moment_rate_function(
                        index=ns,
                        point=point,
                        target=target,
                        store=sc.engine.get_store(target.store_id),
                    )
                    mrfs_rate.append(mrf_rate)
                    mrfs_time.append(mrf_time)

                fuzzy_moment_rate(ax, mrfs_rate, mrfs_time)

            ax.plot(ref_mrf_times, ref_mrf_rates, "-k", alpha=0.8, linewidth=1.0)
            format_axes(ax, remove=["top", "right"])

            if po.outformat == "display":
                plt.show()
            else:
                logger.info("saving figure to %s" % outpath)
                fig.savefig(outpath, format=po.outformat, dpi=po.dpi)

        else:
            logger.info("Plot exists! Use --force to overwrite!")


def source_geometry(
    fault,
    ref_sources,
    event,
    datasets=None,
    values=None,
    cmap=None,
    title=None,
    show=True,
    cbounds=None,
    clabel="",
):
    """
    Plot source geometry in 3d rotatable view

    Parameters
    ----------
    fault: :class:`beat.ffi.fault.FaultGeometry`
    ref_sources: list
        of :class:'beat.sources.RectangularSource'
    """

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    alpha = 0.7

    def plot_subfault(ax, source, color, refloc):
        source.anchor = "top"
        shift_ne = otd.latlon_to_ne(refloc.lat, refloc.lon, source.lat, source.lon)
        coords = source.outline()  # (N, E, Z)

        coords[:, 0:2] += shift_ne
        ax.plot(
            coords[:, 1],
            coords[:, 0],
            coords[:, 2] * -1.0,
            color=color,
            linewidth=2,
            alpha=alpha,
        )
        ax.plot(
            coords[0:2, 1],
            coords[0:2, 0],
            coords[0:2, 2] * -1.0,
            "-k",
            linewidth=2,
            alpha=alpha,
        )
        center = source.center  # (E, N, Z)

        center[0] += shift_ne[1]
        center[1] += shift_ne[0]
        ax.scatter(
            center[0],
            center[1],
            center[2] * -1,
            marker="o",
            s=20,
            color=color,
            alpha=alpha,
        )

    def set_axes_radius(ax, origin, radius, axes=["xyz"]):
        if "x" in axes:
            ax.set_xlim3d([origin[0] - radius, origin[0] + radius])

        if "y" in axes:
            ax.set_ylim3d([origin[1] - radius, origin[1] + radius])

        if "z" in axes:
            ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def set_axes_equal(ax, axes="xyz"):
        """
        Make axes of 3D plot have equal scale so that spheres appear as
        spheres, cubes as cubes, etc..
        This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Parameters
        ----------
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        limits = num.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])

        origin = num.mean(limits, axis=1)
        radius = 0.5 * num.max(num.abs(limits[:, 1] - limits[:, 0]))
        set_axes_radius(ax, origin, radius, axes=axes)

    fig = plt.figure(figsize=mpl_papersize("a5", "landscape"))
    ax = fig.add_subplot(111, projection="3d")
    extfs = fault.get_all_subfaults()

    arr_coords = []
    for idx, (refs, exts) in enumerate(zip(ref_sources, extfs)):

        plot_subfault(ax, exts, color=mpl_graph_color(idx), refloc=event)
        plot_subfault(ax, refs, color=scolor("aluminium4"), refloc=event)
        for i, patch in enumerate(fault.get_subfault_patches(idx)):
            coords = patch.outline()
            shift_ne = otd.latlon_to_ne(event.lat, event.lon, patch.lat, patch.lon)
            coords[:, 0:2] += shift_ne
            coords[:, 2] *= -1.0
            coords[:, [0, 1]] = coords[:, [1, 0]]  # swap columns to [E, N, Z] (X, Y, Z)
            arr_coords.append(coords)
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                zorder=2,
                color=mpl_graph_color(idx),
                linewidth=0.5,
                alpha=alpha,
            )
            ax.text(
                patch.east_shift + shift_ne[1],
                patch.north_shift + shift_ne[0],
                patch.center[2] * -1.0,
                str(i + fault.cum_subfault_npatches[idx]),
                zorder=3,
                fontsize=8,
            )

    if values is not None:

        if cmap is None:
            cmap = plt.cm.get_cmap("RdYlBu_r")

        poly_patches = Poly3DCollection(verts=arr_coords, zorder=1, cmap=cmap)
        poly_patches.set_array(values)

        if cbounds is None:
            poly_patches.set_clim(values.min(), values.max())
        else:
            poly_patches.set_clim(*cbounds)

        poly_patches.set_alpha(0.6)
        poly_patches.set_edgecolor("k")
        ax.add_collection(poly_patches)
        cbs = plt.colorbar(poly_patches, ax=ax, orientation="vertical", cmap=cmap)

        if clabel is not None:
            cbs.set_label(clabel)

    if datasets:
        for dataset in datasets:
            # print(dataset.east_shifts, dataset.north_shifts)
            ax.scatter(
                dataset.east_shifts,
                dataset.north_shifts,
                dataset.coords5[:, 4],
                s=10,
                alpha=0.6,
                marker="o",
                color="black",
            )

    scale = {"scale": 1.0 / km}
    scale_axes(ax.xaxis, **scale)
    scale_axes(ax.yaxis, **scale)
    scale_axes(ax.zaxis, **scale)
    ax.set_zlabel("Depth [km]")
    ax.set_ylabel("North_shift [km]")
    ax.set_xlabel("East_shift [km]")
    set_axes_equal(ax, axes="xy")

    strikes = num.array([extf.strike for extf in extfs])
    dips = num.array([extf.strike for extf in extfs])

    azim = strikes.mean() - 270
    elev = dips.mean()
    logger.debug("Viewing azimuth %s and elevation angles %s", azim, ax.elev)
    ax.view_init(ax.elev, azim)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    return fig, ax


def fuzzy_rupture_fronts(
    ax, rupture_fronts, xgrid, ygrid, alpha=0.6, linewidth=7, zorder=0
):
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
    cmap = LinearSegmentedColormap.from_list("dummy", ["white", "black"], N=ncolors)

    res_km = 25  # pixel per km

    xmin = xgrid.min()
    xmax = xgrid.max()
    ymin = ygrid.min()
    ymax = ygrid.max()
    extent = (xmin, xmax, ymin, ymax)
    grid = num.zeros(
        (
            int((num.abs(ymax) - num.abs(ymin)) * res_km),
            int((num.abs(xmax) - num.abs(xmin)) * res_km),
        ),
        dtype="float64",
    )
    for rupture_front in rupture_fronts:
        for level in rupture_front:
            for line in level:
                draw_line_on_array(
                    line[:, 0],
                    line[:, 1],
                    grid=grid,
                    extent=extent,
                    grid_resolution=grid.shape,
                    linewidth=linewidth,
                )

    # increase contrast reduce high intense values
    truncate = len(rupture_fronts) / 2
    grid[grid > truncate] = truncate
    ax.imshow(
        grid,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        alpha=alpha,
        zorder=zorder,
    )


def fault_slip_distribution(
    fault,
    mtrace=None,
    transform=lambda x: x,
    alpha=0.9,
    ntickmarks=5,
    reference=None,
    nensemble=1,
):
    """
    Draw discretized fault geometry rotated to the 2-d view of the foot-wall
    of the fault.
    Parameters
    ----------
    fault : :class:`ffi.fault.FaultGeometry`
    """

    def draw_quivers(
        ax,
        uperp,
        uparr,
        xgr,
        ygr,
        rake,
        color="black",
        draw_legend=False,
        normalisation=None,
        zorder=0,
    ):

        # positive uperp is always dip-normal- have to multiply -1
        angles = num.arctan2(-uperp, uparr) * mt.r2d + rake
        slips = num.sqrt((uperp**2 + uparr**2)).ravel()

        if normalisation is None:
            from beat.models.laplacian import distances

            centers = num.vstack((xgr, ygr)).T
            # interpatch_dists = distances(centers, centers)
            normalisation = slips.max()

        slips /= normalisation

        slipsx = num.cos(angles * mt.d2r) * slips
        slipsy = num.sin(angles * mt.d2r) * slips

        # slip arrows of slip on patches
        quivers = ax.quiver(
            xgr.ravel(),
            ygr.ravel(),
            slipsx,
            slipsy,
            units="dots",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=1.0,
            color=color,
            zorder=zorder,
        )

        if draw_legend:
            quiver_legend_length = (
                num.ceil(num.max(slips * normalisation) * 10.0) / 10.0
            )

            # ax.quiverkey(
            #    quivers, 0.9, 0.8, quiver_legend_length,
            #    '{} [m]'.format(quiver_legend_length), labelpos='E',
            #    coordinates='figure')

        return quivers, normalisation

    def draw_patches(
        ax, fault, subfault_idx, patch_values, cmap, alpha, cbounds=None, xlim=None
    ):

        lls = fault.get_subfault_patch_attributes(
            subfault_idx, attributes=["bottom_left"]
        )
        widths, lengths = fault.get_subfault_patch_attributes(
            subfault_idx, attributes=["width", "length"]
        )
        sf = fault.get_subfault(subfault_idx)

        # subtract reference fault lower left and rotate
        rot_lls = utility.rotate_coords_plane_normal(lls, sf)[:, 1::-1]

        d_patches = []
        for ll, width, length in zip(rot_lls, widths, lengths):
            d_patches.append(
                Rectangle(ll, width=length, height=width, edgecolor="black")
            )

        lower = rot_lls.min(axis=0)
        pad = sf.length / km * 0.05

        # xlim = [lower[0] - pad, lower[0] + sf.length / km + pad]
        if xlim is None:
            xlim = [lower[1] - pad, lower[1] + sf.width / km + pad]

        ax.set_aspect(1)
        # ax.set_xlim(*xlim)
        ax.set_xlim(*xlim)

        scale_y = {"scale": 1, "offset": (-sf.width / km)}
        scale_axes(ax.yaxis, **scale_y)

        ax.set_xlabel("strike-direction [km]", fontsize=fontsize)
        ax.set_ylabel("dip-direction [km]", fontsize=fontsize)

        xticker = MaxNLocator(nbins=ntickmarks)
        yticker = MaxNLocator(nbins=ntickmarks)

        ax.get_xaxis().set_major_locator(xticker)
        ax.get_yaxis().set_major_locator(yticker)

        pa_col = PatchCollection(d_patches, alpha=alpha, match_original=True, zorder=0)
        pa_col.set(array=patch_values, cmap=cmap)

        if cbounds is not None:
            pa_col.set_clim(*cbounds)

        ax.add_collection(pa_col)
        return pa_col

    def draw_colorbar(fig, ax, cb_related, labeltext, ntickmarks=4):
        cbaxes = fig.add_axes([0.88, 0.4, 0.03, 0.3])
        cb = fig.colorbar(cb_related, ax=axs, cax=cbaxes)
        cb.set_label(labeltext, fontsize=fontsize)
        cb.locator = MaxNLocator(nbins=ntickmarks)
        cb.update_ticks()
        ax.set_aspect("equal", adjustable="box")

    def get_values_from_trace(mtrace, fault, varname, reference):
        try:
            u = transform(mtrace.get_values(varname, combine=True, squeeze=True))
        except (ValueError, KeyError):
            u = num.atleast_2d(
                fault.var_from_point(index=None, point=reference, varname=varname)
            )
        return u

    from tqdm import tqdm

    from beat.colormap import slip_colormap

    fontsize = 12

    reference_slip = fault.get_total_slip(index=None, point=reference)
    slip_bounds = [0, reference_slip.max()]

    figs = []
    axs = []

    flengths_max = num.array([sf.length / km for sf in fault.iter_subfaults()]).max()
    pad = flengths_max * 0.03
    xmax = flengths_max + pad
    for ns in range(fault.nsubfaults):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize("a5", "landscape")
        )

        # alphas = alpha * num.ones(np_h * np_w, dtype='int8')

        try:
            ext_source = fault.get_subfault(ns, component="uparr")
        except TypeError:
            ext_source = fault.get_subfault(ns, component="utens")

        patch_idxs = fault.get_patch_indexes(ns)

        pa_col = draw_patches(
            ax,
            fault,
            subfault_idx=ns,
            patch_values=reference_slip[patch_idxs],
            xlim=[-pad, xmax],
            cmap=slip_colormap(100),
            alpha=0.65,
            cbounds=slip_bounds,
        )

        # patch central locations
        centers = fault.get_subfault_patch_attributes(ns, attributes=["center"])
        rot_centers = utility.rotate_coords_plane_normal(centers, ext_source)[:, 1::-1]

        xgr, ygr = rot_centers.T
        if "seismic" in fault.datatypes:
            shp = fault.ordering.get_subfault_discretization(ns)
            xgr = xgr.reshape(shp)
            ygr = ygr.reshape(shp)

            if mtrace is not None:
                _, dummy_ax = plt.subplots(
                    nrows=1, ncols=1, figsize=mpl_papersize("a5", "landscape")
                )

                nchains = len(mtrace)
                csteps = 6
                rupture_fronts = []
                csteps = float(nchains) / nensemble
                idxs = num.floor(num.arange(0, nchains, csteps)).astype("int32")
                logger.info("Rendering rupture fronts ...")
                for i in tqdm(idxs):
                    point = mtrace.point(idx=i)
                    sts = fault.point2starttimes(point, index=ns)

                    contours = dummy_ax.contour(xgr, ygr, sts)
                    rupture_fronts.append(contours.allsegs)

                fuzzy_rupture_fronts(
                    ax, rupture_fronts, xgr, ygr, alpha=1.0, linewidth=7, zorder=-1
                )

            # rupture durations
            if False:
                durations = transform(
                    mtrace.get_values("durations", combine=True, squeeze=True)
                )
                std_durations = durations.std(axis=0)
                # alphas = std_durations.min() / std_durations

                fig2, ax2 = plt.subplots(
                    nrows=1, ncols=1, figsize=mpl_papersize("a5", "landscape")
                )

                reference_durations = reference["durations"][patch_idxs]

                pa_col2 = draw_patches(
                    ax2,
                    fault,
                    subfault_idx=ns,
                    patch_values=reference_durations,
                    cmap=plt.cm.seismic,
                    alpha=alpha,
                    xlim=[-pad, xmax],
                )

                draw_colorbar(fig2, ax2, pa_col2, labeltext="durations [s]")
                figs.append(fig2)
                axs.append(ax2)

            ref_starttimes = fault.point2starttimes(reference, index=ns)
            contours = ax.contour(
                xgr, ygr, ref_starttimes, colors="black", linewidths=0.5, alpha=0.9
            )

            # draw subfault hypocenter
            dip_idx, strike_idx = fault.fault_locations2idxs(
                ns,
                reference["nucleation_dip"][ns],
                reference["nucleation_strike"][ns],
                backend="numpy",
            )
            psize_strike = fault.ordering.patch_sizes_strike[ns]
            psize_dip = fault.ordering.patch_sizes_dip[ns]
            nuc_strike = strike_idx * psize_strike + (psize_strike / 2.0)
            nuc_dip = dip_idx * psize_dip + (psize_dip / 2.0)
            ax.plot(
                nuc_strike,
                ext_source.width / km - nuc_dip,
                marker="*",
                color="k",
                markersize=12,
            )

            # label contourlines
            plt.clabel(
                contours, inline=True, fontsize=10, fmt=FormatStrFormatter("%.1f")
            )

        if mtrace is not None:
            logger.info("Drawing quantiles ...")

            uparr = get_values_from_trace(mtrace, fault, "uparr", reference)[
                :, patch_idxs
            ]
            uperp = get_values_from_trace(mtrace, fault, "uperp", reference)[
                :, patch_idxs
            ]
            utens = get_values_from_trace(mtrace, fault, "utens", reference)[
                :, patch_idxs
            ]

            uparrmean = uparr.mean(axis=0)
            uperpmean = uperp.mean(axis=0)
            utensmean = utens.mean(axis=0)

            if uparrmean.sum() != 0.0:
                logger.info("Found slip shear components!")
                normalisation = slip_bounds[1] / 3
                quivers, normalisation = draw_quivers(
                    ax,
                    uperpmean,
                    uparrmean,
                    xgr,
                    ygr,
                    ext_source.rake,
                    color="grey",
                    draw_legend=False,
                    normalisation=normalisation,
                )
                uparrstd = uparr.std(axis=0) / normalisation
                uperpstd = uperp.std(axis=0) / normalisation
            elif utensmean.sum() != 0:
                logger.info(
                    "Found tensile slip components! Not drawing quivers!"
                    " Circle radius shows standard deviations!"
                )
                uperpstd = uparrstd = utens.std(axis=0)
                normalisation = utens.max()
                quivers = None

            slipvecrotmat = mt.euler_to_matrix(0.0, 0.0, ext_source.rake * mt.d2r)

            circle = num.linspace(0, 2 * num.pi, 100)
            # 2sigma error ellipses
            for i, (upe, upa) in enumerate(zip(uperpstd, uparrstd)):
                ellipse_x = 2 * upa * num.cos(circle)
                ellipse_y = 2 * upe * num.sin(circle)
                ellipse = num.vstack(
                    [ellipse_x, ellipse_y, num.zeros_like(ellipse_x)]
                ).T
                rot_ellipse = ellipse.dot(slipvecrotmat)

                xcoords = xgr.ravel()[i] + rot_ellipse[:, 0]
                ycoords = ygr.ravel()[i] + rot_ellipse[:, 1]
                if quivers is not None:
                    xcoords += quivers.U[i]
                    ycoords += quivers.V[i]
                ax.plot(xcoords, ycoords, "-k", linewidth=0.5, zorder=2)
        else:
            normalisation = None

        uperp = reference["uperp"][patch_idxs]
        uparr = reference["uparr"][patch_idxs]

        if uparr.mean() != 0.0:
            logger.info("Drawing slip vectors ...")
            draw_quivers(
                ax,
                uperp,
                uparr,
                xgr,
                ygr,
                ext_source.rake,
                color="black",
                draw_legend=True,
                normalisation=normalisation,
                zorder=3,
            )

        draw_colorbar(fig, ax, pa_col, labeltext="slip [m]")
        format_axes(ax, remove=["top", "right"])

        # fig.tight_layout()
        figs.append(fig)
        axs.append(ax)

    return figs, axs


class ModeError(Exception):
    pass


def draw_slip_dist(problem, po):

    mode = problem.config.problem_config.mode

    if mode != ffi_mode_str:
        raise ModeError(
            "Wrong optimization mode: %s! This plot "
            'variant is only valid for "%s" mode' % (mode, ffi_mode_str)
        )

    datatype, gc = list(problem.composites.items())[0]

    fault = gc.load_fault_geometry()

    if not po.reference:
        stage = load_stage(
            problem, stage_number=po.load_stage, load="trace", chains=[-1]
        )
        reference = problem.config.problem_config.get_test_point()
        res_point = get_result_point(stage.mtrace, po.post_llk)
        reference.update(res_point)
        llk_str = po.post_llk
        mtrace = stage.mtrace
        stage_number = stage.number
    else:
        reference = po.reference
        llk_str = "ref"
        mtrace = None
        stage_number = -1

    figs, axs = fault_slip_distribution(
        fault, mtrace, reference=reference, nensemble=po.nensemble
    )

    if po.outformat == "display":
        plt.show()
    else:
        outpath = os.path.join(
            problem.outfolder,
            po.figure_dir,
            "slip_dist_%i_%s_%i" % (stage_number, llk_str, po.nensemble),
        )

        logger.info("Storing slip-distribution to: %s" % outpath)
        if po.outformat == "pdf":
            with PdfPages(outpath + ".pdf") as opdf:
                for fig in figs:
                    opdf.savefig(fig, dpi=po.dpi)
        else:
            for i, fig in enumerate(figs):
                fig.savefig(outpath + "_%i.%s" % (i, po.outformat), dpi=po.dpi)


def draw_3d_slip_distribution(problem, po):

    varname_choices = ["coupling", "euler_slip", "slip_variation"]

    if po.outformat == "svg":
        raise NotImplementedError("SVG format is not supported for this plot!")

    mode = problem.config.problem_config.mode

    if mode != ffi_mode_str:
        raise ModeError(
            "Wrong optimization mode: %s! This plot "
            'variant is only valid for "%s" mode' % (mode, ffi_mode_str)
        )

    if po.load_stage is None:
        po.load_stage = -1

    stage = load_stage(problem, stage_number=po.load_stage, load="trace", chains=[-1])

    if not po.reference:
        reference = problem.config.problem_config.get_test_point()
        res_point = get_result_point(stage.mtrace, po.post_llk)
        reference.update(res_point)
        llk_str = po.post_llk
        mtrace = stage.mtrace
    else:
        reference = po.reference
        llk_str = "ref"
        mtrace = None

    datatype, cconf = list(problem.composites.items())[0]

    fault = cconf.load_fault_geometry()

    if po.plot_projection in ["local", "latlon"]:
        perspective = "135/30"
    else:
        perspective = po.plot_projection

    gc = problem.config.geodetic_config
    if gc:
        for corr in gc.corrections_config.euler_poles:
            if corr.enabled:
                if len(po.varnames) > 0 and po.varnames[0] in varname_choices:
                    from beat.ffi import euler_pole2slips

                    logger.info("Plotting %s ...!", po.varnames[0])
                    reference["euler_slip"] = euler_pole2slips(
                        point=reference, fault=fault, event=problem.config.event
                    )

                    # TODO: cleanup iforgy with slip units etc ...
                    if po.varnames[0] == "coupling":
                        slip_units = "%"
                    else:
                        slip_units = "m/yr"
                else:
                    logger.info(
                        "Found Euler pole correction assuming interseismic "
                        "slip-rates ..."
                    )
                    slip_units = "m/yr"
            else:
                logger.info(
                    "Did not find Euler pole correction-assuming " "co-seismic slip ..."
                )
                slip_units = "m"

        if po.varnames[0] == "slip_variation":
            from pandas import read_csv

            from beat.backend import extract_bounds_from_summary

            summarydf = read_csv(
                os.path.join(problem.outfolder, "summary.txt"), sep="\s+"
            )
            bounds = extract_bounds_from_summary(
                summarydf, varname="uparr", shape=(fault.npatches,)
            )
            reference["slip_variation"] = bounds[1] - bounds[0]
            slip_units = "m"

    if len(po.varnames) == 0:
        varnames = None
    else:
        varnames = po.varnames

    if len(po.varnames) == 1:
        slip_label = po.varnames[0]
    else:
        slip_label = "slip"

    if po.source_idxs is None:
        source_idxs = [0, fault.nsubfaults]
    else:
        source_idxs = po.source_idxs

    outpath = os.path.join(
        problem.outfolder,
        po.figure_dir,
        "3d_%s_distribution_%i_%s_%i.%s"
        % (slip_label, po.load_stage, llk_str, po.nensemble, po.outformat),
    )

    if not os.path.exists(outpath) or po.force or po.outformat == "display":
        logger.info("Drawing 3d slip-distribution plot ...")

        gmt = slip_distribution_3d_gmt(
            fault,
            reference,
            mtrace,
            perspective,
            slip_units,
            slip_label,
            varnames,
            source_idxs=source_idxs,
        )

        logger.info("saving figure to %s" % outpath)
        gmt.save(outpath, resolution=300, size=10)
    else:
        logger.info("Plot exists! Use --force to overwrite!")


def slip_distribution_3d_gmt(
    fault,
    reference,
    mtrace=None,
    perspective="135/30",
    slip_units="m",
    slip_label="slip",
    varnames=None,
    gmt=None,
    bin_width=1,
    cptfilepath=None,
    transparency=0,
    source_idxs=None,
):

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError("GMT needs to be installed for station_map plot!")

    p = "z%s/0" % perspective
    # bin_width = 1  # major grid and tick increment in [deg]

    if gmt is None:
        font_size = 12
        font = "1"
        h = 15  # outsize in cm
        w = 22

        gmtconfig = get_gmt_config(gmtpy, h=h, w=w, fontsize=11)

        gmtconfig["MAP_FRAME_TYPE"] = "plain"
        gmtconfig["MAP_SCALE_HEIGHT"] = "11p"
        # gmtconfig.pop('PS_MEDIA')

        gmt = gmtpy.GMT(config=gmtconfig)

    sf_lonlats = num.vstack(
        [sf.outline(cs="lonlat") for sf in fault.iter_subfaults(source_idxs)]
    )

    sf_xyzs = num.vstack(
        [sf.outline(cs="xyz") for sf in fault.iter_subfaults(source_idxs)]
    )
    _, _, max_depth = sf_xyzs.max(axis=0) / km

    lon_min, lat_min = sf_lonlats.min(axis=0)
    lon_max, lat_max = sf_lonlats.max(axis=0)

    lon_tolerance = (lon_max - lon_min) * 0.1
    lat_tolerance = (lat_max - lat_min) * 0.1

    R = utility.list2string(
        [
            lon_min - lon_tolerance,
            lon_max + lon_tolerance,
            lat_min - lat_tolerance,
            lat_max + lat_tolerance,
            -max_depth,
            0,
        ],
        "/",
    )
    Jg = "-JM%fc" % 20
    Jz = "-JZ%gc" % 3
    J = [Jg, Jz]

    B = [
        "-Bxa%gg%g" % (bin_width, bin_width),
        "-Bya%gg%g" % (bin_width, bin_width),
        "-Bza10+Ldepth [km]",
        "-BWNesZ",
    ]
    args = J + B

    gmt.pscoast(R=R, D="a", G="gray90", S="lightcyan", p=p, *J)

    gmt.psbasemap(R=R, p=p, *args)

    if slip_label == "coupling":
        from beat.ffi import backslip2coupling

        euler_slips = reference["euler_slip"]
        reference_slips = backslip2coupling(reference, euler_slips)

    elif slip_label == "euler_slip":
        reference_slips = reference["euler_slip"]

    elif slip_label == "slip_variation":
        reference_slips = reference[slip_label]

    else:
        reference_slips = fault.get_total_slip(
            index=None, point=reference, components=varnames
        )

    autos = AutoScaler(snap="on", approx_ticks=3)

    cmin, cmax, cinc = autos.make_scale(
        (0, reference_slips.max()), override_mode="min-max"
    )

    if cptfilepath is None:
        cptfilepath = "/tmp/tempfile.cpt"
        gmt.makecpt(
            C="hot",
            I="c",
            T="%f/%f" % (cmin, cmax),
            out_filename=cptfilepath,
            suppress_defaults=True,
        )

    tmp_patch_fname = "/tmp/temp_patch.txt"

    for idx in range(*source_idxs):
        slips = fault.vector2subfault(index=idx, vector=reference_slips)
        for i, source in enumerate(fault.get_subfault_patches(idx)):
            lonlats = source.outline(cs="lonlat")
            xyzs = source.outline(cs="xyz") / km
            depths = xyzs[:, 2] * -1.0  # make depths negative
            in_rows = num.hstack((lonlats, num.atleast_2d(depths).T))

            num.savetxt(
                tmp_patch_fname, in_rows, header="> -Z%f" % slips[i], comments=""
            )

            gmt.psxyz(
                tmp_patch_fname,
                R=R,
                C=cptfilepath,
                L=True,
                t=transparency,
                W="0.1p",
                p=p,
                *J
            )

    # add a colorbar

    azimuth, elev_angle = perspective.split("/")

    if float(azimuth) < 180:
        ypos = 0
    else:
        ypos = 10

    D = "x1.5c/%ic+w6c/0.5c+jMC+h" % ypos
    F = False

    gmt.psscale(
        B="xa%f +l %s [%s]" % (cinc, slip_label, slip_units),
        D=D,
        F=F,
        C=cptfilepath,
        finish=True,
    )

    return gmt
