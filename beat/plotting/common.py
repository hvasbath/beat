import logging
import os

import numpy as num
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pymc3 import quantiles
from pyrocko import orthodrome as otd
from pyrocko.guts import Bool, Dict, Int, List, Object, String, StringChoice, load
from pyrocko.plot import mpl_graph_color, mpl_papersize
from scipy.stats import kde
from theano import config as tconfig

from beat import utility

logger = logging.getLogger("plotting.common")

km = 1000.0

u_nm = "$[Nm]$"
u_km = "$[km]$"
u_km_s = "$[km/s]$"
u_deg = "$[^{\circ}]$"
u_deg_myr = "$[^{\circ} / myr]$"
u_m = "$[m]$"
u_v = "$[m^3]$"
u_s = "$[s]$"
u_rad = "$[rad]$"
u_hyp = ""
u_percent = "[$\%$]"
u_nanostrain = "nstrain"

plot_units = {
    "east_shift": u_km,
    "north_shift": u_km,
    "depth": u_km,
    "width": u_km,
    "length": u_km,
    "dip": u_deg,
    "dip1": u_deg,
    "dip2": u_deg,
    "strike": u_deg,
    "strike1": u_deg,
    "strike2": u_deg,
    "rake": u_deg,
    "rake1": u_deg,
    "rake2": u_deg,
    "mix": u_hyp,
    "volume_change": u_v,
    "diameter": u_km,
    "slip": u_m,
    "opening_fraction": u_hyp,
    "azimuth": u_deg,
    "bl_azimuth": u_deg,
    "amplitude": u_nm,
    "bl_amplitude": u_m,
    "locking_depth": u_km,
    "nucleation_dip": u_km,
    "nucleation_strike": u_km,
    "nucleation_x": u_hyp,
    "nucleation_y": u_hyp,
    "time_shift": u_s,
    "coupling": u_percent,
    "uperp": u_m,
    "uparr": u_m,
    "utens": u_m,
    "durations": u_s,
    "velocities": u_km_s,
    "mnn": u_nm,
    "mee": u_nm,
    "mdd": u_nm,
    "mne": u_nm,
    "mnd": u_nm,
    "med": u_nm,
    "magnitude": u_hyp,
    "eps_xx": u_nanostrain,
    "eps_yy": u_nanostrain,
    "eps_xy": u_nanostrain,
    "rotation": u_nanostrain,
    "pole_lat": u_deg,
    "pole_lon": u_deg,
    "omega": u_deg_myr,
    "w": u_rad,
    "v": u_rad,
    "kappa": u_rad,
    "sigma": u_rad,
    "h": u_hyp,
    "distance": u_km,
    "delta_depth": u_km,
    "delta_time": u_s,
    "time": u_s,
    "duration": u_s,
    "peak_ratio": u_hyp,
    "h_": u_hyp,
    "like": u_hyp,
}


plot_projections = ["latlon", "local", "individual"]


def get_matplotlib_version():
    from matplotlib import __version__ as mplversion

    return float(mplversion[0]), float(mplversion[2:])


def cbtick(x):
    rx = num.floor(x * 1000.0) / 1000.0
    return [-rx, rx]


def plot_cov(target, point_size=20):

    ax = plt.axes()
    im = ax.scatter(
        target.lons,
        target.lats,
        point_size,
        num.array(target.covariance.pred_v.sum(axis=0)).flatten(),
        edgecolors="none",
    )
    plt.colorbar(im)
    plt.title("Prediction Covariance [m2] %s" % target.name)
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
    mask[cov_mat < 0] = -1.0
    im = ax.imshow(num.multiply(num.log(num.abs(cov_mat)), mask))
    plt.colorbar(im)
    plt.show()


def get_gmt_config(gmtpy, fontsize=14, h=20.0, w=20.0):

    if gmtpy.is_gmt5(version="newest"):
        gmtconfig = {
            "MAP_GRID_PEN_PRIMARY": "0.1p",
            "MAP_GRID_PEN_SECONDARY": "0.1p",
            "MAP_FRAME_TYPE": "fancy",
            "FONT_ANNOT_PRIMARY": "%ip,Helvetica,black" % fontsize,
            "FONT_ANNOT_SECONDARY": "%ip,Helvetica,black" % fontsize,
            "FONT_LABEL": "%ip,Helvetica,black" % fontsize,
            "FORMAT_GEO_MAP": "D",
            "GMT_TRIANGULATE": "Watson",
            "PS_MEDIA": "Custom_%ix%i" % (w * gmtpy.cm, h * gmtpy.cm),
        }
    else:
        gmtconfig = {
            "MAP_FRAME_TYPE": "fancy",
            "GRID_PEN_PRIMARY": "0.01p",
            "ANNOT_FONT_PRIMARY": "1",
            "ANNOT_FONT_SIZE_PRIMARY": "12p",
            "PLOT_DEGREE_FORMAT": "D",
            "GRID_PEN_SECONDARY": "0.01p",
            "FONT_LABEL": "%ip,Helvetica,black" % fontsize,
            "PS_MEDIA": "Custom_%ix%i" % (w * gmtpy.cm, h * gmtpy.cm),
        }
    return gmtconfig


def hypername(varname):
    if varname in list(plot_units.keys()):
        return varname
    else:
        return "h_"


class PlotOptions(Object):
    post_llk = String.T(
        default="max",
        help='Which model to plot on the specified plot; Default: "max";'
        ' Options: "max", "min", "mean", "all"',
    )
    plot_projection = StringChoice.T(
        default="local",
        choices=plot_projections,
        help='Projection to use for plotting geodetic data; options: "latlon"',
    )
    utm_zone = Int.T(
        default=36, optional=True, help='Only relevant if plot_projection is "utm"'
    )
    load_stage = Int.T(default=-1, help="Which stage to select for plotting")
    figure_dir = String.T(
        default="figures", help="Name of the output directory of plots"
    )
    reference = Dict.T(
        default={},
        help="Reference point for example from a synthetic test.",
        optional=True,
    )
    outformat = String.T(default="pdf")
    dpi = Int.T(default=300)
    force = Bool.T(default=False)
    varnames = List.T(default=[], optional=True, help="Names of variables to plot")
    source_idxs = List.T(
        default=None,
        optional=True,
        help="Indexes to patches of slip distribution to draw marginals for",
    )
    nensemble = Int.T(
        default=1, help="Number of draws from the PPD to display fuzzy results."
    )


def str_unit(quantity):
    """
    Return string representation of waveform unit.
    """
    if quantity == "displacement":
        return "$m$"
    elif quantity == "velocity":
        return "$m/s$"
    elif quantity == "acceleration":
        return "$m/s^2$"
    else:
        raise ValueError("Quantity %s not supported!" % quantity)


def str_dist(dist):
    """
    Return string representation of distance.
    """
    if dist < 10.0:
        return "%g m" % dist
    elif 10.0 <= dist < 1.0 * km:
        return "%.0f m" % dist
    elif 1.0 * km <= dist < 10.0 * km:
        return "%.1f km" % (dist / km)
    else:
        return "%.0f km" % (dist / km)


def str_duration(t):
    """
    Convert time to str representation.
    """
    from pyrocko import util

    s = ""
    if t < 0.0:
        s = "-"

    t = abs(t)

    if t < 60.0:
        return s + "%.2g s" % t
    elif 60.0 <= t < 3600.0:
        return s + util.time_to_str(t, format="%M:%S min")
    elif 3600.0 <= t < 24 * 3600.0:
        return s + util.time_to_str(t, format="%H:%M h")
    else:
        return s + "%.1f d" % (t / (24.0 * 3600.0))


def get_result_point(mtrace, point_llk="max"):
    """
    Return Point dict from multitrace

    Parameters
    ----------
    mtrace: pm.MultiTrace
        sampled result trace containing the posterior ensemble
    point_llk: str
        returning according point with 'max', 'min', 'mean' likelihood

    Returns
    -------
    point: dict
        keys varnames, values numpy ndarrays
    """

    if point_llk != "None":
        llk = mtrace.get_values(varname="like", combine=True)

        posterior_idxs = utility.get_fit_indexes(llk)

        point = mtrace.point(idx=posterior_idxs[point_llk])

    else:
        point = None

    return point


def histplot_op(
    ax,
    data,
    reference=None,
    alpha=0.35,
    color=None,
    cmap=None,
    bins=None,
    tstd=None,
    qlist=[0.01, 99.99],
    cbounds=None,
    kwargs={},
):
    """
    Modified from pymc3. Additional color argument.
    """

    cumulative = kwargs.pop("cumulative", False)
    nsources = kwargs.pop("nsources", False)
    isource = kwargs.pop("isource", 0)

    if color is not None and cmap is not None:
        logger.debug("Using color for histogram edgecolor ...")

    if cmap is not None:
        from matplotlib.colors import Colormap

        if not isinstance(cmap, Colormap):
            raise TypeError("The colormap needs to be a valid matplotlib colormap!")

        histtype = "bar"
    else:
        if not cumulative:
            histtype = "stepfilled"
        else:
            histtype = "step"

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

        if bins is None:
            step = ((maxd - mind) / 40).astype(tconfig.floatX)

            if step == 0:
                step = num.finfo(tconfig.floatX).eps

            bins = int(num.ceil((maxd - mind) / step))
            if bins == 0:
                bins = 10

        major, minor = get_matplotlib_version()
        if major < 3:
            kwargs["normed"] = True
        else:
            kwargs["density"] = True

        n, outbins, patches = ax.hist(
            d,
            bins=bins,
            stacked=True,
            alpha=alpha,
            align="left",
            histtype=histtype,
            color=color,
            edgecolor=color,
            cumulative=cumulative,
            **kwargs,
        )

        if cmap:
            bin_centers = 0.5 * (outbins[:-1] + outbins[1:])
            if cbounds is None:
                col = bin_centers - min(bin_centers)
                col /= max(col)
            else:
                col = (bin_centers - cbounds[0]) / (cbounds[1] - cbounds[0])

            for c, p in zip(col, patches):
                plt.setp(p, "facecolor", cmap(c))

        left, right = ax.get_xlim()
        leftb = mind - tstd
        rightb = maxd + tstd

        if left != 0.0 or right != 1.0:
            leftb = num.minimum(leftb, left)
            rightb = num.maximum(rightb, right)

        logger.debug("Histogram bounds: left %f, right %f", leftb, rightb)
        ax.set_xlim(leftb, rightb)
        if cumulative:
            # need left plot bound, leftb
            sigma_quants = quantiles(d, [5, 68, 95])

            for quantile, value in sigma_quants.items():
                quantile /= 100.0
                if nsources == 1:
                    x = [leftb, value, value]
                    y = [quantile, quantile, 0.0]
                else:
                    x = [leftb, rightb]
                    y = [quantile, quantile]

                fontsize = 6

                if isource + 1 == nsources:
                    # plot for last hist in axis
                    ax.plot(x, y, "--k", linewidth=0.5)

                    xval = (value - leftb) / 2 + leftb

                    ax.text(
                        xval,
                        quantile,
                        "{}%".format(int(quantile * 100)),
                        fontsize=fontsize,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                    )

                if nsources == 1:
                    ax.text(
                        value,
                        quantile / 2,
                        "%.3f" % value,
                        fontsize=fontsize,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )


def kde2plot_op(ax, x, y, grid=200, **kwargs):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    extent = kwargs.pop("extent", [])
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
    lats0, lons0, lats=None, lons=None, grid_size=(200, 200), sigma=None
):

    from beat.models.distributions import vonmises_fisher, vonmises_std

    if sigma is None:
        logger.debug("No sigma given, estimating VonMisesStd ...")
        sigmahat = vonmises_std(lats=lats0, lons=lons0)
        sigma = 1.06 * sigmahat * lats0.size**-0.2
        logger.debug("suggested sigma: %f, sigmahat: %f" % (sigma, sigmahat))

    if lats is None and lons is None:
        lats_vec = num.linspace(-90.0, 90, grid_size[0])
        lons_vec = num.linspace(-180.0, 180, grid_size[1])

        lons, lats = num.meshgrid(lons_vec, lats_vec)

    if lats is not None:
        assert lats.size == lons.size

    batch_size = 500
    cycles, rest = utility.mod_i(lats0.size, batch_size)

    if rest != 0:
        logger.debug("Processing rest of spherical kde samples %i" % (rest))
        vmf = vonmises_fisher(
            lats=lats, lons=lons, lats0=lats0[-rest:], lons0=lons0[-rest:], sigma=sigma
        )
        kde = (
            num.exp(vmf)
            .sum(axis=-1)
            .reshape((grid_size[0], grid_size[1]))  # , b=self.weights)
        )
    else:
        logger.debug("Init new spherical kde samples")
        kde = num.zeros(grid_size)

    logger.info("Drawing lune plot for %i samples ... " % lats0.size)
    for cyc in range(cycles):
        cyc_s = cyc * batch_size
        cyc_e = cyc_s + batch_size
        logger.debug("Looping over spherical kde samples %i - %i" % (cyc_s, cyc_e))

        vmf = vonmises_fisher(
            lats=lats,
            lons=lons,
            lats0=lats0[cyc_s:cyc_e],
            lons0=lons0[cyc_s:cyc_e],
            sigma=sigma,
        )
        kde += num.exp(vmf).sum(axis=-1)

    return kde, lats, lons


def format_axes(ax, remove=["right", "top", "left"], linewidth=None, visible=False):
    """
    Removes box top, left and right.
    """
    for rm in remove:
        ax.spines[rm].set_visible(visible)
        if linewidth is not None:
            ax.spines[rm].set_linewidth(linewidth)


def scale_axes(axis, scale, offset=0.0):
    from matplotlib.ticker import ScalarFormatter

    class FormatScaled(ScalarFormatter):
        @staticmethod
        def __call__(value, pos):
            return "{:,.1f}".format(offset + value * scale).replace(",", " ")

    axis.set_major_formatter(FormatScaled())


def set_anchor(sources, anchor):
    for source in sources:
        source.anchor = anchor


def get_gmt_colorstring_from_mpl(i):
    color = (num.array(mpl_graph_color(i)) * 255).tolist()
    return utility.list2string(color, "/")


def get_latlon_ratio(lat, lon):
    """
    Get latlon ratio at given location
    """
    dlat_meters = otd.distance_accurate50m(lat, lon, lat - 1.0, lon)
    dlon_meters = otd.distance_accurate50m(lat, lon, lat, lon - 1.0)
    return dlat_meters / dlon_meters


def plot_inset_hist(
    axes,
    data,
    best_data,
    bbox_to_anchor,
    linewidth=0.5,
    labelsize=5,
    cmap=None,
    cbounds=None,
    color="orange",
    alpha=0.4,
    background_alpha=1.0,
):

    in_ax = inset_axes(
        axes,
        width="100%",
        height="100%",
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=axes.transAxes,
        loc=2,
        borderpad=0,
    )
    histplot_op(
        in_ax, data, alpha=alpha, color=color, cmap=cmap, cbounds=cbounds, tstd=0.0
    )

    format_axes(in_ax)
    format_axes(in_ax, remove=["bottom"], visible=True, linewidth=linewidth)

    if best_data:
        in_ax.axvline(x=best_data, color="red", lw=linewidth)

    in_ax.tick_params(axis="both", direction="in", labelsize=labelsize, width=linewidth)
    in_ax.yaxis.set_visible(False)
    xticker = MaxNLocator(nbins=2)
    in_ax.xaxis.set_major_locator(xticker)
    in_ax.patch.set_alpha(background_alpha)
    return in_ax


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
        return num.clip(num.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)

    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = _weighted_line(
            c0, r0, c1, r1, w=w, rmin=cmin, rmax=cmax, cmin=rmin, cmax=rmax
        )
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return _weighted_line(
            r1, c1, r0, c0, w=w, rmin=rmin, rmax=rmax, cmin=cmin, cmax=cmax
        )

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

    yy = num.floor(y).reshape(-1, 1) + num.arange(
        -thickness - 1, thickness + 2
    ).reshape(1, -1)
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
    X, Y, grid=None, extent=[], grid_resolution=(400, 400), linewidth=1
):
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
                "Gridsize of given grid is inconistent for axis %i!"
                " Expected %i got %i" % (axis, naim, ngr)
            )

    def check_line_in_grid(idxs, axis, nmax, extent):
        imax = idxs.max()
        if imax > nmax:
            raise TypeError(
                'Line endpoint outside of given grid Axis "%s"! %i > %i'
                " Extent [%s]" % (axis, imax, nmax, utility.list2string(extent))
            )

    nxs = len(X)
    nys = len(Y)
    if nxs != nys:
        raise TypeError("Length of X and Y have to be identical! %i != %i" % (nxs, nys))

    if len(extent) == 0:
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        extent = [xmin, xmax, ymin, ymax]
    elif len(extent) == 4:
        xmin, xmax, ymin, ymax = extent
    else:
        raise TypeError("extent has to be of length 4! [xmin, xmax, ymin, ymax]")

    if len(grid_resolution) != 2:
        raise TypeError("grid_resolution has to be of length 2! [xstep, ystep]!")

    ynstep, xnstep = grid_resolution

    xvec, xstep = num.linspace(xmin, xmax, xnstep, endpoint=True, retstep=True)
    yvec, ystep = num.linspace(ymin, ymax, ynstep, endpoint=True, retstep=True)

    if grid is not None:
        if grid.ndim != 2:
            raise TypeError("Given grid has to be of dimension 2!")

        for axis, (ngr, naim) in enumerate(zip(grid.shape, grid_resolution)):
            check_grid_shape(ngr, naim, axis)
    else:
        grid = num.zeros((ynstep, xnstep), dtype="float64")

    xidxs = utility.positions2idxs(X, min_pos=xmin, cell_size=xstep, dtype="int32")
    yidxs = utility.positions2idxs(Y, min_pos=ymin, cell_size=ystep, dtype="int32")

    check_line_in_grid(xidxs, "x", nmax=xnstep - 1, extent=extent)
    check_line_in_grid(yidxs, "y", nmax=ynstep - 1, extent=extent)
    new_grid = num.zeros_like(grid)
    for i in range(1, nxs):
        c0 = xidxs[i - 1]
        r0 = yidxs[i - 1]
        c1 = xidxs[i]
        r1 = yidxs[i]
        try:
            rr, cc, w = _weighted_line(
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                w=linewidth,
                rmax=ynstep - 1,
                cmax=xnstep - 1,
            )
            new_grid[rr, cc] = w.astype(grid.dtype)
        except ValueError:
            # line start and end fall in the same grid point can't be drawn
            pass

    grid += new_grid
    return grid, extent


def get_nice_plot_bounds(dmin, dmax, override_mode="min-max"):
    """
    Get nice min, max and increment for plots
    """
    from pyrocko.plot import AutoScaler, nice_value

    inc = nice_value(dmax - dmin)
    autos = AutoScaler(inc=inc, snap="on", approx_ticks=2)
    return autos.make_scale((dmin, dmax), override_mode=override_mode)


def plot_covariances(datasets, covariances):

    cmap = plt.get_cmap("seismic")

    ndata = len(covariances)

    fontsize = 10
    ndmax = 3

    fullfig, restfig = utility.mod_i(ndata, ndmax)
    factors = num.ones(fullfig).tolist()
    if restfig:
        factors.append(float(restfig) / ndmax)

    figures = []
    axes = []
    for f in factors:
        figsize = list(mpl_papersize("a4", "portrait"))
        figsize[1] *= f

        fig, ax = plt.subplots(nrows=int(round(ndmax * f)), ncols=2, figsize=figsize)
        fig.tight_layout()
        fig.subplots_adjust(
            left=0.08,
            right=1.0 - 0.03,
            bottom=0.05,
            top=1.0 - 0.03,
            wspace=0.2,
            hspace=0.25,
        )
        figures.append(fig)
        ax_a = num.atleast_2d(ax)
        axes.append(ax_a)

    cbl = 0.76
    cbh = 0.01
    cbw = 0.15

    for kidx, (cov, dataset) in enumerate(zip(covariances, datasets)):

        figidx, rowidx = utility.mod_i(kidx, ndmax)
        axs = axes[figidx][rowidx, :]

        f = factors[figidx]
        if f > 2.0 / 3:
            cbb = 0.68 - (0.3075 * rowidx)
        elif f > 1.0 / 2:
            cbb = 0.53 - (0.47 * rowidx)
        elif f > 1.0 / 4:
            cbb = 0.06

        vmin, vmax = cov.get_min_max_components()
        for l, attr in enumerate(["data", "pred_v"]):
            cmat = getattr(cov, attr)
            ax = axs[l]
            if cmat is not None and cmat.sum() != 0.0:

                im = ax.imshow(
                    cmat,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )

                xticker = MaxNLocator(nbins=2)
                yticker = MaxNLocator(nbins=2)
                ax.xaxis.set_major_locator(xticker)
                ax.yaxis.set_major_locator(yticker)
                if l == 0:
                    ax.set_ylabel("Sample idx")
                    ax.set_xlabel("Sample idx")
                    ax.set_title(dataset.name)

                    cbaxes = fig.add_axes([cbl, cbb, cbw, cbh])
                    cblabel = "Covariance [m²]"
                    cbs = plt.colorbar(
                        im,
                        ax=ax,
                        ticks=(vmin, vmax),
                        format=lambda x, _: f"{x:.2e}",
                        cax=cbaxes,
                        orientation="horizontal",
                    )
                    cbs.set_label(cblabel, fontsize=fontsize)
            else:
                logger.info(
                    'Did not find "%s" covariance component for %s', attr, dataset.name
                )
                fig.delaxes(ax)

    return figures, axes


def get_weights_point(composite, best_point, config):

    if composite.config.noise_estimator.structure == "non-toeplitz":
        # nT run is done with test point covariances!
        if config.sampler_config.parameters.update_covariances:
            logger.info("Non-Toeplitz noise structure: Using BestPoint for Covariance!")
            tpoint = best_point
        else:
            logger.info("Non-Toeplitz noise structure: Using TestPoint for Covariance!")
            tpoint = config.problem_config.get_test_point()
    else:
        tpoint = best_point

    return tpoint
