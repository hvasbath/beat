from matplotlib import pyplot as plt
from pyrocko.plot import mpl_papersize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator

from beat.bem import slip_comp_to_idx
import numpy as num
from .common import set_locator_axes, scale_axes, set_axes_equal_3d

km = 1000.0


def cb_round(value, decimal=3):
    return num.round(value, decimal)


def slip_distribution_3d(
    discretized_sources, slip_vectors, perspective="150/30", debug=False
):
    # fontsize_title = 12
    fontsize = 8

    camera = [float(angle) for angle in perspective.split("/")]

    fig = plt.figure(figsize=mpl_papersize("a5", "landscape"))
    slip_comps = ["strike", "dip", "normal"]

    axs = []
    sources_coord_limits = (
        num.dstack(
            [dsource.get_minmax_triangles_xyz() for dsource in discretized_sources]
        )
        / km
    )
    min_limits = num.floor(sources_coord_limits.min(2).min(1)) * km
    max_limits = num.ceil(sources_coord_limits.max(2).max(1)) * km
    for j, comp in enumerate(slip_comps):
        cmap = plt.get_cmap("hot") if comp == "normal" else plt.get_cmap("seismic")
        ax = fig.add_subplot(
            1, len(slip_comps), j + 1, projection="3d", computed_zorder=False
        )
        for k, (dsource, slips3d) in enumerate(zip(discretized_sources, slip_vectors)):
            pa_col = Poly3DCollection(
                dsource.triangles_xyz,
            )

            a = slips3d[:, slip_comp_to_idx[comp]]

            if comp in ["strike", "dip"]:
                absmax = num.max([num.abs(a.min()), a.max()])
                cbounds = [-cb_round(absmax), cb_round(absmax)]
            else:
                cbounds = [cb_round(a.min()), cb_round(a.max())]

            assert a.size == dsource.n_triangles

            ax.add_collection(pa_col)

            if num.diff(cbounds) == 0:
                colors = ["white" for _ in range(a.size)]
                pa_col.set_facecolor(colors)
                pa_col.set(edgecolor="k", linewidth=0.1, alpha=0.25)
            else:
                cbl = 0.1 + j * 0.3
                cbb = 0.2 - k * 0.08
                cbw = 0.15
                cbh = 0.01

                cbaxes = fig.add_axes([cbl, cbb, cbw, cbh])

                pa_col.set_cmap(cmap)

                pa_col.set_array(a)
                pa_col.set_clim(*cbounds)
                pa_col.set(edgecolor="k", linewidth=0.2, alpha=0.75)

                cbs = plt.colorbar(
                    pa_col,
                    ax=ax,
                    ticks=cbounds,
                    cax=cbaxes,
                    orientation="horizontal",
                )
                cbs.set_label(f"{comp}-slip [m]", fontsize=fontsize)
                cbs.ax.tick_params(labelsize=fontsize)

                unit_vectors = getattr(dsource, f"unit_{comp}_vectors")

                ax.quiver(
                    dsource.centroids[::3, 0],
                    dsource.centroids[::3, 1],
                    dsource.centroids[::3, 2],
                    unit_vectors[::3, 0],
                    unit_vectors[::3, 1],
                    unit_vectors[::3, 2],
                    color="k",
                    length=dsource.mesh_size,
                    linewidth=1.0,
                )

                if False:
                    # plot vector normals for debugging
                    unit_vectors = getattr(dsource, "unit_normal_vectors")

                    ax.quiver(
                        dsource.centroids[::, 0],
                        dsource.centroids[::, 1],
                        dsource.centroids[::, 2],
                        unit_vectors[::, 0],
                        unit_vectors[::, 1],
                        unit_vectors[::, 2],
                        color="k",
                        length=dsource.mesh_size,
                        normalize=True,
                        linewidth=1.0,
                    )
            if debug:
                for tri_idx in range(dsource.n_triangles):
                    ax.text(
                        dsource.centroids[tri_idx, 0],
                        dsource.centroids[tri_idx, 1],
                        dsource.centroids[tri_idx, 2],
                        tri_idx,
                        fontsize=6,
                    )

        ax.tick_params(labelsize=fontsize, rotation=-30)
        if j == 2:
            ax.set_xlabel("East-Distance [km]", fontsize=fontsize)
            ax.set_ylabel("North-Distance [km]", fontsize=fontsize)
            ax.set_zlabel("Depth [km]", fontsize=fontsize)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        ax.set_xlim([min_limits[0], max_limits[0]])
        ax.set_ylim([min_limits[1], max_limits[1]])
        ax.set_zlim([min_limits[2], max_limits[2]])

        set_axes_equal_3d(ax, axes="xyz")

        set_locator_axes(ax.get_xaxis(), MaxNLocator(nbins=3))
        set_locator_axes(ax.get_yaxis(), MaxNLocator(nbins=3))
        set_locator_axes(ax.get_zaxis(), MaxNLocator(nbins=3))

        scale = {"scale": 1 / km}

        scale_axes(ax.get_xaxis(), **scale)
        scale_axes(ax.get_yaxis(), **scale)
        scale_axes(ax.get_zaxis(), **scale)

        #     ax.invert_zaxis()
        # ax.set_aspect(1)
        ax.view_init(*camera[::-1])

    fig.subplots_adjust(
        left=0.03,
        right=1.0 - 0.08,
        bottom=0.06,
        top=1.0 - 0.06,
        wspace=0.0,
        hspace=0.1,
    )
    # fig.tight_layout()
    return fig, axs
