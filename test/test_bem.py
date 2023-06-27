import logging
import unittest

import numpy as num

from numpy.testing import assert_allclose
from pyrocko import util
from pyrocko.gf.targets import StaticTarget

from beat.bem import BEMEngine, RingfaultBEMSource, DiskBEMSource, slip_comp_to_idx
from beat.config import BEMConfig
from matplotlib import pyplot as plt

km = 1.0e3
pi = num.pi
logger = logging.getLogger("test_sources")


def get_static_target(bounds, npoints=100):
    xs = num.linspace(*bounds, npoints)
    ys = num.linspace(*bounds, npoints)
    obsx, obsy = num.meshgrid(xs, ys)
    return StaticTarget(east_shifts=obsx.ravel(), north_shifts=obsy.ravel())


def plot_static_result(result, target, npoints=100):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), dpi=300)
    for i, comp in enumerate(["n", "e", "d"]):
        ax = axs[i]
        disp_grid = result[f"displacement.{comp}"].reshape((npoints, npoints))
        grd_e = target.east_shifts.reshape((npoints, npoints))
        grd_n = target.north_shifts.reshape((npoints, npoints))
        cntf = ax.contourf(grd_e, grd_n, disp_grid, levels=21)
        ax.contour(
            grd_e,
            grd_n,
            disp_grid,
            colors="k",
            linestyles="-",
            linewidths=0.5,
            levels=21,
        )
        print("displ min max", disp_grid.min(), disp_grid.max())
        ax.set_title(f"$u_{comp}$")

    cb = plt.colorbar(cntf)
    cb.set_label("Displacement [m]")
    fig.tight_layout()
    return fig, axs


def plot_sources_3d(discretized_sources, slip_vectors):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(15, 5), dpi=300)
    slip_comps = ["strike", "dip", "tensile"]

    axs = []
    for j, comp in enumerate(slip_comps):
        cmap = plt.get_cmap("hot") if comp == "tensile" else plt.get_cmap("seismic")
        ax = fig.add_subplot(1, len(slip_comps), j + 1, projection="3d")

        for dsource, slips3d in zip(discretized_sources, slip_vectors):
            pa_col = Poly3DCollection(
                dsource.triangles_xyz,
                cmap=cmap,
                rasterized=True,
            )
            a = slips3d[:, slip_comp_to_idx[comp]]

            if comp == "strike":
                absmax = num.max([num.abs(a.min()), a.max()])
                cbounds = [-absmax, absmax]
            else:
                cbounds = [a.min(), a.max()]

            assert a.size == dsource.n_triangles

            pa_col.set_array(a)
            pa_col.set_clim(*cbounds)
            pa_col.set(edgecolor="k", linewidth=0.5, alpha=0.8)

            ax.add_collection(pa_col)

        cbs = plt.colorbar(
            pa_col,
            ax=ax,
            orientation="vertical",
        )
        cbs.set_label("Slip [m]")

        ax.set_title(f"${comp}-slip$")
        ax.set_xlabel("East- Distance [km]")
        ax.set_ylabel("North- Distance [km]")
        ax.set_zlabel("Depth [km]")

        ax.set_xlim([-3000, 5000])
        ax.set_ylim([-3000, 5000])
        ax.set_zlim([-1000, -5000])
        ax.invert_zaxis()
        axs.append(ax)

    fig.tight_layout()
    return fig, axs


def get_disk_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        DiskBEMSource(
            tensile_traction=2.15e9,
            north_shift=0.5 * km,
            depth=3.5 * km,
            major_axis=2 * km,
            dip=10,
            strike=30,
        )
    ]
    return BEMConfig(), sources, targets


def get_disk_ringfault_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        DiskBEMSource(
            tensile_traction=2.15e9,
            north_shift=0.5 * km,
            east_shift=3.5 * km,
            depth=4.0 * km,
            major_axis=2 * km,
            dip=10,
            strike=40,
        ),
        RingfaultBEMSource(
            north_shift=0.0,
            delta_north_shift_bottom=0.5 * km,
            east_shift=3.55 * km,
            depth=0.5 * km,
            delta_depth_bottom=4.0 * km,
            major_axis=2 * km,
            minor_axis=1 * km,
            major_axis_bottom=1.5 * km,
            strike=45,
        ),
    ]
    config = BEMConfig()
    for bcond in config.boundary_conditions.iter_conditions():
        bcond.receiver_idxs = [0, 1]
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = [1]

    return config, sources, targets


class TestBEM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def _run_bem_engine(self, setup_function, plot=False):
        config, sources, targets = setup_function()

        engine = BEMEngine(config)
        response = engine.process(sources=sources, targets=targets)

        results = response.static_results()
        if plot:
            for i, result in enumerate(results):
                plot_static_result(result.result, targets[i])

            slip_vectors = response.source_slips()
            plot_sources_3d(response.discretized_sources, slip_vectors)
            plt.show()

    def test_bem_engine_dike(self):
        self._run_bem_engine(get_disk_setup)

    def test_bem_engine_dike_ringfault(self):
        self._run_bem_engine(get_disk_ringfault_setup)


if __name__ == "__main__":
    util.setup_logging("test_bem", "info")
    unittest.main()
