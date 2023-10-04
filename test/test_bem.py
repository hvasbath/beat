import logging
import unittest

import numpy as num

from numpy.testing import assert_allclose
from pyrocko import util
from pyrocko.gf.targets import StaticTarget

from beat.bem import (
    BEMEngine,
    RingfaultBEMSource,
    DiskBEMSource,
    TriangleBEMSource,
    check_intersection,
)
from beat.plotting.bem import slip_distribution_3d
from beat.config import BEMConfig
from matplotlib import pyplot as plt

km = 1.0e3
pi = num.pi
logger = logging.getLogger("test_sources")

mesh_size = 1.0


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
        if comp == "d":
            disp_grid *= -1
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
        # print("displ min max", disp_grid.min(), disp_grid.max())
        ax.set_title(f"$u_{comp}$")

    cb = plt.colorbar(cntf)
    cb.set_label("Displacement [m]")
    fig.tight_layout()
    return fig, axs


def get_triangle_tensile_only_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        TriangleBEMSource(
            tensile_traction=2.15e6,
            p1=(km, km, -2 * km),
            p2=(-km, -0.5 * km, -2 * km),
            p3=(1 * km, -1.5 * km, -2 * km),
        )
    ]

    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = []
    return config, sources, targets


def get_disk_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        DiskBEMSource(
            strike_traction=1.15e6,
            tensile_traction=0,
            north_shift=0.5 * km,
            depth=3.5 * km,
            major_axis=2 * km,
            minor_axis=1.8 * km,
            dip=0,
            strike=30,
        )
    ]
    return BEMConfig(mesh_size=mesh_size), sources, targets


def get_disk_tensile_only_setup():
    targets = [get_static_target([-10 * km, 10 * km], 100)]
    sources = [
        DiskBEMSource(
            tensile_traction=2.15e6,
            north_shift=0.5 * km,
            depth=3.5 * km,
            major_axis=1 * km,
            minor_axis=1.0 * km,
            dip=0,
            strike=30,
        )
    ]
    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = []
    return config, sources, targets


def get_disk_ringfault_setup(intersect=False):
    targets = [get_static_target([-10 * km, 10 * km], 100)]

    if intersect:
        major_axis_bottom = 1.5 * km
        minor_axis_bottom = 1.0 * km
    else:
        major_axis_bottom = 3.5 * km
        minor_axis_bottom = 3.0 * km

    sources = [
        DiskBEMSource(
            tensile_traction=2.15e6,
            north_shift=0.5 * km,
            east_shift=3.5 * km,
            depth=4.0 * km,
            major_axis=2 * km,
            minor_axis=1.8 * km,
            dip=10,
            strike=0,
        ),
        RingfaultBEMSource(
            north_shift=0.0,
            delta_north_shift_bottom=0.5 * km,
            east_shift=3.55 * km,
            depth=0.5 * km,
            delta_depth_bottom=4.0 * km,
            major_axis=2 * km,
            minor_axis=1 * km,
            major_axis_bottom=major_axis_bottom,
            minor_axis_bottom=minor_axis_bottom,
            strike=5,
        ),
    ]

    config = BEMConfig(mesh_size=mesh_size)
    for bcond in config.boundary_conditions.iter_conditions():
        bcond.receiver_idxs = [0, 1]
        if bcond.slip_component in ["strike", "dip"]:
            bcond.source_idxs = [1]

    return config, sources, targets


class TestBEM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def _run_bem_engine(self, setup_function, plot=True, **kwargs):
        print(kwargs)
        config, sources, targets = setup_function(**kwargs)

        engine = BEMEngine(config)
        response = engine.process(sources=sources, targets=targets)

        results = response.static_results()
        if plot and response.is_valid:
            for i, result in enumerate(results):
                plot_static_result(result.result, targets[i])

            slip_vectors = response.source_slips()

            slip_distribution_3d(response.discretized_sources, slip_vectors, debug=True)
            plt.show()

    def test_bem_engine_tensile_only_triangle(self):
        self._run_bem_engine(get_triangle_tensile_only_setup)

    def test_bem_engine_tensile_only_dike(self):
        self._run_bem_engine(get_disk_tensile_only_setup)

    def test_bem_engine_dike(self):
        self._run_bem_engine(get_disk_setup)

    def test_bem_engine_dike_ringfault(self):
        kwargs = {"intersect": True}
        self._run_bem_engine(get_disk_ringfault_setup, **kwargs)

        kwargs = {"intersect": False}
        self._run_bem_engine(get_disk_ringfault_setup, **kwargs)

    def test_bem_source_intersection(self):
        config, sources, _ = get_disk_ringfault_setup(intersect=True)

        intersect = check_intersection(sources, mesh_size=config.mesh_size * km)
        assert intersect == True

        config, sources, _ = get_disk_ringfault_setup(intersect=False)

        intersect = check_intersection(sources, mesh_size=config.mesh_size * km)
        assert intersect == False


if __name__ == "__main__":
    util.setup_logging("test_bem", "info")
    unittest.main()
