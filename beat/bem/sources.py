from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from time import time

import numpy as num
from pyrocko.gf.seismosizer import Source, outline_rect_source
from pyrocko.guts import Float, Tuple
from pyrocko.orthodrome import ne_to_latlon

try:
    import pygmsh

    gmsh = pygmsh.helpers.gmsh

    nthreads = os.environ.get("NUM_THREADS", "1")

except ImportError:
    raise ImportError("'Pygmsh' needs to be installed!")

try:
    from cutde.geometry import compute_efcs_to_tdcs_rotations

except ImportError:
    raise ImportError("'Cutde' needs to be installed!")


logger = logging.getLogger("bem.sources")

origintypes = {"east_shift", "north_shift", "depth"}


DEG2RAD = num.pi / 180.0
km = 1.0e3


slip_comp_to_idx = {
    "strike": 0,
    "dip": 1,
    "normal": 2,
}


__all__ = [
    "DiscretizedBEMSource",
    "RingfaultBEMSource",
    "DiskBEMSource",
    "TriangleBEMSource",
    "RectangularBEMSource",
    "CurvedBEMSource",
    "source_catalog",
    "check_intersection",
]


def get_node_name(prefix: str, suffix):
    if prefix:
        return f"{prefix}_{suffix}_node"
    else:
        return f"{suffix}_node"


class DiscretizedBEMSource(object):
    def __init__(self, mesh, dtype=None, tractions=(0, 0, 0), mesh_size=1.0):
        self._points = mesh.points.astype(dtype) if dtype is not None else mesh.points
        self._mesh = mesh
        self._centroids = None
        self._tdcs = None
        self._e_strike = None
        self._e_dip = None
        self._e_normal = None
        self.mesh_size = mesh_size

        self.tractions = tractions

    def __repr__(self):
        return self._mesh.__repr__()

    def get_traction_vector(self, slip_component):
        return (
            num.ones((self.n_triangles))
            * self.tractions[slip_comp_to_idx[slip_component]]
        )

    @property
    def vertices(self):
        """Coordinates of vertices in [m] (n_vertices, 3)"""
        return self._points

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def triangles_idxs(self):
        return self._mesh.cells_dict["triangle"]

    @property
    def triangles_xyz(self):
        """
        Returns:
            :class:`numpy.ndarray` (n_triangles, n_points [3], n_dimensions [3])
        """
        return self.vertices[self.triangles_idxs]

    @property
    def p1_xyz(self):
        """
        Coordinates xyz [m] of all points p1

        Returns:
            :class:`numpy.ndarray` [n_triangles, 3]
        """
        return self.triangles_xyz[:, 0]

    @property
    def p2_xyz(self):
        """
        Coordinates xyz [m] of all points p2

        Returns:
            :class:`numpy.ndarray` [n_triangles, 3]
        """
        return self.triangles_xyz[:, 1]

    @property
    def p3_xyz(self):
        """
        Coordinates xyz [m] of all points p3

        Returns:
            :class:`numpy.ndarray` [n_triangles, 3]
        """
        return self.triangles_xyz[:, 2]

    @property
    def vector_p1p2(self):
        return self.p2_xyz - self.p1_xyz

    @property
    def vector_p1p3(self):
        return self.p3_xyz - self.p1_xyz

    def get_areas_triangles(self):
        """
        Area of triangles [$m^2$]

        Returns:
            :class:`numpy.ndarray` [n_triangles]
        """
        return (
            num.linalg.norm(num.cross(self.vector_p1p2, self.vector_p1p3), axis=1) / 2
        )

    def get_minmax_triangles_xyz(self):
        mins = self.triangles_xyz.min(0)
        maxs = self.triangles_xyz.max(0)
        return num.vstack([mins, maxs]).T

    @property
    def n_triangles(self):
        return self.triangles_xyz.shape[0]

    @property
    def centroids(self):
        if self._centroids is None:
            self._centroids = num.mean(self.triangles_xyz, axis=1)

        return self._centroids

    @property
    def vectors_tdcs(self):
        """
        Unit vectors in Triangular Dislocation Coordinate System
        """
        if self._tdcs is None:
            self._tdcs = compute_efcs_to_tdcs_rotations(self.triangles_xyz)

        return self._tdcs

    @property
    def unit_strike_vectors(self):
        if self._e_strike is None:
            strike_vec = self.vectors_tdcs[:, 0, :]
            strike_vec /= num.linalg.norm(strike_vec, axis=1)[:, None]
            self._e_strike = strike_vec
        return self._e_strike

    @property
    def unit_dip_vectors(self):
        if self._e_dip is None:
            dip_vec = self.vectors_tdcs[:, 1, :]
            dip_vec /= num.linalg.norm(dip_vec, axis=1)[:, None]
            self._e_dip = dip_vec
        return self._e_dip

    @property
    def unit_normal_vectors(self):
        if self._e_normal is None:
            normal_vec = self.vectors_tdcs[:, 2, :]
            normal_vec /= num.linalg.norm(normal_vec, axis=1)[:, None]
            self._e_normal = normal_vec
        return self._e_normal


@dataclass
class Node:
    """Class for storing coordinates of a node in a mesh."""

    x: float
    y: float
    z: float


class BEMSource(Source):
    def _init_points_geometry(
        self, geom=None, prefixes=(""), suffixes=(""), mesh_size=1.0
    ):
        for prefix in prefixes:
            for suffix in suffixes:
                node_name = get_node_name(prefix, suffix)
                node = getattr(self, node_name)

                if geom is not None:
                    self.points[node_name] = geom.add_point(node, mesh_size=mesh_size)
                else:
                    raise ValueError("Geometry needs to be initialized first!")

    def get_tractions(self):
        raise NotImplementedError("Implement in inherited class")

    def _get_arch_points(self, node_names: list[str]) -> list:
        try:
            return [self.points[node_name] for node_name in node_names]
        except KeyError:
            raise ValueError("Points are not fully initialized in geometry!")

    def get_source_surface(self, geom, mesh_size):
        raise NotImplementedError

    def discretize_basesource(self, mesh_size, target=None, plot=False):
        with pygmsh.geo.Geometry() as geom:
            gmsh.option.setNumber("General.NumThreads", int(nthreads))

            surf = self.get_source_surface(geom, mesh_size)
            if len(surf) > 1:
                geom.add_surface_loop(surf)

            mesh = geom.generate_mesh()

            if plot:
                gmsh.fltk.run()

        return DiscretizedBEMSource(
            mesh=mesh,
            mesh_size=mesh_size,
            dtype="float32",
            tractions=self.get_tractions(),
        )


class TriangleBEMSource(BEMSource):
    strike_traction = Float.T(
        default=0.0, help="Traction [Pa] in strike-direction of the Triangles"
    )
    dip_traction = Float.T(
        default=0.0, help="Traction [Pa] in dip-direction of the Triangles"
    )
    normal_traction = Float.T(
        default=0.0, help="Traction [Pa] in normal-direction of the Triangles"
    )

    p1 = Tuple.T(3, Float.T(), default=(0, 1, -1))
    p2 = Tuple.T(3, Float.T(), default=(1, 0, -1))
    p3 = Tuple.T(3, Float.T(), default=(-1, 0, -1))

    def get_source_surface(self, geom, mesh_size):
        gp1 = geom.add_point(self.p1, mesh_size=mesh_size)
        gp2 = geom.add_point(self.p2, mesh_size=mesh_size)
        gp3 = geom.add_point(self.p3, mesh_size=mesh_size)

        l1 = geom.add_line(gp1, gp2)
        l2 = geom.add_line(gp2, gp3)
        l3 = geom.add_line(gp3, gp1)

        edge = geom.add_curve_loop([l1, l2, l3])
        return [geom.add_surface(edge)]

    def get_tractions(self):
        return (
            self.strike_traction,  # coordinate transform ENU - NED
            self.dip_traction,
            self.normal_traction,
        )


class EllipseBEMSource(BEMSource):
    a_half_axis = Float.T(default=0.5 * km)
    b_half_axis = Float.T(default=0.3 * km)

    strike = Float.T(default=0.0)

    normal_traction = Float.T(
        default=0.0, help="Traction [Pa] in normal-direction of the Triangles"
    )

    def __init__(self, **kwargs):
        BEMSource.__init__(self, **kwargs)
        self.points = {}

    def get_tractions(self):
        raise NotImplementedError("Needs implementation in inheriting class")

    @property
    def _origin(self):
        return Node(x=self.east_shift, y=self.north_shift, z=-self.depth)

    @property
    def origin_node(self):
        return (
            self._origin.x,
            self._origin.y,
            self._origin.z,
        )

    @property
    def left_a_node(self):
        return (
            self._origin.x,
            self._origin.y - self.a_half_axis,
            self._origin.z,
        )

    @property
    def right_a_node(self):
        return (
            self._origin.x,
            self._origin.y + self.a_half_axis,
            self._origin.z,
        )

    @property
    def upper_b_node(self):
        return (
            self._origin.x + self.b_half_axis,
            self._origin.y,
            self._origin.z,
        )

    @property
    def lower_b_node(self):
        return (
            self._origin.x - self.b_half_axis,
            self._origin.y,
            self._origin.z,
        )

    def _get_node_suffixes(self):
        return (
            "left_a",
            "right_a",
            "upper_b",
            "lower_b",
            "origin",
        )

    def get_top_upper_left_arch_points(self):
        return self._get_arch_points(
            [
                "left_a_node",
                "origin_node",
                "upper_b_node",
                "upper_b_node",
            ]
        )

    def get_top_upper_right_arch_points(self):
        return self._get_arch_points(
            [
                "upper_b_node",
                "origin_node",
                "right_a_node",
                "right_a_node",
            ]
        )

    def get_top_lower_right_arch_points(self):
        return self._get_arch_points(
            [
                "right_a_node",
                "origin_node",
                "lower_b_node",
                "lower_b_node",
            ]
        )

    def get_top_lower_left_arch_points(self):
        return self._get_arch_points(
            [
                "lower_b_node",
                "origin_node",
                "left_a_node",
                "left_a_node",
            ]
        )


class DiskBEMSource(EllipseBEMSource):
    plunge = Float.T(default=0.0)
    dip = Float.T(default=0.0)
    rake = Float.T(default=0.0, help="Rake-angle [deg] towards the North.")
    traction = Float.T(default=0.0, help="Traction [Pa] in rake direction.")

    def get_tractions(self):
        strike_traction = -num.cos(self.rake * DEG2RAD) * self.traction
        dip_traction = -num.sin(self.rake * DEG2RAD) * self.traction
        return (
            strike_traction,
            dip_traction,
            self.normal_traction,
        )

    def __init__(self, **kwargs):
        EllipseBEMSource.__init__(self, **kwargs)

    def outline(self, cs="xy", npoints=50):
        return num.flipud(
            get_ellipse_points(
                self.lon,
                self.lat,
                self.east_shift,
                self.north_shift,
                self.a_half_axis,
                self.b_half_axis,
                self.dip,
                self.plunge,
                self.strike,
                cs=cs,
                npoints=npoints,
            )
        )

    def get_source_surface(self, geom, mesh_size):
        self._init_points_geometry(
            geom,
            prefixes=("",),
            suffixes=self._get_node_suffixes(),
            mesh_size=mesh_size,
        )

        rotations = (-self.plunge, -self.dip, self.strike)
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for point in self.points.values():
            for rot_angle, axis in zip(rotations, axes):
                if rot_angle != 0:
                    # TODO if rotation results in one point ending at the exact
                    # same location of other point it will be removed
                    geom.rotate(
                        point,
                        self.origin_node,
                        -rot_angle * DEG2RAD,
                        axis,
                    )

        t_arch_ul = geom.add_ellipse_arc(*self.get_top_upper_left_arch_points())
        t_arch_ur = geom.add_ellipse_arc(*self.get_top_upper_right_arch_points())
        t_arch_lr = geom.add_ellipse_arc(*self.get_top_lower_right_arch_points())
        t_arch_ll = geom.add_ellipse_arc(*self.get_top_lower_left_arch_points())

        ellipse = geom.add_curve_loop([t_arch_ul, t_arch_ur, t_arch_lr, t_arch_ll])

        return [geom.add_surface(ellipse)]


class RingfaultBEMSource(EllipseBEMSource):
    delta_east_shift_bottom = Float.T(default=0.0 * km)
    delta_north_shift_bottom = Float.T(default=0.0 * km)
    depth_bottom = Float.T(default=1.0 * km)

    a_half_axis = Float.T(default=0.5 * km)
    b_half_axis = Float.T(default=0.3 * km)
    a_half_axis_bottom = Float.T(default=0.55 * km)
    b_half_axis_bottom = Float.T(default=0.35 * km)

    strike_traction = Float.T(
        default=0.0, help="Traction [Pa] in strike-direction of the Triangles"
    )
    dip_traction = Float.T(
        default=0.0, help="Traction [Pa] in dip-direction of the Triangles"
    )

    def __init__(self, **kwargs):
        EllipseBEMSource.__init__(self, **kwargs)

    def get_tractions(self):
        return (
            -self.strike_traction,  # coordinate transform ENU - NED
            self.dip_traction,
            self.normal_traction,
        )

    @property
    def _bottom_origin(self):
        return Node(
            x=self._origin.x + self.delta_east_shift_bottom,
            y=self._origin.y + self.delta_north_shift_bottom,
            z=-self.depth_bottom,
        )

    @property
    def bottom_origin_node(self):
        return (
            self._bottom_origin.x,
            self._bottom_origin.y,
            self._bottom_origin.z,
        )

    @property
    def bottom_left_a_node(self):
        return (
            self._bottom_origin.x,
            self._bottom_origin.y - self.a_half_axis_bottom,
            self._bottom_origin.z,
        )

    @property
    def bottom_right_a_node(self):
        return (
            self._bottom_origin.x,
            self._bottom_origin.y + self.a_half_axis_bottom,
            self._bottom_origin.z,
        )

    @property
    def bottom_upper_b_node(self):
        return (
            self._bottom_origin.x + self.b_half_axis_bottom,
            self._bottom_origin.y,
            self._bottom_origin.z,
        )

    @property
    def bottom_lower_b_node(self):
        return (
            self._bottom_origin.x - self.b_half_axis_bottom,
            self._bottom_origin.y,
            self._bottom_origin.z,
        )

    def get_bottom_upper_left_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_left_a_node",
                "bottom_origin_node",
                "bottom_upper_b_node",
                "bottom_upper_b_node",
            ]
        )

    def get_bottom_upper_right_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_upper_b_node",
                "bottom_origin_node",
                "bottom_right_a_node",
                "bottom_right_a_node",
            ]
        )

    def get_bottom_lower_right_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_right_a_node",
                "bottom_origin_node",
                "bottom_lower_b_node",
                "bottom_lower_b_node",
            ]
        )

    def get_bottom_lower_left_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_lower_b_node",
                "bottom_origin_node",
                "bottom_left_a_node",
                "bottom_left_a_node",
            ]
        )

    def get_left_a_connecting_points(self):
        return self._get_arch_points(["left_a_node", "bottom_left_a_node"])

    def get_right_a_connecting_points(self):
        return self._get_arch_points(["right_a_node", "bottom_right_a_node"])

    def get_upper_b_connecting_points(self):
        return self._get_arch_points(["upper_b_node", "bottom_upper_b_node"])

    def get_lower_b_connecting_points(self):
        return self._get_arch_points(["lower_b_node", "bottom_lower_b_node"])

    def get_source_surface(self, geom, mesh_size):
        self._init_points_geometry(
            geom,
            prefixes=("", "bottom"),
            suffixes=self._get_node_suffixes(),
            mesh_size=mesh_size,
        )

        if self.strike != 0:
            for point in self.points.values():
                geom.rotate(
                    point,
                    (self.east_shift, self.north_shift, -self.depth),
                    -self.strike * DEG2RAD,
                    (0.0, 0.0, 1.0),
                )

        t_arch_ul = geom.add_ellipse_arc(*self.get_top_upper_left_arch_points())
        t_arch_ur = geom.add_ellipse_arc(*self.get_top_upper_right_arch_points())
        t_arch_lr = geom.add_ellipse_arc(*self.get_top_lower_right_arch_points())
        t_arch_ll = geom.add_ellipse_arc(*self.get_top_lower_left_arch_points())

        b_arch_ul = geom.add_ellipse_arc(*self.get_bottom_upper_left_arch_points())
        b_arch_ur = geom.add_ellipse_arc(*self.get_bottom_upper_right_arch_points())
        b_arch_lr = geom.add_ellipse_arc(*self.get_bottom_lower_right_arch_points())
        b_arch_ll = geom.add_ellipse_arc(*self.get_bottom_lower_left_arch_points())

        c_lmaj = geom.add_line(*self.get_left_a_connecting_points())
        c_rmaj = geom.add_line(*self.get_right_a_connecting_points())
        c_umin = geom.add_line(*self.get_upper_b_connecting_points())
        c_lmin = geom.add_line(*self.get_lower_b_connecting_points())

        m_top_left = geom.add_curve_loop([t_arch_ul, c_umin, -b_arch_ul, -c_lmaj])
        m_top_right = geom.add_curve_loop([t_arch_ur, c_rmaj, -b_arch_ur, -c_umin])
        m_bottom_right = geom.add_curve_loop([t_arch_lr, c_lmin, -b_arch_lr, -c_rmaj])
        m_bottom_left = geom.add_curve_loop([t_arch_ll, c_lmaj, -b_arch_ll, -c_lmin])
        mantle = [
            geom.add_surface(quadrant)
            for quadrant in (m_top_left, m_top_right, m_bottom_right, m_bottom_left)
        ]
        return mantle
        # return geom.add_surface_loop(mantle)

    def outline(self, cs="xy", npoints=50):
        upper_ellipse = get_ellipse_points(
            self.lon,
            self.lat,
            self.east_shift,
            self.north_shift,
            self.a_half_axis,
            self.b_half_axis,
            0.0,
            0.0,
            self.strike,
            cs=cs,
            npoints=npoints,
        )
        lower_ellipse = get_ellipse_points(
            self.lon,
            self.lat,
            self.east_shift + self.delta_east_shift_bottom,
            self.north_shift + self.delta_north_shift_bottom,
            self.a_half_axis_bottom,
            self.b_half_axis_bottom,
            0.0,
            0.0,
            self.strike,
            cs=cs,
            npoints=npoints,
        )
        return num.vstack([upper_ellipse, lower_ellipse])


class RectangularBEMSource(BEMSource):
    width = Float.T(default=5 * km, help="Width [m] of the fault plane.")
    length = Float.T(default=10 * km, help="Length [m] of the fault plane.")
    dip = Float.T(default=0, help="Dip-angle [deg] towards the horizontal.")
    strike = Float.T(default=0.0, help="Strike-angle [deg] towards the North.")
    rake = Float.T(default=0.0, help="Rake-angle [deg] towards the North.")
    traction = Float.T(default=0.0, help="Traction [Pa] in rake direction.")
    normal_traction = Float.T(
        default=0.0, help="Traction [Pa] in normal-direction of the Triangles"
    )

    def __init__(self, **kwargs):
        BEMSource.__init__(self, **kwargs)
        self.points = {}

    def outline(self, cs="xyz"):
        points = outline_rect_source(
            self.strike, self.dip, self.length, self.width, anchor="top"
        )

        points[:, 0] += self.north_shift
        points[:, 1] += self.east_shift
        points[:, 2] += self.depth
        if cs == "xyz":
            return points
        elif cs == "xy":
            return points[:, :2]
        elif cs in ("latlon", "lonlat"):
            latlon = ne_to_latlon(self.lat, self.lon, points[:, 0], points[:, 1])

            latlon = num.array(latlon).T
            if cs == "latlon":
                return latlon
            else:
                return latlon[:, ::-1]

    def get_tractions(self):
        strike_traction = -num.cos(self.rake * DEG2RAD) * self.traction
        dip_traction = -num.sin(self.rake * DEG2RAD) * self.traction
        return (
            strike_traction,
            dip_traction,
            self.normal_traction,
        )

    @property
    def _origin(self):
        return Node(x=self.east_shift, y=self.north_shift, z=-self.depth)

    @property
    def origin_node(self):
        return (
            self._origin.x,
            self._origin.y,
            self._origin.z,
        )

    @property
    def _top_right(self):
        return Node(
            self._origin.x,
            self._origin.y + self.length / 2,
            self._origin.z,
        )

    @property
    def top_right_node(self):
        return (
            self._top_right.x,
            self._top_right.y,
            self._top_right.z,
        )

    @property
    def _top_left(self):
        return Node(
            self._origin.x,
            self._origin.y - self.length / 2,
            self._origin.z,
        )

    @property
    def top_left_node(self):
        return (
            self._top_left.x,
            self._top_left.y,
            self._top_left.z,
        )

    @property
    def _bottom_left(self):
        return Node(
            self._origin.x + self.width,
            self._origin.y - self.length / 2,
            self._origin.z,
        )

    @property
    def bottom_left_node(self):
        return (
            self._bottom_left.x,
            self._bottom_left.y,
            self._bottom_left.z,
        )

    @property
    def _bottom_right(self):
        return Node(
            self._origin.x + self.width,
            self._origin.y + self.length / 2,
            self._origin.z,
        )

    @property
    def bottom_right_node(self):
        return (
            self._bottom_right.x,
            self._bottom_right.y,
            self._bottom_right.z,
        )

    def _get_node_suffixes(self):
        return ("left", "right")

    def get_top_edge(self):
        return self._get_arch_points(["top_left_node", "top_right_node"])

    def get_bottom_edge(self):
        return self._get_arch_points(["bottom_left_node", "bottom_right_node"])

    def get_left_edge(self):
        return self._get_arch_points(["top_left_node", "bottom_left_node"])

    def get_right_edge(self):
        return self._get_arch_points(["top_right_node", "bottom_right_node"])

    def get_source_surface(self, geom, mesh_size):
        self._init_points_geometry(
            geom,
            prefixes=("top", "bottom"),
            suffixes=self._get_node_suffixes(),
            mesh_size=mesh_size,
        )

        top = geom.add_bezier(self.get_top_edge())
        right = geom.add_bezier(self.get_right_edge())
        bottom = geom.add_bezier(self.get_bottom_edge())
        left = geom.add_bezier(self.get_left_edge())

        rectangle = geom.add_curve_loop([-top, left, bottom, -right])
        rectangle_surface = geom.add_surface(rectangle)

        rotations = (-self.dip, self.strike)
        axes = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        for rot_angle, axis in zip(rotations, axes):
            if rot_angle != 0:
                geom.rotate(
                    rectangle_surface,
                    self.origin_node,
                    -rot_angle * DEG2RAD,
                    axis,
                )

        return [rectangle_surface]


class CurvedBEMSource(RectangularBEMSource):
    curv_location_bottom = Float.T(0.0)
    curv_amplitude_bottom = Float.T(0.0)
    bend_location = Float.T(0.0)
    bend_amplitude = Float.T(0.0)

    @property
    def bend_left_node(self):
        return (
            self._top_left.x + self.width * self.bend_location,
            self._top_left.y,
            self._top_left.z + self.width * self.bend_amplitude,
        )

    @property
    def bend_right_node(self):
        return (
            self._top_right.x + self.width * self.bend_location,
            self._top_right.y,
            self._top_right.z + self.width * self.bend_amplitude,
        )

    @property
    def curve_left_node(self):
        """Shallow edge - no curve for now"""
        return (
            self._origin.x,
            self._origin.y,
            self._origin.z,
        )

    @property
    def curve_right_node(self):
        return (
            self._bottom_left.x,
            self._bottom_left.y + self.length * self.curv_location_bottom,
            self._bottom_left.z + self.length * self.curv_amplitude_bottom,
        )

    def get_top_edge(self):
        return self._get_arch_points(
            ["top_left_node", "curve_left_node", "top_right_node"]
        )

    def get_bottom_edge(self):
        return self._get_arch_points(
            ["bottom_left_node", "curve_right_node", "bottom_right_node"]
        )

    def get_left_edge(self):
        return self._get_arch_points(
            ["top_left_node", "bend_left_node", "bottom_left_node"]
        )

    def get_right_edge(self):
        return self._get_arch_points(
            ["top_right_node", "bend_right_node", "bottom_right_node"]
        )

    def get_source_surface(self, geom, mesh_size):
        self._init_points_geometry(
            geom,
            prefixes=("top", "bottom", "curve", "bend"),
            suffixes=self._get_node_suffixes(),
            mesh_size=mesh_size,
        )

        top = geom.add_bezier(self.get_top_edge())
        right = geom.add_bezier(self.get_right_edge())
        bottom = geom.add_bezier(self.get_bottom_edge())
        left = geom.add_bezier(self.get_left_edge())

        quadrangle = geom.add_curve_loop([top, right, -bottom, -left])
        quad_surface = geom.add_surface(quadrangle)

        rotations = (-self.dip, self.strike)
        axes = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        for rot_angle, axis in zip(rotations, axes):
            if rot_angle != 0:
                geom.rotate(
                    quad_surface,
                    self.origin_node,
                    -rot_angle * DEG2RAD,
                    axis,
                )
        return [quad_surface]


def get_ellipse_points(
    lon: float,
    lat: float,
    east_shift: float,
    north_shift: float,
    a_half_axis: float,
    b_half_axis: float,
    dip: float,
    plunge: float,
    strike: float,
    cs: str = "xy",
    npoints: int = 50,
) -> num.ndarray:
    a_half_axis_rot = a_half_axis * num.cos(dip * DEG2RAD)
    b_half_axis_rot = b_half_axis * num.cos(plunge * DEG2RAD)

    ring = num.linspace(0, 2 * num.pi, npoints)
    ellipse = num.array(
        [b_half_axis_rot * num.cos(ring), a_half_axis_rot * num.sin(ring)]
    )

    strike_rad = -strike * DEG2RAD
    rot_strike = num.array(
        [
            [num.cos(strike_rad), -num.sin(strike_rad)],
            [num.sin(strike_rad), num.cos(strike_rad)],
        ]
    )
    ellipse_rot = rot_strike.dot(ellipse)

    points = num.atleast_2d(num.zeros([npoints, 2]))
    points[:, 0] += ellipse_rot[1, :] + north_shift
    points[:, 1] += ellipse_rot[0, :] + east_shift

    if cs == "xy":
        return points
    elif cs in ("latlon", "lonlat"):
        latlon = ne_to_latlon(lat, lon, points[:, 0], points[:, 1])

        latlon = num.array(latlon).T
        if cs == "latlon":
            return latlon
        else:
            return latlon[:, ::-1]
    else:
        raise NotImplementedError(f"Coordinate system '{cs}' is not implemented.")


def check_intersection(sources: list, mesh_size: float = 0.5) -> bool:
    """
    Computationally expensive check for source intersection.
    """
    n_sources = len(sources)
    if n_sources > 1:
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber("General.NumThreads", int(nthreads))
            gmsh.option.setNumber("General.Verbosity", 1)  # silence warnings

            surfaces = []
            for source in sources:
                logger.debug(source.__str__())
                surf = source.get_source_surface(geom, mesh_size)
                surfaces.append(surf)

            gmsh.model.occ.synchronize()
            before = len(gmsh.model.getEntities())
            logger.debug("Building source union ...")
            t0 = time()
            geom.boolean_union(surfaces, False, False)
            logger.debug("Time for union: %f", time() - t0)

            logger.debug("Synchronize")
            gmsh.model.occ.synchronize()
            after = len(gmsh.model.getEntities())

        if after - before:
            logger.debug("Sources intersect")
            return True

    logger.debug("Sources do not intersect")
    return False


source_names = """
    TriangleBEMSource
    DiskBEMSource
    RingfaultBEMSource
    RectangularBEMSource
    CurvedBEMSource
    """.split()

source_classes = [
    TriangleBEMSource,
    DiskBEMSource,
    RingfaultBEMSource,
    RectangularBEMSource,
    CurvedBEMSource,
]

source_catalog = dict(zip(source_names, source_classes))
