import logging
import numpy as num

from pyrocko.guts import Float, Tuple
from pyrocko.orthodrome import ne_to_latlon
from pyrocko.gf.seismosizer import Source

from dataclasses import dataclass

try:
    import pygmsh

    gmsh = pygmsh.helpers.gmsh

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
    "source_catalog",
]


def get_node_name(ellipse_prefix, node_suffix):
    if ellipse_prefix:
        return f"{ellipse_prefix}_{node_suffix}_node"
    else:
        return f"{node_suffix}_node"


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
        return self._points

    @property
    def triangles_idxs(self):
        return self._mesh.cells_dict["triangle"]

    @property
    def triangles_xyz(self):
        return self.vertices[self.triangles_idxs]

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
class Origin:
    """Class for storing coordinates of a node in a mesh."""

    x: float
    y: float
    z: float


class BEMSource(Source):
    strike_traction = Float.T(
        default=0.0, help="Traction in strike-direction of the Triangles"
    )
    dip_traction = Float.T(
        default=0.0, help="Traction in dip-direction of the Triangles"
    )
    tensile_traction = Float.T(
        default=0.0, help="Traction in normal-direction of the Triangles"
    )


class TriangleBEMSource(BEMSource):

    p1 = Tuple.T(3, Float.T(), default=(0, 1, -1))
    p2 = Tuple.T(3, Float.T(), default=(1, 0, -1))
    p3 = Tuple.T(3, Float.T(), default=(-1, 0, -1))

    def discretize_basesource(self, mesh_size, plot=False, optimize=False):
        with pygmsh.geo.Geometry() as geom:

            gp1 = geom.add_point(self.p1, mesh_size=mesh_size)
            gp2 = geom.add_point(self.p2, mesh_size=mesh_size)
            gp3 = geom.add_point(self.p3, mesh_size=mesh_size)

            l1 = geom.add_line(gp1, gp2)
            l2 = geom.add_line(gp2, gp3)
            l3 = geom.add_line(gp3, gp1)

            edge = geom.add_curve_loop([l1, l2, l3])
            geom.add_surface(edge)

            mesh = geom.generate_mesh()

            if plot:
                gmsh.fltk.run()

            return DiscretizedBEMSource(
                mesh=mesh,
                mesh_size=mesh_size,
                dtype="float32",
                tractions=(
                    self.strike_traction,
                    self.dip_traction,
                    self.tensile_traction,
                ),
            )


class EllipseBEMSource(BEMSource):
    major_axis = Float.T(default=0.5 * km)
    minor_axis = Float.T(default=0.3 * km)

    strike = Float.T(default=0.0)

    def __init__(self, **kwargs):
        BEMSource.__init__(self, **kwargs)
        self.points = {}

    @property
    def _origin(self):
        return Origin(x=self.east_shift, y=self.north_shift, z=-self.depth)

    @property
    def origin_node(self):
        return (
            self._origin.x,
            self._origin.y,
            self._origin.z,
        )

    @property
    def left_major_node(self):
        return (
            self._origin.x,
            self._origin.y - self.major_axis,
            self._origin.z,
        )

    @property
    def right_major_node(self):
        return (
            self._origin.x,
            self._origin.y + self.major_axis,
            self._origin.z,
        )

    @property
    def upper_minor_node(self):
        return (
            self._origin.x + self.minor_axis,
            self._origin.y,
            self._origin.z,
        )

    @property
    def lower_minor_node(self):
        return (
            self._origin.x - self.minor_axis,
            self._origin.y,
            self._origin.z,
        )

    def _get_arch_points(self, node_names):
        try:
            return [self.points[node_name] for node_name in node_names]
        except KeyError:
            raise ValueError("Points are not fully initialized in geometry!")

    def get_top_upper_left_arch_points(self):
        return self._get_arch_points(
            [
                "left_major_node",
                "origin_node",
                "upper_minor_node",
                "upper_minor_node",
            ]
        )

    def get_top_upper_right_arch_points(self):
        return self._get_arch_points(
            [
                "upper_minor_node",
                "origin_node",
                "right_major_node",
                "right_major_node",
            ]
        )

    def get_top_lower_right_arch_points(self):
        return self._get_arch_points(
            [
                "right_major_node",
                "origin_node",
                "lower_minor_node",
                "lower_minor_node",
            ]
        )

    def get_top_lower_left_arch_points(self):
        return self._get_arch_points(
            [
                "lower_minor_node",
                "origin_node",
                "left_major_node",
                "left_major_node",
            ]
        )

    def _init_points_geometry(self, geom=None, ellipse_prefixes=(""), mesh_size=1.0):
        for ellipse_prefix in ellipse_prefixes:
            for node_suffix in (
                "left_major",
                "right_major",
                "upper_minor",
                "lower_minor",
                "origin",
            ):
                node_name = get_node_name(ellipse_prefix, node_suffix)
                node = getattr(self, node_name)

                if geom is not None:
                    self.points[node_name] = geom.add_point(node, mesh_size=mesh_size)
                else:
                    raise ValueError("Geometry needs to be initialized first!")


class DiskBEMSource(EllipseBEMSource):
    plunge = Float.T(default=0.0)
    dip = Float.T(default=0.0)

    def __init__(self, **kwargs):
        EllipseBEMSource.__init__(self, **kwargs)

    def outline(self, cs="xy", npoints=50):
        return get_ellipse_points(
            self.lon,
            self.lat,
            self.east_shift,
            self.north_shift,
            self.major_axis,
            self.minor_axis,
            self.dip,
            self.plunge,
            self.strike,
            cs=cs,
            npoints=npoints,
        )

    def discretize_basesource(self, mesh_size, target=None, optimize=False, plot=False):

        with pygmsh.geo.Geometry() as geom:
            self._init_points_geometry(
                geom, ellipse_prefixes=("",), mesh_size=mesh_size
            )

            rotations = (self.dip, self.plunge, self.strike)
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

            disk = geom.add_surface(ellipse)

            if optimize:
                geom.synchronize()

                gmsh.model.mesh.generate()
                gmsh.model.mesh.optimize()
                mesh = pygmsh.helpers.extract_to_meshio()
            else:
                mesh = geom.generate_mesh()

            if plot:
                gmsh.fltk.run()

            return DiscretizedBEMSource(
                mesh=mesh,
                mesh_size=mesh_size,
                dtype="float32",
                tractions=(
                    self.strike_traction,
                    self.dip_traction,
                    self.tensile_traction,
                ),
            )


class RingfaultBEMSource(EllipseBEMSource):
    delta_east_shift_bottom = Float.T(default=0.0 * km)
    delta_north_shift_bottom = Float.T(default=0.0 * km)
    delta_depth_bottom = Float.T(default=1.0 * km)

    major_axis = Float.T(default=0.5 * km)
    minor_axis = Float.T(default=0.3 * km)
    major_axis_bottom = Float.T(default=0.55 * km)
    minor_axis_bottom = Float.T(default=0.35 * km)

    def __init__(self, **kwargs):
        EllipseBEMSource.__init__(self, **kwargs)

    @property
    def _bottom_origin(self):
        return Origin(
            x=self._origin.x + self.delta_east_shift_bottom,
            y=self._origin.y + self.delta_north_shift_bottom,
            z=self._origin.z - self.delta_depth_bottom,
        )

    @property
    def bottom_origin_node(self):
        return (
            self._bottom_origin.x,
            self._bottom_origin.y,
            self._bottom_origin.z,
        )

    @property
    def bottom_left_major_node(self):
        return (
            self._bottom_origin.x,
            self._bottom_origin.y - self.major_axis_bottom,
            self._bottom_origin.z,
        )

    @property
    def bottom_right_major_node(self):
        return (
            self._bottom_origin.x,
            self._bottom_origin.y + self.major_axis_bottom,
            self._bottom_origin.z,
        )

    @property
    def bottom_upper_minor_node(self):
        return (
            self._bottom_origin.x + self.minor_axis_bottom,
            self._bottom_origin.y,
            self._bottom_origin.z,
        )

    @property
    def bottom_lower_minor_node(self):
        return (
            self._bottom_origin.x - self.minor_axis_bottom,
            self._bottom_origin.y,
            self._bottom_origin.z,
        )

    def get_bottom_upper_left_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_left_major_node",
                "bottom_origin_node",
                "bottom_upper_minor_node",
                "bottom_upper_minor_node",
            ]
        )

    def get_bottom_upper_right_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_upper_minor_node",
                "bottom_origin_node",
                "bottom_right_major_node",
                "bottom_right_major_node",
            ]
        )

    def get_bottom_lower_right_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_right_major_node",
                "bottom_origin_node",
                "bottom_lower_minor_node",
                "bottom_lower_minor_node",
            ]
        )

    def get_bottom_lower_left_arch_points(self):
        return self._get_arch_points(
            [
                "bottom_lower_minor_node",
                "bottom_origin_node",
                "bottom_left_major_node",
                "bottom_left_major_node",
            ]
        )

    def get_left_major_connecting_points(self):
        return self._get_arch_points(["left_major_node", "bottom_left_major_node"])

    def get_right_major_connecting_points(self):
        return self._get_arch_points(["right_major_node", "bottom_right_major_node"])

    def get_upper_minor_connecting_points(self):
        return self._get_arch_points(["upper_minor_node", "bottom_upper_minor_node"])

    def get_lower_minor_connecting_points(self):
        return self._get_arch_points(["lower_minor_node", "bottom_lower_minor_node"])

    def discretize_basesource(self, mesh_size, target=None, optimize=False, plot=False):
        with pygmsh.geo.Geometry() as geom:
            self._init_points_geometry(
                geom, ellipse_prefixes=("", "bottom"), mesh_size=mesh_size
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

            c_lmaj = geom.add_line(*self.get_left_major_connecting_points())
            c_rmaj = geom.add_line(*self.get_right_major_connecting_points())
            c_umin = geom.add_line(*self.get_upper_minor_connecting_points())
            c_lmin = geom.add_line(*self.get_lower_minor_connecting_points())

            m_top_left = geom.add_curve_loop([t_arch_ul, c_umin, -b_arch_ul, -c_lmaj])
            m_top_right = geom.add_curve_loop([t_arch_ur, c_rmaj, -b_arch_ur, -c_umin])
            m_bottom_right = geom.add_curve_loop(
                [t_arch_lr, c_lmin, -b_arch_lr, -c_rmaj]
            )
            m_bottom_left = geom.add_curve_loop(
                [t_arch_ll, c_lmaj, -b_arch_ll, -c_lmin]
            )
            mantle = [
                geom.add_surface(quadrant)
                for quadrant in (m_top_left, m_top_right, m_bottom_right, m_bottom_left)
            ]

            geom.add_surface_loop(mantle)

            if optimize:
                geom.synchronize()

                # set compound entities after sync
                for c in geom._COMPOUND_ENTITIES:
                    gmsh.model.mesh.setCompound(*c)

                gmsh.model.mesh.generate()
                gmsh.model.mesh.optimize()
                mesh = pygmsh.helpers.extract_to_meshio()
            else:
                mesh = geom.generate_mesh()

            if plot:
                gmsh.fltk.run()

            return DiscretizedBEMSource(
                mesh=mesh,
                mesh_size=mesh_size,
                dtype="float32",
                tractions=(
                    self.strike_traction,
                    self.dip_traction,
                    self.tensile_traction,
                ),
            )

    def outline(self, cs="xy", npoints=50):
        upper_ellipse = get_ellipse_points(
            self.lon,
            self.lat,
            self.east_shift,
            self.north_shift,
            self.major_axis,
            self.minor_axis,
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
            self.major_axis_bottom,
            self.minor_axis_bottom,
            0.0,
            0.0,
            self.strike,
            cs=cs,
            npoints=npoints,
        )
        return num.vstack([upper_ellipse, lower_ellipse])


def get_ellipse_points(
    lon,
    lat,
    east_shift,
    north_shift,
    major_axis,
    minor_axis,
    dip,
    plunge,
    strike,
    cs="xy",
    npoints=50,
):
    major_axis_rot = major_axis * num.cos(dip * DEG2RAD)
    minor_axis_rot = minor_axis * num.cos(plunge * DEG2RAD)

    ring = num.linspace(0, 2 * num.pi, npoints)
    ellipse = num.array(
        [minor_axis_rot * num.cos(ring), major_axis_rot * num.sin(ring)]
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
        raise NotImplemented(f"Coordinate system '{cs}' is not implemented.")


source_names = """
    DiskBEMSource
    RingfaultBEMSource
    """.split()

source_classes = [DiskBEMSource, RingfaultBEMSource]

source_catalog = dict(zip(source_names, source_classes))
