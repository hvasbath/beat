from __future__ import annotations

import logging

import numpy as num
from time import time
from pyrocko.moment_tensor import symmat6
from pyrocko.gf import StaticResult, Response, Request
from pyrocko.guts_array import Array
from pyrocko.guts import Int, Object, List

from .sources import DiscretizedBEMSource, slip_comp_to_idx, check_intersection

try:
    from cutde import halfspace as HS
    from cutde.geometry import strain_to_stress

except ImportError:
    raise ImportError("'Cutde' needs to be installed!")


logger = logging.getLogger("bem")
km = 1.0e3


class BEMResponse(Object):
    sources = List.T(default=[])
    targets = List.T(default=[])
    discretized_sources = List.T()
    displacements = Array.T(
        shape=(None,), dtype=num.float32, serialize_as="base64", optional=True
    )
    target_ordering = Array.T(shape=(None,), dtype=num.int64, optional=True)
    source_ordering = Array.T(shape=(None,), dtype=num.int64, optional=True)
    inverted_slip_vectors = Array.T(shape=(None, 3), dtype=num.float32, optional=True)

    @property
    def n_sources(self):
        return len(self.sources)

    @property
    def n_targets(self):
        return len(self.targets)

    @property
    def is_valid(self):
        if self.discretized_sources is None:
            return False
        else:
            return True

    def static_results(self) -> list[StaticResult]:
        """
        Get target specific surface displacements in NED coordinates.
        """
        results = []
        for target_idx in range(self.n_targets):
            start_idx = self.target_ordering[target_idx]
            end_idx = self.target_ordering[target_idx + 1]

            result = {
                "displacement.n": self.displacements[start_idx:end_idx, 1],
                "displacement.e": self.displacements[start_idx:end_idx, 0],
                "displacement.d": -self.displacements[start_idx:end_idx, 2],
            }
            results.append(StaticResult(result=result))

        return results

    def source_slips(self) -> list[num.ndarray]:
        """
        Get inverted slip vectors for sources

        Returns
        -------
        array_like: [n_triangles, 3]
            where columns are: strike, dip and tensile slip-components"""
        slips = []
        for src_idx in range(self.n_sources):
            if self.source_ordering is not None:
                start_idx = self.source_ordering[src_idx]
                end_idx = self.source_ordering[src_idx + 1]
                slips.append(self.inverted_slip_vectors[start_idx:end_idx, :])
            else:
                slips.append(None)
        return slips


class BEMEngine(object):
    def __init__(self, config) -> None:
        self.config = config
        self._obs_points = None
        self._ncoords_targets = None

    def cache_target_coords3(self, targets, dtype="float32"):
        ncoords_targets = num.cumsum([0] + [target.ncoords for target in targets])
        if self._ncoords_targets is None:
            self._ncoords_targets = ncoords_targets
            coords_diff = 0
        else:
            coords_diff = self._ncoords_targets.sum() - ncoords_targets.sum()

        if self._obs_points is None or coords_diff:
            coords5 = num.vstack([target.coords5 for target in targets])
            obs_pts = num.zeros((coords5.shape[0], 3))
            obs_pts[:, 0] = coords5[:, 3]
            obs_pts[:, 1] = coords5[:, 2]
            self._obs_points = obs_pts.astype(dtype)
            self._ncoords_targets = ncoords_targets

        return self._obs_points

    def clear_target_cache(self):
        self._obs_points = None
        self._ncoords_targets = None

    def process(self, sources: list, targets: list) -> num.ndarray:
        mesh_size = self.config.mesh_size * km

        if self.config.check_mesh_intersection:
            intersect = check_intersection(sources, mesh_size=mesh_size)
        else:
            intersect = False

        obs_points = self.cache_target_coords3(targets, dtype="float32")

        if intersect:
            return BEMResponse(
                sources=sources,
                targets=targets,
                discretized_sources=None,
                displacements=num.full((obs_points.shape[0], 3), -num.inf),
                target_ordering=self._ncoords_targets,
                source_ordering=None,
                inverted_slip_vectors=None,
            )

        discretized_sources = [
            source.discretize_basesource(mesh_size=mesh_size, plot=False)
            for source in sources
        ]

        coefficient_matrix = self.get_interaction_matrix(discretized_sources)
        tractions = self.config.boundary_conditions.get_traction_field(
            discretized_sources
        )

        # solve with least squares
        inv_slips = num.linalg.multi_dot(
            [
                num.linalg.inv(coefficient_matrix.T.dot(coefficient_matrix)),
                coefficient_matrix.T,
                tractions,
            ]
        )

        all_triangles = num.vstack(
            [source.triangles_xyz for source in discretized_sources]
        )
        disp_mat = HS.disp_matrix(
            obs_pts=obs_points, tris=all_triangles, nu=self.config.nu
        )

        n_all_triangles = all_triangles.shape[0]
        slips = num.zeros((n_all_triangles, 3))

        start_idx = 0
        sources_ntriangles = num.cumsum(
            [start_idx] + [source.n_triangles for source in discretized_sources]
        )
        for bcond in self.config.boundary_conditions.iter_conditions():
            for source_idx in bcond.source_idxs:
                source_mesh = discretized_sources[source_idx]
                end_idx = start_idx + source_mesh.n_triangles

                slips[
                    sources_ntriangles[source_idx] : sources_ntriangles[source_idx + 1],
                    slip_comp_to_idx[bcond.slip_component],
                ] = inv_slips[start_idx:end_idx]

                start_idx += source_mesh.n_triangles

        displacements = disp_mat.reshape((-1, n_all_triangles * 3)).dot(slips.ravel())
        return BEMResponse(
            sources=sources,
            targets=targets,
            discretized_sources=discretized_sources,
            displacements=displacements.reshape((-1, 3)),
            target_ordering=self._ncoords_targets,
            source_ordering=sources_ntriangles,
            inverted_slip_vectors=slips,
        )

    def get_interaction_matrix(self, discretized_sources: list) -> num.ndarray:
        G_slip_components = [[], [], []]
        for bcond in self.config.boundary_conditions.iter_conditions():
            for source_idx in bcond.source_idxs:
                source_mesh = discretized_sources[source_idx]

                Gs_strike = []
                Gs_dip = []
                Gs_normal = []
                for receiver_idx in bcond.receiver_idxs:
                    receiver_mesh = discretized_sources[receiver_idx]
                    g_strike, g_dip, g_normal = get_coefficient_matrices_tdcs(
                        receiver_mesh,
                        source_mesh.triangles_xyz,
                        bcond.slip_component,
                        nu=self.config.nu,
                        mu=self.config.mu,
                    )

                    Gs_strike.append(g_strike)
                    Gs_dip.append(g_dip)
                    Gs_normal.append(g_normal)

                G_slip_components[0].append(num.vstack(Gs_strike))
                G_slip_components[1].append(num.vstack(Gs_dip))
                G_slip_components[2].append(num.vstack(Gs_normal))

        return num.block(G_slip_components)

    def get_store(self, store_id):
        """Dummy method to allow compatibility"""
        return None


def get_coefficient_matrices_tdcs(
    discretized_bem_source: DiscretizedBEMSource,
    triangles_xyz: num.ndarray,
    slip_component: str,
    nu: float,
    mu: float,
) -> list[num.ndarray]:
    """
    Calculates interaction matrix between source triangles and receiver triangles.

    Parameters
    ----------
    slip_component:

    Returns
    -------
    """
    strain_mat = HS.strain_matrix(
        discretized_bem_source.centroids, triangles_xyz, nu=nu
    )

    strain_mat_T = num.transpose(strain_mat, (0, 3, 2, 1))
    stress_mat_T = strain_to_stress(strain_mat_T, mu=mu, nu=nu)

    stress_mat_T = num.transpose(stress_mat_T, (0, 2, 1, 3))
    stress_mat_m9s = symmat6(*stress_mat_T.T).T

    # select relevant source slip vector component indexs (0-strike, 1-dip, 2-tensile)
    slip_idx = slip_comp_to_idx[slip_component]

    # get traction vector from Stress tensor
    tvs = num.sum(
        stress_mat_m9s[:, :, slip_idx]
        * discretized_bem_source.unit_normal_vectors[:, None, None, :],
        axis=-1,
    )

    # get stress components from traction vector
    g_strike = num.sum(
        tvs * discretized_bem_source.unit_strike_vectors[:, None, :], axis=-1
    )
    g_dip = num.sum(tvs * discretized_bem_source.unit_dip_vectors[:, None, :], axis=-1)
    g_normal = num.sum(
        tvs * discretized_bem_source.unit_normal_vectors[:, None, :], axis=-1
    )
    return g_strike, g_dip, -g_normal  # Minus is needed due to ENU convention
