from __future__ import annotations

import logging

import numpy as num

from pyrocko.moment_tensor import symmat6
from pyrocko.gf import StaticResult, Response

from .sources import DiscretizedBEMSource, slip_comp_to_idx

try:
    from cutde import halfspace as HS
    from cutde.geometry import strain_to_stress

except ImportError:
    raise ImportError("'Cutde' needs to be installed!")


logger = logging.getLogger("bem")


class BEMResponse(object):
    def __init__(
        self, displacements: num.ndarray, ordering: list, n_sources: int
    ) -> None:
        displacements = displacements
        ordering = ordering
        n_sources = n_sources

    def static_results(self):
        results = []
        for src_idx in range(self.n_sources):
            start_idx = self.ordering[src_idx]
            end_idx = self.ordering[src_idx + 1]
            result = {
                "displacement.n": self.displacements[start_idx:end_idx, 1],
                "displacement.e": self.displacements[start_idx:end_idx, 0],
                "displacement.d": self.displacements[start_idx:end_idx, 2],
            }
            results.append(StaticResult(result))


class BEMEngine(object):
    def __init__(self, config) -> None:
        self.config = config
        self._obs_points = None

    def cache_target_coords3(self, targets, dtype="float32"):
        if self._obs_points is not None:
            coords5 = num.vstack([target.coords5() for target in targets])
            obs_pts = num.zeros((coords5.shape[0], 3))
            obs_pts[:, 0] = coords5[:, 3]
            obs_pts[:, 1] = coords5[:, 2]
            self._obs_points = obs_pts.astype(dtype)

        return self._obs_points

    def process(self, sources: list, targets: list) -> num.ndarray:
        discretized_sources = [
            source.discretize_basesource(mesh_size=self.config.mesh_size)
            for source in sources
        ]
        obs_points = self.cache_target_coords3(targets, dtype="float32")

        coefficient_matrix = self.get_interaction_matrix(self, discretized_sources)
        tractions = self.config.boundary_conditions.get_traction_field(
            discretized_sources
        )

        inv_slips, _, _, _ = num.linalg.lstsq(coefficient_matrix, tractions, rcond=None)

        result_sources = [
            discretized_sources[src_idx]
            for bcond in self.config.boundary_conditions
            for src_idx in bcond.source_idxs
        ]

        all_triangles = num.vstack([source.triangles_xyz for source in result_sources])
        disp_mat = HS.disp_matrix(
            obs_pts=obs_points, tris=all_triangles, nu=self.config.nu
        )

        n_all_triangles = all_triangles.shape[0]
        slips = num.ones((n_all_triangles, 3))

        start_idx = 0
        n_triangles_result_sources = [0]
        for bcond in self.config.boundary_conditions:
            for source_idx in bcond.source_idxs:
                source_mesh = discretized_sources[source_idx]
                end_idx = start_idx + source_mesh.n_triangles - 1

                slips[
                    start_idx:end_idx,
                    slip_comp_to_idx[bcond.slip_component],
                ] = inv_slips[start_idx:end_idx]
                start_idx += source_mesh.n_triangles
                n_triangles_result_sources.append(source_mesh.n_triangles)

        displacements = disp_mat.reshape((-1, n_all_triangles * 3)).dot(slips.flatten())
        return Response(
            displacements=displacements,
            source_ordering=n_triangles_result_sources,
            n_sources=len(result_sources),
        )

    def get_interaction_matrix(self, discretized_sources: list) -> num.ndarray:
        G_slip_components = [[], [], []]
        for bcond in self.config.boundary_conditions:
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

    strain_mat_T = num.transpose(strain_mat, (0, 2, 3, 1))
    stress_mat_T = strain_to_stress(strain_mat_T, mu=mu, nu=nu)
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
