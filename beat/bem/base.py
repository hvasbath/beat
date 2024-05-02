from __future__ import annotations

import logging

import numpy as num
from matplotlib import pyplot as plt
from pyrocko.gf import StaticResult
from pyrocko.guts import List, Object
from pyrocko.guts_array import Array
from pyrocko.moment_tensor import moment_to_magnitude, symmat6

from .sources import DiscretizedBEMSource, check_intersection, slip_comp_to_idx

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
            where columns are: strike, dip and normal slip-components"""
        slips = []
        for src_idx in range(self.n_sources):
            if self.source_ordering is not None:
                start_idx = self.source_ordering[src_idx]
                end_idx = self.source_ordering[src_idx + 1]
                slips.append(self.inverted_slip_vectors[start_idx:end_idx, :])
            else:
                slips.append(None)
        return slips

    def get_derived_parameters(self, shear_modulus: float) -> list[num.ndarray]:
        """
        Calculate derived source parameters magnitude[Mw]
        and average slip amplitude[m]

        Parameters
        ----------
            shear_modulus: float

        Returns
        -------
            list of derived parameters, each entry is for each source
        """
        inverted_slips = self.source_slips()
        total_slips = [num.linalg.norm(slips, axis=1) for slips in inverted_slips]

        derived = []
        for source, slips in zip(self.discretized_sources, total_slips):
            moments = source.get_areas_triangles() * slips * shear_modulus
            derived.append(
                num.hstack([moment_to_magnitude(moments.sum()), slips.sum()])
            )

        return derived


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

    def process(self, sources: list, targets: list, debug=False) -> num.ndarray:
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
                displacements=num.full(
                    (obs_points.shape[0], 3), -99.0, dtype="float64"
                ),
                target_ordering=self._ncoords_targets,
                source_ordering=None,
                inverted_slip_vectors=None,
            )

        discretized_sources = [
            source.discretize_basesource(mesh_size=mesh_size, plot=False)
            for source in sources
        ]

        coefficient_matrix = self.get_interaction_matrix(
            discretized_sources, debug=debug
        )
        tractions = self.config.boundary_conditions.get_traction_field(
            discretized_sources
        )

        if debug:
            ax = plt.axes()
            im = ax.matshow(coefficient_matrix)
            ax.set_title("Interaction matrix")
            plt.colorbar(im)
            print("CEF shape", coefficient_matrix.shape)

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
            obs_pts=obs_points, tris=all_triangles, nu=self.config.poissons_ratio
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

    def get_interaction_matrix(
        self, discretized_sources: list, debug: bool
    ) -> num.ndarray:
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
                        nu=self.config.poissons_ratio,
                        mu=self.config.shear_modulus,
                    )

                    if debug:
                        figs, axs = plt.subplots(1, 3)
                        for k, (comp, g_comp) in enumerate(
                            zip(
                                ("strike", "dip", "normal"), (g_strike, g_dip, g_normal)
                            )
                        ):
                            axs[k].matshow(g_comp)
                            axs[k].set_title(comp)

                        plt.show()

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

    # select relevant source slip vector component indexs (0-strike, 1-dip, 2-normal)
    slip_idx = slip_comp_to_idx[slip_component]
    comp_strain_mat = strain_mat[:, :, :, slip_idx]
    comp_strain_mat_T = num.transpose(comp_strain_mat, (0, 2, 1))

    comp_stress_mat_T = strain_to_stress(
        comp_strain_mat_T.reshape((-1, 6)), mu, nu
    ).reshape(comp_strain_mat_T.shape)

    stress_mat_m9s = symmat6(*comp_stress_mat_T.T).T

    # get traction vector from Stress tensor
    tvs = num.sum(
        stress_mat_m9s * discretized_bem_source.unit_normal_vectors[:, None, None, :],
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
