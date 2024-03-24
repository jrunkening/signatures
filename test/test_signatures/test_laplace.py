import trimesh
import torch

import prototype.laplace as laplace_proto

from signatures import laplace

from test_signatures import FILE_PATH_MESH


def test_mass_matrix():
    mesh = trimesh.load(FILE_PATH_MESH)

    mass_matrix_actual = laplace.mass_matrix(mesh)
    mass_matrix_expected = torch.tensor(laplace_proto.build_mass_matrix(mesh).todense()).to(mass_matrix_actual.dtype)

    torch.testing.assert_close(mass_matrix_actual, mass_matrix_expected)
