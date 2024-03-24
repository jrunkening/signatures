import unittest
import trimesh
import torch
import numpy

import prototype.laplace as laplace_proto

from signatures import laplace

from test_signatures import FILE_PATH_MESH


class LaplaceTestCase(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.load(FILE_PATH_MESH)

    def assert_close(self, tensor_sparse, scipy_sparse):
        actual = tensor_sparse.to_dense()
        expected = torch.tensor(scipy_sparse.todense()).to(actual.dtype)

        torch.testing.assert_close(actual, expected)

    def test_mass_matrix(self):
        self.assert_close(
            laplace.mass_matrix(self.mesh),
            laplace_proto.build_mass_matrix(self.mesh)
        )

    def test_laplacian_beltrami(self):
        self.assert_close(
            laplace.laplacian_beltrami(self.mesh),
            laplace_proto.build_laplace_beltrami_matrix(self.mesh)
        )

    def test_cotan(self):
        n = 5

        vector0 = torch.rand(n, 3)
        vector1 = torch.rand(n, 3)

        actual = laplace.cotan(vector0, vector1)
        expected = torch.tensor(
            numpy.sum(vector0.numpy()*vector1.numpy()) / numpy.linalg.norm(numpy.cross(vector0.numpy(), vector1.numpy()), axis=-1)
        )
        torch.testing.assert_close(actual, expected)

    def test_laplacian_beltrami(self):
        self.assert_close(
            laplace.laplacian_cotangens(self.mesh),
            laplace_proto.build_cotangens_matrix(self.mesh)
        )
