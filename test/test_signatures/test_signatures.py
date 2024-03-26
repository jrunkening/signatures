import unittest
import trimesh
import torch

import prototype.signatures as signatures_proto

from signatures import signatures

from test_signatures import FILE_PATH_MESH


class SignaturesTestCase(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.load(FILE_PATH_MESH)

    def assert_close(self, tensor_sparse, scipy_sparse):
        actual = tensor_sparse.to_dense()
        expected = torch.tensor(scipy_sparse.todense()).to(actual.dtype)

        torch.testing.assert_close(actual, expected)

    def test_hks(self):
        signatures.hks(self.mesh, 10, 5)
