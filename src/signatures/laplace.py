import torch
import trimesh


def mass_matrix(mesh: trimesh.Trimesh):
    n = len(mesh.vertices)
    areas = torch.zeros(n)
    for face, area in zip(mesh.faces, mesh.area_faces):
        areas[face] += area/3.0

    return torch.sparse.spdiags(areas, torch.zeros(1, dtype=torch.long), (n, n))


def laplacian_beltrami(mesh: trimesh.Trimesh):
    n = len(mesh.vertices)
    indices = torch.hstack([
        torch.tensor(mesh.edges).transpose(0, 1), # for every (i, j) in `mesh.edges`
        torch.arange(n).unsqueeze(0).expand(2, -1), # for every (i, j) where i==j
    ])
    values = torch.hstack([
        -torch.ones(len(mesh.edges)), # -1
        torch.tensor(mesh.vertex_degree), # deg(mesh.vertices[i])
    ])

    return torch.sparse_coo_tensor(indices, values)


def cotan(vector0, vector1):
    return torch.sum(vector0*vector1) / torch.linalg.cross(vector0, vector1).square().sum(dim=-1).sqrt()


def laplacian_cotangens(mesh: trimesh.Trimesh):
    r"""
        idx vtx
         a   l
        / \ / \
        i-j u-v
        \ / \ /
         b   r
    """
    n = len(mesh.vertices)

    ij = mesh.face_adjacency_edges # (#edges_shared, 2)
    ab = mesh.face_adjacency_unshared # (#edges_unshared, 2)

    uv = torch.tensor(mesh.vertices[ij]) # (#edges_shared, 2, 3)
    lr = torch.tensor(mesh.vertices[ab]) # (#edges_unshared, 2, 3)

    ca = cotan(lr[:, 0] - uv[:, 0], lr[:, 0] - uv[:, 1])
    cb = cotan(lr[:, 1] - uv[:, 0], lr[:, 1] - uv[:, 1])

    w_ij = (0.5 * (ca + cb)).unsqueeze(1)

    indices = torch.empty((2, 4*len(ij)), dtype=torch.long)
    for index, (i, j) in enumerate(ij):
        indices[:, (index*4):(index*4+4)] = torch.tensor([[i, j, i, j], [j, i, i, j]])
    values = torch.hstack([-w_ij, -w_ij, w_ij, w_ij]).reshape(-1)

    return torch.sparse_coo_tensor(indices, values)
