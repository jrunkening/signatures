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
        torch.vstack([torch.arange(n).unsqueeze(0), torch.arange(n).unsqueeze(0)]), # for every (i, j) where i==j
    ])
    values = torch.hstack([
        -torch.ones(len(mesh.edges)), # -1
        torch.tensor(mesh.vertex_degree), # deg(mesh.vertices[i])
    ])

    return torch.sparse_coo_tensor(indices, values)
