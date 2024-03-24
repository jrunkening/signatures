import torch
import trimesh


def mass_matrix(mesh: trimesh.Trimesh):
    areas = torch.zeros(len(mesh.vertices))
    for face, area in zip(mesh.faces, mesh.area_faces):
        areas[face] += area/3.0

    return torch.diag(areas)
