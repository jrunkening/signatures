# this file is copy from [msig/laplace](https://github.com/DominikPenk/mesh-signatures/blob/main/msig/laplace.py)

import math

import numpy as np
import scipy
import trimesh


def build_mass_matrix(mesh : trimesh.Trimesh) -> scipy.sparse.diags:
    """Build the sparse diagonal mass matrix for a given mesh

    Args:
        mesh (trimesh.Trimesh): Mesh to use.

    Returns:
        A sparse diagonal matrix of size (#vertices, #vertices).
    """
    areas = np.zeros(shape=(len(mesh.vertices)))
    for face, area in zip(mesh.faces, mesh.area_faces):
        areas[face] += area / 3.0

    return scipy.sparse.diags(areas)


def build_laplace_beltrami_matrix(mesh : trimesh.Trimesh) -> scipy.sparse.coo_matrix:
    """Build the sparse laplace beltrami matrix of the given mesh M=(V, E).
    This is a positive semi-definite matrix C:

           -1         if (i, j) in E
    C_ij =  deg(V(i)) if i == j
            0         otherwise

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C
    """
    n = len(mesh.vertices)
    IJ = np.concatenate([
        mesh.edges,
        [[i, i] for i in range(n)]
    ], axis=0)
    V  = np.concatenate([
        [-1 for _ in range(len(mesh.edges))],
        mesh.vertex_degree
    ], axis= 0)


    A = scipy.sparse.coo_matrix((V, (IJ[..., 0], IJ[..., 1])), shape=(n, n), dtype=np.float64)
    return A


def build_cotangens_matrix(mesh : trimesh.Trimesh) -> scipy.sparse.coo_matrix:
    """Build the sparse cotangens weight matrix of the given mesh M=(V, E).
    This is a positive semi-definite matrix C:

           -0.5 * (tan(a) + tan(b))  if (i, j) in E
    C_ij = -sum_{j in N(i)} (C_ij)   if i == j
            0                        otherwise

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    n = len(mesh.vertices)
    ij = mesh.face_adjacency_edges
    ab = mesh.face_adjacency_unshared

    uv = mesh.vertices[ij]
    lr = mesh.vertices[ab]


    def cotan(v1, v2):
        return np.sum(v1*v2) / np.linalg.norm(np.cross(v1, v2), axis=-1)

    ca = cotan(lr[:, 0] - uv[:, 0], lr[:, 0] - uv[:, 1])
    cb = cotan(lr[:, 1] - uv[:, 0], lr[:, 1] - uv[:, 1])

    wij = np.maximum(0.5 * (ca + cb), 0.0)

    I = []
    J = []
    V = []
    for idx, (i, j) in enumerate(ij):
        I += [i, j, i, j]
        J += [j, i, i, j]
        V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]

    A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
    return A


def build_mesh_laplace_matrix(mesh : trimesh.Trimesh) -> scipy.sparse.coo_matrix:
    """Build the sparse mesh laplacian matrix of the given mesh M=(V, E).
    This is a positive semi-definite matrix C:

           -1/(4pi*h^2) * e^(-||vi-vj||^2/(4h)) if (i, j) in E
    C_ij = -sum_{j in N(i)} (C_ij)              if i == j
            0                                   otherwise
    here h is the average edge length

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    n = len(mesh.vertices)
    h = np.mean(mesh.edges_unique_length)
    a = 1.0 / (4 * math.pi * h*h)
    wij = a * np.exp(-mesh.edges_unique_length**2/(4.0*h))
    I = []
    J = []
    V = []
    for idx, (i, j) in enumerate(mesh.edges_unique):
        I += [i, j, i, j]
        J += [j, i, i, j]
        V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]

    A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
    return A
