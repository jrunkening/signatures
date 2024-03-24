# this file is copy from [msig/signature](https://github.com/DominikPenk/mesh-signatures/blob/main/msig/signature.py)

import math

import numpy
import scipy
import trimesh

from prototype.laplace import build_mass_matrix, build_laplace_beltrami_matrix


def hks(mesh: trimesh.Trimesh, n_basis: int, dim: int):
    # init
    m = build_mass_matrix(mesh)
    w = build_laplace_beltrami_matrix(mesh)
    n_basis = min(len(mesh.vertices)-1, n_basis)
    sigma = -0.01

    lu = scipy.sparse.linalg.splu((w - sigma*m).tocsc())
    op_inv = scipy.sparse.linalg.LinearOperator(
        matvec = lu.solve,
        shape = w.shape,
        dtype = w.dtype
    )
    values, vectors = scipy.sparse.linalg.eigsh(w, n_basis, m, sigma=sigma, OPinv=op_inv)

    # hks
    t_min  = 4 * math.log(10) / values[-1]
    t_max  = 4 * math.log(10) / values[1]
    times = numpy.geomspace(t_min, t_max, dim)

    phi2       = numpy.square(vectors[:, 1:])
    exp        = numpy.exp(-values[1:, None] * times[None])
    s          = numpy.sum(phi2[..., None] * exp[None], axis=1)
    heat_trace = numpy.sum(exp, axis=0)
    s          = s / heat_trace[None]

    return s
