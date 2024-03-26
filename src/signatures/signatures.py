import math

import torch
import trimesh

from signatures.laplace import mass_matrix, laplacian_beltrami


def heat_kernel_signatures(
    mesh: trimesh.Trimesh,
    n_basis: int, n_time_steps: int, t_min: float = None, t_max: float = None,
    laplacian = laplacian_beltrami,
    device = "cpu"
):
    n_basis = min(len(mesh.vertices)-1, n_basis)

    return hks(
        laplacian(mesh).to(device), mass_matrix(mesh).to(device),
        n_basis, n_time_steps, t_min, t_max
    )


def hks(
    laplacian_mesh: torch.tensor, mass_matrix: torch.tensor,
    n_basis: int, n_time_steps: int, t_min: float = None, t_max: float = None,
):
    assert mass_matrix.device == laplacian_mesh.device
    device = laplacian_mesh.device

    spectrum, modes = torch.lobpcg(laplacian_mesh, n_basis, mass_matrix) # (n_basis,), (#v, n_basis)

    if t_min is None:
        t_min = torch.finfo(torch.float32).eps / spectrum.max()
    if t_max is None:
        t_max = torch.finfo(torch.float32).eps * 1e5 / spectrum.min()

    # times = torch.logspace(math.log10(t_min), math.log10(t_max), n_time_steps).to(device) # (n_time_steps,)
    times = torch.linspace(t_min, t_max, n_time_steps).to(device) # (n_time_steps,)

    phi2 = torch.square(modes) # (#v, n_basis)
    exp = torch.exp(-spectrum[..., None] * times[None]) # (n_basis, 1) * (1, n_time_steps) -> (n_basis, n_time_steps)
    signatures = torch.sum(phi2[..., None] * exp[None], dim=1) # (#v, n_basis, 1) * (1, n_basis, n_time_steps) -> (#v, n_basis, n_time_steps) -> (#v, n_time_steps)
    heat_trace = torch.sum(exp, dim=0, keepdim=True) # (1, n_time_steps)
    signatures = signatures / heat_trace # (#v, n_time_steps) / (1, n_time_steps) -> (#v, n_time_steps)

    return signatures
