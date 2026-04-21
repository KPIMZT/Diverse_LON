"""
ns_utils.py
===============
Coverage and novelty computation for novelty search.
"""

import torch
import numpy as np
from typing import Tuple

# ── genome layout ────────────────────────────────────────────────────────────

def genome_slices(num_gauss: int) -> Tuple[slice, slice]:
    return slice(0, num_gauss), slice(num_gauss, 2 * num_gauss)

def encode_genome(
    alphas: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([alphas, sigma])

def decode_genome(
    genome: torch.Tensor,
    num_gauss: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a, s = genome_slices(num_gauss)
    return genome[a], genome[s]

def decode_genome_batch(
    genomes: torch.Tensor,
    num_gauss: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a, s = genome_slices(num_gauss)
    return genomes[:, a], genomes[:, s]

def random_genome(
    num_gauss: int,
    device: str,
    generator: torch.Generator,
    alpha_min: float,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    alphas = torch.rand(num_gauss, device=device, generator=generator).clamp(min=alpha_min)
    alphas = alphas / alphas.max()
    sigma = (torch.rand(num_gauss, device=device, generator=generator) * (sigma_max - sigma_min) + sigma_min)
    return encode_genome(alphas, sigma)

def random_genome_batch(
    N: int,
    num_gauss: int,
    device: str,
    generator: torch.Generator,
    alpha_min: float,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    alphas = torch.rand(N, num_gauss, device=device, generator=generator).clamp(min=alpha_min)
    sigma = (torch.rand(N, num_gauss, device=device, generator=generator) * (sigma_max - sigma_min) + sigma_min)
    return torch.cat([alphas, sigma], dim=1)

# ── normalize / crossover / mutate ──────────────────────────────────────────

def normalize_genomes(
    genomes: torch.Tensor,
    num_gauss: int,
    alpha_min: float,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    g = genomes.clone()
    a_sl, s_sl = genome_slices(num_gauss)
    alphas = g[:, a_sl].clamp(min=alpha_min)
    alphas = alphas / alphas.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    g[:, a_sl] = alphas
    g[:, s_sl] = g[:, s_sl].clamp(sigma_min, sigma_max)
    return g

def crossover_genomes(
    parents: torch.Tensor,
    n_children: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mu, dim = parents.shape
    device = parents.device
    ia = torch.randint(0, mu, (n_children,), device=device, generator=generator)
    ib = torch.randint(0, mu, (n_children,), device=device, generator=generator)
    mask = torch.rand(n_children, dim, device=device, generator=generator) < 0.5
    return torch.where(mask, parents[ia], parents[ib])

def mutate_genomes(
    genomes: torch.Tensor,
    num_gauss: int,
    alpha_std: float,
    sigma_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    N, device = genomes.shape[0], genomes.device
    mutants = genomes.clone()
    a_sl, s_sl = genome_slices(num_gauss)
    mutants[:, a_sl] += torch.randn(N, num_gauss, device=device, generator=generator) * alpha_std
    mutants[:, s_sl] += torch.randn(N, num_gauss, device=device, generator=generator) * sigma_std
    return mutants

# ── coverage / novelty ──────────────────────────────────────────

def calc_coverage(
    features: torch.Tensor,
    scale: np.ndarray,
    bins: int,
) -> float:
    pts = features.detach().cpu().numpy()
    norm = pts / (scale + 1e-12)
    idxs = np.minimum((norm * bins + 1e-12).astype(int), bins - 1)
    return len(set(map(tuple, idxs))) / (bins ** pts.shape[1])

def compute_novelty(
    features: torch.Tensor,
    archive_features: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if archive_features.shape[0] == 0:
        return torch.zeros(features.shape[0], device=features.device)
    ref = torch.cat([archive_features, features], dim=0)
    dist = torch.cdist(features, ref, p=2)
    k_eff = min(k, dist.shape[1] - 1)
    if k_eff <= 0:
        return torch.zeros(features.shape[0], device=features.device)
    return torch.topk(dist, k_eff, largest=False).values.mean(dim=1)
