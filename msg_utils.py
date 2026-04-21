"""
msg_utils.py
=================
Constants, Sobol centers, Mahalanobis, MSG evaluation.
"""
import torch
import numpy as np

def make_archetype1(
    means: torch.Tensor,
    sigma_c: float,
    sigma_s: float,
    alpha_min: float,
    generator: np.random.Generator,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sphere-like: single global basin, LON nodes ~ 1."""
    num_gauss, _ = means.shape
    device = means.device
    c_idx = int(generator.integers(0, num_gauss))
    mu_c = means[c_idx].unsqueeze(0)
    sigma_c_t = torch.tensor([sigma_c], device=device)
    d_sq = iso_mahal_sq(means, mu_c, sigma_c_t).squeeze(1)
    alphas = torch.exp(-0.5 * d_sq).clamp(min=alpha_min)-0.001
    alphas[c_idx] = 1
    sigma = torch.full((num_gauss,), sigma_s, device=device)
    sigma[c_idx] = sigma_c

    if verbose:
        print(f"[Archetype1] center={c_idx} alpha=[{alphas.min():.4f}, {alphas.max():.4f}]")
    return alphas, sigma

def make_archetype2(
    means: torch.Tensor,
    sigma_val: float,
    alpha_min: float,
    generator: np.random.Generator,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rastrigin-like: many local optima, distance-graded alpha."""
    num_gauss, _ = means.shape
    device = means.device
    c_idx = int(generator.integers(0, num_gauss))
    mu_c = means[c_idx]  # (D,)
    dists = (means - mu_c.unsqueeze(0)).norm(dim=1)
    corner_dists = torch.max(mu_c, 1.0 - mu_c)  # (D,)
    d_max = corner_dists.norm().clamp(min=1e-8)
    alphas = (1.0 - (1.0 - alpha_min) * (dists / d_max)).clamp(min=alpha_min)
    sigma = torch.full((num_gauss,), sigma_val, device=device)
    if verbose:
        print(f"[Archetype2] center={c_idx} "
              f"alpha=[{alphas.min():.4f}, {alphas.max():.4f}] ")
    return alphas, sigma

def make_archetype5(
    means: torch.Tensor,
    sigma_val: float,
    generator: torch.Generator,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearly uniform alpha (1 + small noise)."""
    num_gauss, _ = means.shape
    device = means.device
    alphas = torch.ones(num_gauss, device=device)
    alphas = alphas + torch.randn(num_gauss, device=device, generator=generator) * 0.01
    sigma = torch.full((num_gauss,), sigma_val, device=device)
    if verbose:
        print(f"[Archetype5] alpha=[{alphas.min():.4f}, {alphas.max():.4f}]")
    return alphas, sigma

# ── Mahalanobis / evaluation ────────────────────────────────────────────────

@torch.no_grad()
def iso_mahal_sq(
    X: torch.Tensor,
    means: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """||x - mu_i||^2 / sigma_i^2  ->  (..., M)"""
    shape = X.shape[:-1]
    X_flat = X.reshape(-1, X.shape[-1])
    # d = X_flat.unsqueeze(1) - means.unsqueeze(0)
    # dist_sq = (d**2).sum(dim=-1)
    dist_sq = (X_flat**2).sum(dim=1, keepdim=True)\
        - 2 * (X_flat @ means.T)\
        + (means**2).sum(dim=1, keepdim=True).T
    return (dist_sq / sigma.unsqueeze(0) ** 2).reshape(*shape, means.shape[0])

@torch.no_grad()
def compute_pairwise_vals(
    alphas: torch.Tensor,
    means: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """vals[i,j] = alpha_j * exp(-0.5 * ||mu_i - mu_j||^2 / sigma_j^2)"""
    mahal = iso_mahal_sq(means, means, sigma)
    return alphas.unsqueeze(0) * torch.exp(-0.5 * mahal)

@torch.no_grad()
def msg_eval(
    X: torch.Tensor,
    means: torch.Tensor,
    alphas: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """f(x) = max_i alpha_i * exp(-0.5 * ||x - mu_i||^2 / sigma_i^2)"""
    mahal = iso_mahal_sq(X, means, sigma)
    vals = alphas.unsqueeze(0) * torch.exp(-0.5 * mahal)
    return vals.max(dim=1).values
