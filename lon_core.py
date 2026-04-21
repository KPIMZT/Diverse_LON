"""
lon_core.py
==========
LON construction entry points (isotropic & anisotropic).
"""

import torch
from typing import List, Tuple

from msg_utils import iso_mahal_sq, compute_pairwise_vals
from lon_utils import detect_local_optima, compute_basin_targets


def adj_to_features(
        feature_type: List[str],
        adj: torch.Tensor,
        opt_alphas: torch.Tensor,
        num_gauss: int, 
        device: str,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    all_features = []
    num_opt = adj.shape[0]
    better = opt_alphas.unsqueeze(0) > opt_alphas.unsqueeze(1)  
    mono = adj * better.float()
    max_val, max_idx = mono.max(dim=1)  
    has_edge = max_val > 0
    deco = torch.zeros_like(adj)
    rows = torch.where(has_edge)[0]
    deco[rows, max_idx[rows]] = 1.0
    sinks = (deco.sum(dim=1) == 0)
    opt = opt_alphas.argmax()
    sym = (deco + deco.t()).clamp(max=1)
    labels = torch.arange(num_opt, device=device)
    for _ in range(num_opt):
        neighbors = torch.where(sym > 0, labels.unsqueeze(0).expand(num_opt, num_opt),
                                labels.unsqueeze(1).expand(num_opt, num_opt))
        new_labels = torch.min(neighbors.min(dim=1).values, labels)
        if (new_labels == labels).all():
            break
        labels = new_labels
    mono_rev = mono.t()

    def bfs_distances(source_mask):
        """BFS from source nodes, returns (num_opt,) distance tensor. -1 = unreachable."""
        dist = torch.full((num_opt,), -1, dtype=torch.long, device=device)
        frontier = source_mask.clone()
        dist[frontier] = 0
        visited = frontier.clone()
        step = 0
        while frontier.any():
            step += 1
            neighbors = (mono_rev[frontier].sum(dim=0) > 0)
            frontier = neighbors & ~visited
            dist[frontier] = step
            visited |= frontier
        return dist
    
    value = num_opt/num_gauss
    if "num_nodes" in feature_type: 
        features.append(value)
    all_features.append(value)
    

    n_edges = (mono > 0).sum().item()
    value = n_edges / (0.5 * num_opt * (num_opt + 1))
    if "edge_density" in feature_type:
        features.append(value)
    all_features.append(value)   

    value = sinks.sum().item() / num_gauss
    if "num_sink" in feature_type:
        features.append(value)
    all_features.append(value)   
    

    sink_indices = torch.where(sinks)[0]
    path_means = []
    for s in sink_indices:
        mask = torch.zeros(num_opt, dtype=torch.bool, device=device)
        mask[s] = True
        dist = bfs_distances(mask)
        reachable = dist[dist >= 0].float()
        path_means.append(reachable.mean().item())
    value = torch.tensor(path_means, device=device).mean().item() if path_means else 0.0
    if "avg_path_sinks" in feature_type:
        features.append(value)
    all_features.append(value)   


    mask = torch.zeros(num_opt, dtype=torch.bool, device=device)
    mask[opt] = True
    dist = bfs_distances(mask)
    reachable = dist[dist >= 0].float()
    value = reachable.mean().item()
    if "avg_path_opt" in feature_type:
        features.append(value)
    all_features.append(value) 
     
    sink_indices = torch.where(sinks)[0]
    value = adj[:, sink_indices].sum(dim=0).mean().item()
    if "in_strength_sinks" in feature_type:
        features.append(value)
    all_features.append(value)
    
    value = adj[:, opt].sum().item()
    if "in_strength_opt" in feature_type:
        features.append(value)
    all_features.append(value)

    value = (labels == labels[opt]).sum().item() / num_opt
    if "global_funnel_size" in feature_type:
        features.append(value)
    all_features.append(value)
    return torch.tensor(features, dtype=torch.float32, device=device), torch.tensor(all_features, dtype=torch.float32, device=device)

@torch.no_grad()
def calc_adj(
    means: torch.Tensor,
    alphas: torch.Tensor,
    sigma: torch.Tensor,
    r: float,
    num_samples: int,
    generator: torch.Generator,
    batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_gauss, dim, device = means.shape[0], means.shape[1], means.device
    vals = compute_pairwise_vals(alphas, means, sigma)
    is_local_optimum = detect_local_optima(vals, alphas)
    opt_idx = torch.where(is_local_optimum)[0]
    num_opt = len(opt_idx)
    opt_alphas = alphas[opt_idx]
    opt_means = means[opt_idx]
    optima_id_map = torch.full((num_gauss,), -1, dtype=torch.long, device=device)
    optima_id_map[opt_idx] = torch.arange(num_opt, device=device)
    basin_target = compute_basin_targets(vals, is_local_optimum, optima_id_map, device)


    if batch_size <= 0:
        batch_size = num_opt
    adj_counts = torch.zeros((num_opt, num_opt), device=device)
    for b_start in range(0, num_opt, batch_size):
        b_end = min(b_start + batch_size, num_opt)
        b_size = b_end - b_start
        centers = opt_means[b_start:b_end]
        

        z = torch.randn(b_size, num_samples, dim, device=device, generator=generator)
        z = z / z.norm(dim=2, keepdim=True).clamp(min=1e-12)
        radii = torch.rand(b_size, num_samples, 1, device=device, generator=generator) ** (1.0 / dim)
        samples = (centers.unsqueeze(1) + r * radii * z).clamp(0.0, 1.0)
        s_flat = samples.reshape(-1, dim)

        mahal_s = iso_mahal_sq(s_flat, means, sigma)
        vals_s = alphas.unsqueeze(0) * torch.exp(-0.5 * mahal_s)
        max_ids = torch.argmax(vals_s, dim=1).reshape(b_size, num_samples)
        terminal = basin_target[max_ids]

        valid = terminal >= 0
        row_idx = torch.arange(b_start, b_end, device=device).unsqueeze(1).expand_as(terminal)
        row_flat = row_idx[valid]
        col_flat = terminal[valid]
        adj_counts.index_put_(
            (row_flat, col_flat),
            torch.ones(row_flat.shape[0], device=device),
            accumulate=True,
        )
    adj_counts /= num_samples
    return adj_counts, opt_alphas, opt_means
