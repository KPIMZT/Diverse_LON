"""
basin_compare.py
================
Compare analytical basin assignment vs gradient-descent basin assignment
for D=2, 5, 10 MSG instances.

For each of N_INSTANCES random genomes:
  - Compute analytical basin assignment (dominant Gaussian → basin_target)
  - Run gradient ascent from sample points and record which local optimum it converges to
  - Compute disagreement metrics and save to CSV
  - Optionally visualize side-by-side basin maps (D=2 only)
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

from ns_utils import decode_genome, random_genome
from msg_utils import compute_pairwise_vals, iso_mahal_sq
from lon_utils import detect_local_optima, compute_basin_targets

# ── default parameters ────────────────────────────────────────────────────────

LON_SEED    = 77
COORD_TOL   = 1e-2
N_SAMPLES   = 5000     # Sobol sample points (all dims)
RESOLUTION  = 100       # D=2 visualization grid resolution
N_INSTANCES = 100
SAVE_DIR    = "./results_basin_compare"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_MIN   = 0.0

# ── helpers ───────────────────────────────────────────────────────────────────

def make_means(dim: int, seed: int = 0, device: str = DEVICE) -> torch.Tensor:
    """Sobol centers (same as NSGeneration.__init__)."""
    num_gauss = 50 * dim
    torch.manual_seed(seed)
    engine = torch.quasirandom.SobolEngine(dim, scramble=True)
    return engine.draw(num_gauss).to(device)


def make_sample_points(
    dim:       int,
    n_samples: int = N_SAMPLES,
    device:    str = DEVICE,
    seed:      int = 1,         # make_means uses seed=0; use different seed here
) -> torch.Tensor:
    """
    Sobol quasi-random sample points in [0,1]^D.
    Uses a separate SobolEngine (different seed) from make_means.
    Returns (n_samples, dim).
    """
    torch.manual_seed(seed)
    engine = torch.quasirandom.SobolEngine(dim, scramble=True)
    return engine.draw(n_samples).to(device)


def analytical_basin_assign(
    XY:     torch.Tensor,   # (N, D)
    alphas: torch.Tensor,
    sigma:  torch.Tensor,
    means:  torch.Tensor,
    device: str = DEVICE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Analytical basin assignment: dominant Gaussian → basin_target.
    Returns (basin_flat, opt_idx, K).
    """
    num_gauss = means.shape[0]
    vals        = compute_pairwise_vals(alphas, means, sigma)
    is_local    = detect_local_optima(vals, alphas)
    opt_idx     = torch.where(is_local)[0]
    K           = len(opt_idx)
    optima_id_map = torch.full((num_gauss,), -1, dtype=torch.long, device=device)
    optima_id_map[opt_idx] = torch.arange(K, device=device)
    basin_target = compute_basin_targets(vals, is_local, optima_id_map, device)

    with torch.no_grad():
        mahal      = iso_mahal_sq(XY, means, sigma)
        vals_grid  = alphas.unsqueeze(0) * torch.exp(-0.5 * mahal)
        dominant   = vals_grid.argmax(dim=1)
        basin_flat = basin_target[dominant].cpu().numpy()

    return basin_flat, opt_idx.cpu().numpy(), K


def gd_basin_assign(
    XY:        torch.Tensor,   # (N, D)
    alphas:    torch.Tensor,
    sigma:     torch.Tensor,
    means:     torch.Tensor,
    opt_means: np.ndarray,     # (K, D)
    coord_tol: float = COORD_TOL,
    lr:        float = 1e-2,
    max_steps: int   = 2000,
    device:    str   = DEVICE,
) -> np.ndarray:
    """
    Run gradient ascent from all sample points in parallel.
    Returns (N,) int array; -1 if not converged to any known optimum.
    """
    opt_means_t = torch.tensor(opt_means, dtype=torch.float32, device=device)
    X = XY.clone()
    N = X.shape[0]

    with torch.no_grad():
        arange_N = torch.arange(N, device=device)
        for _ in range(max_steps):
            mahal = iso_mahal_sq(X, means, sigma)
            vals  = alphas.unsqueeze(0) * torch.exp(-0.5 * mahal)
            k     = vals.argmax(dim=1)
            f_k   = vals[arange_N, k]
            mu_k  = means[k]
            sig_k = sigma[k]
            scale = (lr * f_k / sig_k ** 2).clamp(max=1.0).unsqueeze(1)
            X     = (X + scale * (mu_k - X)).clamp(0.0, 1.0)

    dists    = (X.unsqueeze(1) - opt_means_t.unsqueeze(0)).abs().max(dim=2).values
    nearest  = dists.argmin(dim=1)
    min_dist = dists[arange_N, nearest]

    basin_flat = torch.where(min_dist < coord_tol, nearest,
                             torch.full_like(nearest, -1)).cpu().numpy()
    return basin_flat


def compute_metrics(
    ana_flat: np.ndarray,   # (N,)
    gd_flat:  np.ndarray,   # (N,)
    K: int,
) -> dict:
    """Compute disagreement metrics between two basin assignments."""
    total    = ana_flat.size
    valid    = (gd_flat >= 0)
    n_valid  = valid.sum()
    n_unconverged = total - n_valid

    agree    = (ana_flat == gd_flat) & valid
    disagree = (~agree) & valid

    n_agree    = agree.sum()
    n_disagree = disagree.sum()
    disagree_rate = n_disagree / n_valid if n_valid > 0 else float("nan")

    per_basin = {}
    for k in range(K):
        in_ana   = (ana_flat == k)
        in_gd    = (gd_flat  == k) & valid
        both     = in_ana & in_gd
        only_ana = in_ana & ~in_gd & valid
        only_gd  = in_gd & ~in_ana
        per_basin[k] = {
            "ana_size":  int(in_ana.sum()),
            "gd_size":   int(in_gd.sum()),
            "overlap":   int(both.sum()),
            "only_ana":  int(only_ana.sum()),
            "only_gd":   int(only_gd.sum()),
        }

    return {
        "n_optima":      K,
        "total_cells":   total,
        "n_valid_gd":    int(n_valid),
        "n_unconverged": int(n_unconverged),
        "n_agree":       int(n_agree),
        "n_disagree":    int(n_disagree),
        "disagree_rate": disagree_rate,
        "per_basin":     per_basin,
    }


def visualize_comparison(
    ana_flat:    np.ndarray,
    gd_flat:     np.ndarray,
    opt_means:   np.ndarray,
    resolution:  int,
    instance_id: int,
    save_path:   Optional[str] = None,
):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
    """D=2 only: side-by-side basin maps + disagreement."""
    K = opt_means.shape[0]
    base_cmap = plt.get_cmap("tab20", K)
    colors = [base_cmap(k) for k in range(K)] + [(0.0, 0.0, 0.0, 1.0)]
    cmap = mcolors.ListedColormap(colors)

    ana_map    = ana_flat.reshape(resolution, resolution)
    gd_map     = gd_flat.reshape(resolution, resolution)
    gd_display = gd_map.copy()
    gd_display[gd_map < 0] = K
    disagree   = (ana_map != gd_map) & (gd_map >= 0)

    gd_found = np.isin(np.arange(K), np.unique(gd_flat[gd_flat >= 0]))

    def _scatter_optima(ax):
        if gd_found.any():
            ax.scatter(opt_means[gd_found, 0], opt_means[gd_found, 1],
                       c="white", edgecolors="black", s=60, zorder=5, label="GD found")
        if (~gd_found).any():
            ax.scatter(opt_means[~gd_found, 0], opt_means[~gd_found, 1],
                       c="red", edgecolors="black", s=60, zorder=5,
                       marker="X", label="GD not found")

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(axes[:2],
                                [ana_map.T, gd_display.T],
                                [r"$\text{BoA}_{ours}$", r"$\text{BoA}_{GD}$"]):
        ax.imshow(data, origin="lower", cmap=cmap,
                  vmin=0, vmax=K + 1, extent=[0, 1, 0, 1], aspect="equal")
        _scatter_optima(ax)
        ax.set_title(title, fontsize=32, pad=15)
        ax.set_xlabel("$x_1$", fontsize=40); ax.set_ylabel("$x_2$", fontsize=40)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.tick_params(labelsize=25)
        ax.tick_params(axis="x", pad=15)
    axes[2].imshow(disagree.T.astype(float), origin="lower",
                   cmap="Reds", vmin=0, vmax=1,
                   extent=[0, 1, 0, 1], aspect="equal")
    _scatter_optima(axes[2])
    axes[2].set_title("Differences", fontsize=32, pad=15)
    axes[2].set_xlabel("$x_1$", fontsize=40); axes[2].set_ylabel("$x_2$", fontsize=40)
    axes[2].set_xticks(ticks); axes[2].set_yticks(ticks)
    axes[2].tick_params(labelsize=25)
    axes[2].tick_params(axis="x", pad=15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    # plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def make_grid_points(resolution: int, device: str = DEVICE) -> torch.Tensor:
    """D=2 regular grid for visualization only."""
    xs = torch.linspace(0, 1, resolution, device=device)
    Xg, Yg = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([Xg.flatten(), Yg.flatten()], dim=1)


def run_ex_RQ1(
    dim:         int  = 2,
    n_instances: int  = N_INSTANCES,
    n_samples:   int  = N_SAMPLES,
    resolution:  int  = RESOLUTION,
    visualize:   bool = True,
    save_dir:    str  = SAVE_DIR,
):
    num_gauss  = 50 * dim
    R          = num_gauss ** (-1 / dim)
    sigma_max  = 3 * R
    sigma_min  = R / 4
    os.makedirs(save_dir, exist_ok=True)
    dim_save_dir = os.path.join(save_dir, f"dim{dim}")
    os.makedirs(dim_save_dir, exist_ok=True)

    means = make_means(dim=dim, seed=0, device=DEVICE)
    # experiment: Sobol points (all dims)
    XY = make_sample_points(dim=dim, n_samples=n_samples*dim, device=DEVICE)
    # visualization: D=2 grid (separate from experiment)
    XY_grid = make_grid_points(resolution, device=DEVICE) if (visualize and dim == 2) else None
    N     = XY.shape[0]
    print(f"[D={dim}]  num_gauss={num_gauss}  n_sample_points={N}")

    summary_rows = []

    for inst in range(n_instances):
        print(f"\n  [Instance {inst+1}]")
        gen_rng = torch.Generator(device=DEVICE).manual_seed(inst)
        genome  = random_genome(
            num_gauss=num_gauss, device=DEVICE, generator=gen_rng,
            alpha_min=ALPHA_MIN, sigma_min=sigma_min, sigma_max=sigma_max,
        )
        alphas, sigma = decode_genome(genome, num_gauss)

        # ── analytical basin ──────────────────────────────────────────
        ana_flat, opt_idx, K = analytical_basin_assign(XY, alphas, sigma, means, DEVICE)
        opt_means_np = means[opt_idx].cpu().numpy()
        print(f"    K={K} local optima")

        # ── GD basin ──────────────────────────────────────────────────
        gd_flat = gd_basin_assign(XY, alphas, sigma, means, opt_means_np,
                                  coord_tol=COORD_TOL, device=DEVICE)

        # ── metrics ───────────────────────────────────────────────────
        metrics = compute_metrics(ana_flat, gd_flat, K)
        print(f"    disagree_rate={metrics['disagree_rate']:.4f}  "
              f"unconverged={metrics['n_unconverged']}")

        row = {"dim": dim, "instance": inst+1,
               **{k: v for k, v in metrics.items() if k != "per_basin"}}
        summary_rows.append(row)

        # per-basin CSV
        per_basin_path = os.path.join(dim_save_dir, f"inst{inst+1}_per_basin.csv")
        with open(per_basin_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["basin", "ana_size", "gd_size",
                                               "overlap", "only_ana", "only_gd"])
            w.writeheader()
            for k, d in metrics["per_basin"].items():
                w.writerow({"basin": k, **d})

        # ── visualization (D=2 only, grid-based) ─────────────────────
        if visualize and dim == 2:
            ana_grid, _, _ = analytical_basin_assign(XY_grid, alphas, sigma, means, DEVICE)
            gd_grid        = gd_basin_assign(XY_grid, alphas, sigma, means, opt_means_np,
                                             coord_tol=COORD_TOL, device=DEVICE)
            viz_path = os.path.join(dim_save_dir, f"inst{inst+1}_basin_compare.svg")
            visualize_comparison(ana_grid, gd_grid, opt_means_np,
                                 resolution, inst+1, save_path=viz_path)

    summary_path = os.path.join(dim_save_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    for dim in [2, 5, 10]:
        run_ex_RQ1(dim=dim, n_instances=N_INSTANCES)

