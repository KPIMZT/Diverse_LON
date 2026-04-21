"""
msg_landscape.py
====================
MSG landscape class (D=2 grid evaluation, D-general point evaluation).
"""

import torch
import numpy as np
from typing import Tuple
from ns_utils import decode_genome
from msg_utils import compute_pairwise_vals, msg_eval


class MSGLandscapeIso:
    """
    Isotropic MSG landscape evaluator.

    Parameters
    ----------
    genome : (2M,) flat genome [alphas | sigma]
    means  : (M, D) Sobol fixed centers
    M      : number of Gaussians
    """

    def __init__(self, genome: torch.Tensor, means: torch.Tensor, M: int):
        self.M = M
        self.means = means
        self.alphas, self.sigma = decode_genome(genome, M)

    @torch.no_grad()
    def eval(self, X: torch.Tensor) -> torch.Tensor:
        """(N, D) -> (N,) landscape value f(x) = max_i alpha_i * exp(...)"""
        return msg_eval(X, self.means, self.alphas, self.sigma)

    @torch.no_grad()
    def grid_eval(
        self,
        resolution: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate on [0,1]^2 grid. D=2 only.

        Returns: (Xg, Yg, Z) each of shape (resolution, resolution).
        """
        D = self.means.shape[1]
        assert D == 2, f"grid_eval requires D=2, got D={D}"
        device = self.means.device
        xs = torch.linspace(0, 1, resolution, device=device)
        ys = torch.linspace(0, 1, resolution, device=device)
        Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
        XY = torch.stack([Xg.flatten(), Yg.flatten()], dim=1)
        Z = self.eval(XY).reshape(resolution, resolution)
        return Xg.cpu().numpy(), Yg.cpu().numpy(), Z.cpu().numpy()

    @torch.no_grad()
    def find_optima(
        self,
        atol: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analytically detect local/global optima.

        Returns
        -------
        local_means    : (K_local, D)
        local_fitness  : (K_local,)
        global_mean    : (1, D)
        global_fitness : (1,)

        NOTE: If multiple optima share the maximum fitness, only the
        first (by index) is returned as global; the rest stay in local.
        """
        vals = compute_pairwise_vals(self.alphas, self.means, self.sigma)
        cand = vals.max(dim=1).values

        is_c1 = torch.isclose(cand, self.alphas, atol=atol)
        vals_od = vals.clone()
        vals_od.fill_diagonal_(-float("inf"))
        is_c2 = ~torch.isclose(vals_od.max(dim=1).values, self.alphas, atol=atol)
        is_local = is_c1 & is_c2

        local_means = self.means[is_local]
        local_fitness = cand[is_local]

        if len(local_fitness) == 0:
            is_local = is_c1
            local_means = self.means[is_local]
            local_fitness = cand[is_local]

        g_idx = torch.argmax(local_fitness)
        mask = torch.ones(
            len(local_fitness), dtype=torch.bool, device=local_fitness.device
        )
        mask[g_idx] = False

        return (
            local_means[mask],
            local_fitness[mask],
            local_means[g_idx].unsqueeze(0),
            local_fitness[g_idx].unsqueeze(0),
        )
