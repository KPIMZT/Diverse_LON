"""
msg_ns_io.py
============
Save / load for EvolutionStrategy.
"""

import os
import torch

# ── shared config ────────────────────────────────────────────────────────

_SCALAR_ATTRS = (
    "num_gauss", "dim", "_mu", "_lambda", "generations",
    "r", "num_samples", "feature_type", "seed", "lon_seed",
    "novelty_threshold", "novelty_k",
    "alpha_std", "alpha_min",
    "sigma_std", "sigma_max", "sigma_min",
    "use_crossover", "init_parents", "plus_comma",
    "compute_random_baseline",
)

_COVERAGE_ATTRS = (
    "bins", "feature_scale",
    "coverage_archive", "coverage_novelty", "coverage_random",
)

# ── save / load ──────────────────────────────────────────────────────────

def save_ns(ns, path: str) -> None:
    """Save EvolutionStrategy state to .pt file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    payload = {attr: getattr(ns, attr) for attr in _SCALAR_ATTRS}
    payload["means"] = ns.means.cpu()
    payload["_records_genomes"] = torch.stack([rec["genome"] for rec in ns._all_records])
    payload["_records_features"] = torch.stack([rec["features"] for rec in ns._all_records])
    payload["_records_all_features"] = torch.stack([rec["all_features"] for rec in ns._all_records])
    payload["_records_meta"] = [
        {"gen": rec["gen"], "archived": rec["archived"], "population": rec["population"]}
        for rec in ns._all_records
    ]

    # coverage (optional: only present after compute_coverage_all)
    has_coverage = hasattr(ns, "bins")
    payload["has_coverage"] = has_coverage
    if has_coverage:
        for attr in _COVERAGE_ATTRS:
            payload[attr] = getattr(ns, attr)
        payload["grid_cell_idx_archive"] = ns.grid_cell_idx_archive
        payload["grid_cell_idx_novelty"] = ns.grid_cell_idx_novelty
        payload["grid_cell_idx_random"] = ns.grid_cell_idx_random
        payload["initial_cell_keys"] = list(ns.initial_cell_keys)

    torch.save(payload, path)
    print(f"saved: {path}")

def load_ns(path: str, device: str = "cpu"):
    """Load EvolutionStrategy state from .pt file.

    The loaded instance supports analysis (coverage, visualization)
    but not resuming run() (generators are not restored).
    """
    from ns_core import NSGeneration

    payload = torch.load(path, map_location=device, weights_only=False)
    ns = NSGeneration.__new__(NSGeneration)

    for attr in _SCALAR_ATTRS:
        setattr(ns, attr, payload[attr])
    ns.device = device
    ns.means = payload["means"].to(device)
    genomes = payload["_records_genomes"]
    features = payload["_records_features"]
    all_features = payload["_records_all_features"]
    ns._all_records = [
        {"genome": genomes[i], "features": features[i], "all_features":all_features[i], **payload["_records_meta"][i]}
        for i in range(len(payload["_records_meta"]))
    ]

    if payload.get("has_coverage", False):
        for attr in _COVERAGE_ATTRS:
            setattr(ns, attr, payload[attr])
        ns.grid_cell_idx_archive = payload["grid_cell_idx_archive"]
        ns.grid_cell_idx_novelty = payload["grid_cell_idx_novelty"]
        ns.grid_cell_idx_random = payload["grid_cell_idx_random"]
        ns.initial_cell_keys = set(tuple(k) for k in payload.get("initial_cell_keys", []))

    print(f"loaded: {path} ({len(ns._all_records)} records)")
    return ns