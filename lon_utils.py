"""
msg_lon_core.py
===============
Shared LON construction primitives.
"""

import torch
import numpy as np
import networkx as nx
import logging

def detect_local_optima(
    vals: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian i is a local optimum iff no other Gaussian j achieves
    the same or higher value at mu_i.
    """
    vals_od = vals.clone()
    vals_od.fill_diagonal_(-float("inf"))
    return vals_od.max(dim=1).values < alphas

def compute_basin_targets(
    vals: torch.Tensor,
    is_local_optimum: torch.Tensor,
    optima_id_map: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    num_gauss = vals.shape[0]
    basin_target = torch.full((num_gauss,), -1, dtype=torch.long, device=device)
    basin_target[is_local_optimum] = optima_id_map[is_local_optimum]

    # Off-diagonal argmax:
    vals_od = vals.clone()
    vals_od.fill_diagonal_(-float("inf"))
    max_indices = torch.argmax(vals_od, dim=1)

    for _ in range(num_gauss):
        unresolved = basin_target == -1
        if not unresolved.any():
            break
        u_idx = torch.where(unresolved)[0]
        cand = basin_target[max_indices[u_idx]]
        resolved = cand != -1
        if resolved.any():
            basin_target[u_idx[resolved]] = cand[resolved]
        else:
            break

    still = torch.where(basin_target == -1)[0]
    if len(still) > 0:
        logging.warning(
            f"basin_target: {len(still)} Gaussians unresolved, "
            f"falling back to nearest optimum."
        )
        opt_cols = torch.where(is_local_optimum)[0]
        nn_idx = torch.argmax(vals[still][:, is_local_optimum], dim=1)
        basin_target[still] = optima_id_map[opt_cols[nn_idx]]

    return basin_target


def monotonize(G: nx.DiGraph) -> nx.DiGraph:
    """Keep only edges where the target has strictly higher fitness than the source."""
    mono_G = nx.DiGraph()
    mono_G.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if G.nodes[v].get("fitness", 0.0) > G.nodes[u].get("fitness", 0.0):
            mono_G.add_edge(u, v, **data)
    return mono_G

def decompose(G: nx.DiGraph) -> nx.DiGraph:
    """
    Build subgraph retaining only the highest-prob outgoing edge per node.
    """
    deco_G = nx.DiGraph()
    deco_G.add_nodes_from(G.nodes(data=True))
    for n in G.nodes:
        out_edges = list(G.out_edges(n, data=True))
        if not out_edges:
            continue
        best = max(out_edges, key=lambda e: e[2].get("prob", 0.0))
        deco_G.add_edge(best[0], best[1], **best[2])
    return deco_G

def convert_adj_network(
    adj: np.ndarray,
    fitness: np.ndarray,
    optima_means: np.ndarray,
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from adjacency/reachability matrices."""
    num_opt = adj.shape[0]
    G = nx.DiGraph()
    for i in range(num_opt):
        attrs = {"fitness": float(fitness[i]), "pos": optima_means[i].tolist()}
        G.add_node(i, **attrs)
    rows, cols = np.nonzero(adj)
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), prob=float(adj[i, j]))
    return G
