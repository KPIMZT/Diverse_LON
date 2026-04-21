"""
lon_analysis.py
===================
LON feature extraction and normalization.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Callable, cast
from lon_utils import monotonize, decompose, convert_adj_network

# ── targetting features ─────────────────────────────────────────────────────

def global_funnel_size_gpu(
    adj: torch.Tensor,       
    opt_alphas: torch.Tensor,
    num_gauss: int, 
) -> float:
    num_opt = adj.shape[0]
    if num_opt == 0:
        return 0.0
    # monotonize:
    better = opt_alphas.unsqueeze(0) > opt_alphas.unsqueeze(1)  
    mono = adj * better.float()

    # decompose: 
    max_val, max_idx = mono.max(dim=1)  
    has_edge = max_val > 0
     
    sym = torch.zeros(num_opt, num_opt, device=adj.device)
    rows = torch.where(has_edge)[0]
    sym[rows, max_idx[rows]] = 1
    sym = sym + sym.t() 
    sym.clamp_(max=1)

    labels = torch.arange(num_opt, device=adj.device)
    for _ in range(num_opt):
        neighbors = torch.where(sym > 0, labels.unsqueeze(0).expand(num_opt, num_opt), labels.unsqueeze(1).expand(num_opt, num_opt))
        new_labels = neighbors.min(dim=1).values
        new_labels = torch.min(new_labels, labels)
        if (new_labels == labels).all():
            break
        labels = new_labels

    global_opt = opt_alphas.argmax()
    return (labels == labels[global_opt]).sum().item() / num_opt

GPU_FEATURE_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor, int], float]] = {
    "num_nodes": lambda adj, _, num_gauss: adj.shape[0]/num_gauss,
    "global_funnel_size": global_funnel_size_gpu,
}

def adj_to_features(
    feature_type: List[str],
    adj: torch.Tensor,
    opt_alphas: torch.Tensor,
    num_gauss: int, 
    device: str,
) -> torch.Tensor:
    features = []
    for name in feature_type:
        fn = GPU_FEATURE_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown feature: '{name}'. Available: {list(GPU_FEATURE_REGISTRY)}")
        features.append(fn(adj, opt_alphas, num_gauss))
    return torch.tensor(features, dtype=torch.float32, device=device)

def global_funnel_size_cpu(G: nx.DiGraph, num_gauss: int) -> float:
    """
    Fraction of nodes in the connected component containing the global optimum
    on the max-weight-edge subgraph.
    """
    decoG = decompose(monotonize(G))
    global_opt = max(decoG.nodes, key=lambda nd: decoG.nodes[nd].get("fitness", 0.0))
    for comp in nx.weakly_connected_components(decoG):
        if global_opt in comp:
            return len(comp) / len(decoG)
    return 0.0

CPU_FEATURE_REGISTRY: Dict[str, Callable[[nx.DiGraph, int], float]] = {
    "num_nodes": lambda G, num_gauss: len(G)/num_gauss,
    "global_funnel_size": global_funnel_size_cpu,
}

def adj_to_features_cpu(
    feature_type: List[str],
    adj: torch.Tensor,
    opt_alphas: torch.Tensor,
    opt_means: torch.Tensor,
    num_gauss: int, 
    device: str,
) -> torch.Tensor:
    G = convert_adj_network(adj.cpu().numpy(), opt_alphas.cpu().numpy(), opt_means.cpu().numpy())
    features = []
    for name in feature_type:
        fn = CPU_FEATURE_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown feature: '{name}'. Available: {list(CPU_FEATURE_REGISTRY)}")
        features.append(fn(G, num_gauss))
    return torch.tensor(features, dtype=torch.float32, device=device)

# ── all features from Graph object ─────────────────────────────────────────────────────

def run_analysis(G: nx.DiGraph, monoG: nx.DiGraph, decoG: nx.DiGraph, features: List[str], num_gauss: int) -> Dict[str, float|str]:
    result = {}
    opt = max(G.nodes, key=lambda n: G.nodes[n]['fitness'])
    sinks = [node for node, degree in decoG.out_degree() if degree == 0]
    R = monoG.reverse()
    C = [comp for comp in nx.weakly_connected_components(decoG)]
    for feature in features:
        if feature == 'num_nodes':
            value = len(G)/num_gauss
        elif feature == "edge_density":
            value = len(monoG.edges())/(0.5 * len(monoG) *(len(monoG)+1))
        elif feature == "num_sink":
            value = len(sinks)/num_gauss
        elif feature == "avg_path_sinks":
            value = np.mean([np.mean(list(nx.single_source_shortest_path_length(R, s).values())) for s in sinks])
        elif feature == "avg_path_opt":
            value = np.mean(list(nx.single_source_shortest_path_length(R, opt).values()))
        elif feature == "in_strength_sinks":
            value = np.mean([G.in_degree(sink, weight='prob') for sink in sinks])
        elif feature == "in_strength_opt":
            value = G.in_degree(opt, weight='prob')
        elif feature == "global_funnel_size":
            value = [len(comp) for comp in C if opt in comp].pop()/len(G)
        elif feature == "global_funnel_depth":
            sub = cast(nx.DiGraph, decoG.subgraph([comp for comp in C if opt in comp].pop()))
            value = nx.dag_longest_path_length(sub)
        result[feature] = value
    return result

def dict_to_tensor(
    features: Dict[str, float],
    order_list: List[str],
    device: str = "cpu",
) -> torch.Tensor:
    return torch.tensor([features[k] for k in order_list], dtype=torch.float32, device=device)