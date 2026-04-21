"""
ns_viz.py
=============
Visualization functions for EvolutionStrategy (standalone, take ES as arg).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Optional

def plot_coverage(es, save_path: Optional[str] = None) -> None:
    if not es.coverage_archive:
        print("please call compute_coverage_all()!")
        return

    gens = list(range(1, len(es.coverage_archive) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=es.coverage_novelty,
                             mode="lines", name="novelty(all)",
                             line=dict(color="orange", width=3)))
    fig.add_trace(go.Scatter(x=gens, y=es.coverage_archive,
                             mode="lines", name="novelty(archive)",
                             line=dict(color="orange", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=gens, y=es.coverage_random,
                             mode="lines", name="Random",
                             line=dict(color="gray", width=3, dash="dot")))

    axis_common = dict(
        showline=True, linewidth=1.5, linecolor="black",
        ticks="inside", tickwidth=1.5, ticklen=5,
        mirror=True,
        gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
    )

    fig.update_layout(
        xaxis=dict(title="Generation", **axis_common),
        yaxis=dict(title="Grid Coverage", **axis_common),
        font=dict(family="Times New Roman", size=16, color="black"),
        legend=dict(
            x=0.1, y=0.95, xanchor="left", yanchor="top",
            bordercolor="black", borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=14),
        ),
        width=600, height=450,
        margin=dict(l=70, r=20, t=20, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    if save_path:
        fig.write_image(save_path, scale=3)
    else:
        fig.show()


def plot_grid(
    es,
    population: str = "archive",
    scale: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    if len(es.feature_type) != 2:
        print("only 2-dimensional feature space supported")
        return

    cell_idx= {
        "novelty": es.grid_cell_idx_novelty,
        "archive": es.grid_cell_idx_archive,
        "random": es.grid_cell_idx_random,
    }[population]

    scale = (
        scale
        if scale is not None
        else (es.feature_scale if es.feature_scale is not None else np.ones(2))
    )
    cell_w = scale / es.bins

    fig = go.Figure()

    init_x, init_y = [], []
    disc_x, disc_y = [], []
    for cx, cy in cell_idx.keys():
        if (cx, cy) in es.initial_cell_keys:
            init_x.append(cx * cell_w[0] + cell_w[0] / 2)
            init_y.append(cy * cell_w[1] + cell_w[1] / 2)
        else:
            disc_x.append(cx * cell_w[0] + cell_w[0] / 2)
            disc_y.append(cy * cell_w[1] + cell_w[1] / 2)

    marker_common = dict(
        symbol="square",
        size=max(4, 300 // es.bins),
        line=dict(width=0),
    )

    if disc_x:
        fig.add_trace(go.Scatter(
            x=disc_x, y=disc_y, mode="markers", name="discovered",
            marker=dict(color="rgba(255,165,0,0.5)", **marker_common),
        ))
    if init_x:
        fig.add_trace(go.Scatter(
            x=init_x, y=init_y, mode="markers", name="initial",
            marker=dict(color="rgba(255,0,0,0.7)", **marker_common),
        ))


    shapes = []
    for i in range(es.bins + 1):
        shapes.append(dict(type="line", x0=i * cell_w[0], x1=i * cell_w[0],
                           y0=0, y1=scale[1],
                           line=dict(color="lightgray", width=0.5)))
        shapes.append(dict(type="line", x0=0, x1=scale[0],
                           y0=i * cell_w[1], y1=i * cell_w[1],
                           line=dict(color="lightgray", width=0.5)))

    axis_common = dict(
        showline=True, linewidth=1.5, linecolor="black",
        ticks="inside", tickwidth=1.5, ticklen=5,
        mirror=True,
    )
    fig.update_layout(
        xaxis=dict(title=es.feature_type[0], range=[0, 1], **axis_common),
        yaxis=dict(title=es.feature_type[1], range=[0, 1],
                   scaleanchor="x", **axis_common),
        font=dict(family="Times New Roman", size=16, color="black"),
        legend=dict(
            x=0.65, y=0.95, xanchor="left", yanchor="top",
            itemsizing="constant",
            bordercolor="black", borderwidth=1,
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(size=16),
        ),
        shapes=shapes,
        width=455, height=450,
        margin=dict(l=70, r=20, t=20, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    if save_path:
        fig.write_image(save_path, scale=3)
    else:
        fig.show()


def plot_grid_mpl(
    es,
    population: str = "archive",
    scale: Optional[np.ndarray] = None,
    figsize: tuple = (8, 7),
    color_discovered: str = "#000000",
    save_path: Optional[str] = None,
) -> None:

    if len(es.feature_type) != 2:
        print("only 2-dimensional feature space supported")
        return

    cell_idx = {
        "novelty": es.grid_cell_idx_novelty,
        "archive": es.grid_cell_idx_archive,
        "random":  es.grid_cell_idx_random,
    }[population]

    scale = (
        scale
        if scale is not None
        else (es.feature_scale if es.feature_scale is not None else np.ones(2))
    )

    # 0 = empty, 1 = discovered, 2 = initial
    grid = np.zeros((es.bins, es.bins), dtype=int)
    for cx, cy in cell_idx.keys():
        if (cx, cy) in es.initial_cell_keys:
            grid[cy, cx] = 2
        else:
            grid[cy, cx] = 1

    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    def _mathtt(s):
        return r"$\mathtt{" + s.replace("_", r"\_") + r"}$"

    cmap = plt.matplotlib.colors.ListedColormap(["white", color_discovered, "black"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        grid,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=2,
        extent=[0, float(scale[0]), 0, float(scale[1])],
        aspect="equal",
        interpolation="none",
    )

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks([t * float(scale[0]) for t in ticks])
    ax.set_yticks([t * float(scale[1]) for t in ticks])
    ax.set_xlim(0, float(scale[0]))
    ax.set_ylim(0, float(scale[1]))
    ax.set_xlabel(_mathtt(es.feature_type[0]), fontsize=36)
    ax.set_ylabel(_mathtt(es.feature_type[1]), fontsize=36)
    ax.tick_params(labelsize=30)
    ax.tick_params(axis="x", pad=15)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()






def plot_coverage_boxplot(
    result_dir:  str,
    dims:        List[int] = [2, 5, 10],
    seeds:       List[int] = list(range(10)),
    save_path:   Optional[str] = None,
) -> None:
    """
    Box plot of final grid coverage across seeds for each condition and dimension.

    9 boxes: for each dim in dims → (NS archetype, NS no-archetype, Random)

    File naming convention:
      archetype     : dim{D}_seed{S}.pt
      no archetype  : dim{D}_nonarch_seed{S}.pt

    Parameters
    ----------
    result_dir : directory containing all .pt files
    dims       : list of dimensions, default [2, 5, 10]
    seeds      : list of seeds, default 0-9
    save_path  : if given, save figure to this path; otherwise plt.show()
    """
    import os

    conditions = [
        (r"\texttt{random}", lambda d, s: f"dim{d}_seed{s}_nonarch.pt", "coverage_random"),
        (r"\texttt{NS}",     lambda d, s: f"dim{d}_seed{s}_nonarch.pt", "coverage_novelty"),
        (r"\texttt{NS}$^{+}$", lambda d, s: f"dim{d}_seed{s}.pt",       "coverage_novelty"),
    ]

    # data[dim_idx][cond_idx] = list of final coverage values
    data = [[[] for _ in conditions] for _ in dims]

    for di, dim in enumerate(dims):
        for ci, (label, fname_fn, key) in enumerate(conditions):
            for seed in seeds:
                path = os.path.join(result_dir, fname_fn(dim, seed))
                if not os.path.exists(path):
                    print(f"[warn] not found: {path}")
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                cov_list = payload.get(key, [])
                if cov_list:
                    data[di][ci].append(float(cov_list[-1]))

    n_dims = len(dims)
    n_cond = len(conditions)
    cond_palette = ["#000000", "#EFEFEF", "#EE1010"]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
        })
    fig, ax = plt.subplots(figsize=(10,4))

    group_width = 0.8
    box_width   = group_width / n_cond
    gap         = 0.4

    group_centers = [1 + di * (1 + gap) for di in range(n_dims)]

    for di, (dim, center) in enumerate(zip(dims, group_centers)):
        for ci in range(n_cond):
            pos = center + (ci - (n_cond - 1) / 2) * box_width
            bp = ax.boxplot(
                [data[di][ci]],
                positions=[pos],
                widths=box_width * 0.85,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
            )
            bp["boxes"][0].set_facecolor(cond_palette[ci])
            bp["boxes"][0].set_alpha(0.8)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"$d={d}$" for d in dims], fontsize=16)
    ax.set_xlim(group_centers[0] - 0.7, group_centers[-1] + 0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final Coverage", fontsize=16)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="y", labelsize=16)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cond_palette[ci], alpha=0.8, label=label)
        for ci, (label, _, _) in enumerate(conditions)
    ]

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()
















'''
def plot_grid(
    es,
    store_key: str = "archive",
    scale: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    if len(es.feature_type) != 2:
        print("only 2-dimensional feature space supported")
        return

    store = {
        "archive": es.grid_cell_data,
        "NSall": es.grid_cell_data_NSall,
        "random": es.grid_cell_data_random,
    }[store_key]

    scale = (
        scale
        if scale is not None
        else (es.feature_scale if es.feature_scale is not None else np.ones(2))
    )
    cell_w = scale / es.bins

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.rcParams["font.size"] = 18

    for cx, cy in store.keys():
        is_init = (cx, cy) in es.initial_cell_keys
        ax.add_patch(
            plt.Rectangle(
                (cx * cell_w[0], cy * cell_w[1]),
                cell_w[0],
                cell_w[1],
                facecolor="red" if is_init else "steelblue",
                alpha=0.7 if is_init else 0.5,
            )
        )

    for i in range(es.bins + 1):
        ax.axhline(i * cell_w[1], color="lightgray", lw=0.4)
        ax.axvline(i * cell_w[0], color="lightgray", lw=0.4)

    ax.set_xlim(0, scale[0])
    ax.set_ylim(0, scale[1])
    ax.set_xlabel(es.feature_type[0])
    ax.set_ylabel(es.feature_type[1])
    ax.set_title(f"Grid [{store_key}]  {len(store)} cells")
    ax.legend(
        handles=[
            Patch(facecolor="red", alpha=0.7, label="initial"),
            Patch(facecolor="steelblue", alpha=0.5, label="discovered"),
        ],
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
'''

'''
def plot_scatter(
    es,
    store_key: str = "archive",
    scale: Optional[np.ndarray] = None,
    figsize: tuple = (7, 6),
    s: float = 20,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> None:
    if len(es.feature_type) != 2:
        print("only 2-dimensional feature space supported")
        return

    scale = scale if scale is not None else np.ones(2)
    feat_src = {
        "archive": es.archive_features,
        "NSall": es.NSall_features,
        "random": es.rand_features,
    }
    features_t = feat_src[store_key]
    if features_t is None:
        print(f"No features for {store_key}")
        return
    pts = features_t.detach().cpu().numpy()

    n = len(pts)
    is_init = np.zeros(n, dtype=bool)
    if store_key in ("archive", "NSall"):
        is_init[: es._mu] = True

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams["font.size"] = 20

    mask_disc = ~is_init
    if mask_disc.any():
        ax.scatter(
            pts[mask_disc, 0],
            pts[mask_disc, 1],
            s=s,
            color="steelblue",
            alpha=alpha,
            label="discovered",
            zorder=3,
        )
    if is_init.any():
        ax.scatter(
            pts[is_init, 0],
            pts[is_init, 1],
            s=s * 2,
            color="red",
            alpha=0.8,
            label="initial",
            zorder=4,
        )

    ax.set_xlim(0, scale[0])
    ax.set_ylim(0, scale[1])
    ax.set_xlabel(es.feature_type[0])
    ax.set_ylabel(es.feature_type[1])
    ax.set_title(f"Scatter [{store_key}]  {n} points")
    ax.legend(fontsize=16)
    ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def rebuild_grid_from_features(
    es,
    scale: np.ndarray,
    bins: Optional[int] = 10,
    save_path: Optional[str] = None,
    plot: bool = True,
    save_first: bool = True,
    save_last: bool = False,
    save_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[tuple, dict]]:
    """Reassign features to grid cells at given scale/bins."""
    plt.rcParams["font.size"] = 23
    bins = bins or es.bins
    cell_w = scale / bins

    def _make_entry(i, genomes, features, graphs):
        entry = {
            "genome": genomes[i].detach().clone().cpu(),
            "features": features[i].detach().clone().cpu(),
        }
        if graphs is not None and i < len(graphs):
            entry["G"] = graphs[i]
        return entry

    def _assign_cells(features, genomes, graphs=None):
        pts = features.detach().cpu().numpy()
        norm = pts / (scale + 1e-12)
        idxs = np.minimum((norm * bins + 1e-12).astype(int), bins - 1)
        store_first, store_last = {}, {}
        for i, key in enumerate(map(tuple, idxs)):
            entry = _make_entry(i, genomes, features, graphs)
            if key not in store_first:
                store_first[key] = entry
            store_last[key] = entry
        return store_first, store_last

    def _graphs_from_records(pop_key):
        if not es._all_records:
            return None
        return [rec.get("G") for rec in es._all_records if rec["population"] == pop_key]

    def _graphs_from_grid(features_t, grid_store):
        feat_to_g = {
            tuple(val["features"].numpy().tolist()): val.get("G")
            for val in grid_store.values()
        }
        return [
            feat_to_g.get(tuple(features_t[i].cpu().numpy().tolist()))
            for i in range(len(features_t))
        ]

    if es._all_records:
        ga = _graphs_from_records("archive")
        gn = _graphs_from_records("NSall")
        gr = _graphs_from_records("random")
    else:
        ga = _graphs_from_grid(es.archive_features, es.grid_cell_data)
        gn = _graphs_from_grid(es.NSall_features, es.grid_cell_data_NSall)
        gr = (
            _graphs_from_grid(es.rand_features, es.grid_cell_data_random)
            if es.rand_features is not None
            else None
        )

    grid_a_f, grid_a_l = _assign_cells(es.archive_features, es.archive, ga)
    grid_n_f, grid_n_l = _assign_cells(es.NSall_features, es.NSall, gn)
    if es.rand_features is not None:
        grid_r_f, grid_r_l = _assign_cells(es.rand_features, es.rand, gr)
    else:
        grid_r_f, grid_r_l = {}, {}

    grid_archive = grid_a_f if save_first else grid_a_l
    grid_NSall = grid_n_f if save_first else grid_n_l
    grid_rand = grid_r_f if save_first else grid_r_l

    pts_init = es.NSall_features[: es._mu].detach().cpu().numpy()
    norm_init = pts_init / (scale + 1e-12)
    idxs_init = np.minimum((norm_init * bins + 1e-12).astype(int), bins - 1)
    cells_init = set(map(tuple, idxs_init))

    grids = {
        "archive": (grid_archive, "archive"),
        "NSall": (grid_NSall, "NS"),
        "random": (grid_rand, "random"),
    }

    def _save_store(store, pt_path):
        out = {}
        for cell_key, val in store.items():
            entry = {"genome": val["genome"], "features": val["features"]}
            if val.get("G") is not None:
                entry["G_bytes"] = pickle.dumps(val["G"])
            out[cell_key] = entry
        torch.save(out, pt_path)
        print(f"saved: {pt_path}  ({len(out)} cells)")

    first_grids = {"archive": grid_a_f, "NSall": grid_n_f, "random": grid_r_f}
    last_grids = {"archive": grid_a_l, "NSall": grid_n_l, "random": grid_r_l}
    _save_keys = save_keys or ["archive", "NSall", "random"]

    if save_path:
        if save_first:
            for key, store in first_grids.items():
                if key in _save_keys:
                    _save_store(store, f"{save_path}_{key}_grid_first.pt")
        if save_last:
            for key, store in last_grids.items():
                if key in _save_keys:
                    _save_store(store, f"{save_path}_{key}_grid_last.pt")

    if plot:
        for key, (store, label) in grids.items():
            fig, ax = plt.subplots(figsize=(7, 6))
            for cx, cy in store.keys():
                is_init = (cx, cy) in cells_init
                ax.add_patch(
                    plt.Rectangle(
                        (cx * cell_w[0], cy * cell_w[1]),
                        cell_w[0],
                        cell_w[1],
                        facecolor="red" if is_init else "steelblue",
                        alpha=0.7 if is_init else 0.5,
                    )
                )
            for i in range(bins + 1):
                ax.axhline(i * cell_w[1], color="lightgray", lw=0.4)
                ax.axvline(i * cell_w[0], color="lightgray", lw=0.4)
            ax.set_xlim(0, scale[0])
            ax.set_ylim(0, scale[1])
            ax.set_xlabel(es.feature_type[0])
            ax.set_ylabel(es.feature_type[1])
            ax.set_title(f"Grid [{label}]  {len(store)} cells")
            plt.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}_{key}.pdf", dpi=150, bbox_inches="tight")
            plt.show()

    return {key: store for key, (store, _) in grids.items()}
'''

