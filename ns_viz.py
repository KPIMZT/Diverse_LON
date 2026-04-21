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
    cond_palette = ["#0072B2", "#E69F00", "#009E73"]

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

