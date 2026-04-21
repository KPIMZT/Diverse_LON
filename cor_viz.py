"""
cor_viz.py
============
Visualize results of benchmark or correlation analysis.
"""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import os
import seaborn as sns
def parse_cell_key(key: str):
    return ast.literal_eval(key)


def build_grid(df: pd.DataFrame, value_col: str, bins: int = None):

    coords = df["cell_key"].apply(parse_cell_key)
    df = df.copy()
    df["_x"] = coords.apply(lambda c: c[0])
    df["_y"] = coords.apply(lambda c: c[1])

    x_max = int(df["_x"].max())
    y_max = int(df["_y"].max())
    nx = bins if bins is not None else x_max + 1
    ny = bins if bins is not None else y_max + 1

    x_vals = list(range(nx))
    y_vals = list(range(ny))

    grid = np.full((ny, nx), np.nan)
    for _, row in df.iterrows():
        xi = int(row["_x"])
        yi = int(row["_y"])
        grid[yi, xi] = row[value_col]

    return grid, x_vals, y_vals




def plot_feature_grid(
    bench_path:   str,
    feature_path: str,
    feature_x:    str,
    feature_y:    str,
    alg_name:     str,
    metric:       str = "success_rate",
    n_bins:       int = 10,
    cmap:         str = "coolwarm",
    figsize:      tuple = (8, 7),
    show_values:  bool = True,
    save_path:    Optional[str] = None,
):

    bench_df   = pd.read_csv(bench_path)
    feature_df = pd.read_csv(feature_path)
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    col = f"{alg_name}_{metric}"
    # feature_df may already contain col (when feature_path is the merged CSV)
    if col in feature_df.columns:
        df = feature_df
    else:
        df = feature_df.merge(bench_df[["problem_id", col]], on="problem_id")

    z_vals = df[col].values.astype(float)


    coords = df["cell_key"].apply(parse_cell_key)
    x_idx = np.array([c[0] for c in coords], dtype=int)
    y_idx = np.array([c[1] for c in coords], dtype=int)

    x_edges = np.linspace(0, 1, n_bins + 1)
    y_edges = np.linspace(0, 1, n_bins + 1)

    grid = np.full((n_bins, n_bins), np.nan)
    for xi in range(n_bins):
        for yi in range(n_bins):
            mask = (x_idx == xi) & (y_idx == yi)
            if mask.any():
                grid[xi, yi] = float(np.median(z_vals[mask]))

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    def _mathtt(s):
        return r"$\mathtt{" + s.replace("_", r"\_") + r"}$"

    plt.colorbar(im, ax=ax, label=_mathtt(metric), fraction=0.046, pad=0.04)

    if show_values:
        for xi in range(n_bins):
            for yi in range(n_bins):
                if not np.isnan(grid[xi, yi]):
                    val = grid[xi, yi]
                    text_color = "black" if 0.3 < val < 0.8 else "white"
                    ax.text(x_centers[xi], y_centers[yi], f"{val:.2f}",
                            ha="center", va="center", fontsize=7, color=text_color)

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(_mathtt(feature_x), fontsize=36)
    ax.set_ylabel(_mathtt(feature_y), fontsize=36)
    #ax.set_title(f"{alg_name} / {metric}", fontsize=18)
    ax.tick_params(labelsize=30)
    ax.tick_params(axis="x", pad=15)
    im.colorbar.ax.tick_params(labelsize=29)
    im.colorbar.set_label(_mathtt(metric), fontsize=36)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()





def plot_cor_alldim(
        result_dir: str,
        dim_seeds:  dict[int, int],
        alg_name:   str | list[str],
        save:       bool = None,
        metrics:    list[str] | None = None,
):
    if isinstance(alg_name, str):
        alg_name = [alg_name]

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0, font="Times New Roman")
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })
    save_dir = "./results_cor_reg"
    os.makedirs(save_dir, exist_ok=True)

    dims       = sorted(dim_seeds.keys())
    dim_labels = [f"$d={d}$" for d in dims]

    dfs = {}
    for alg in alg_name:
        dfs[alg] = {}
        for d, label in zip(dims, dim_labels):
            seed = dim_seeds[d]
            path = os.path.join(result_dir, f"dim{d}_seed{seed}_{alg}_spearman.csv")
            if not os.path.exists(path):
                print(f"[warn] not found: {path}")
                dfs[alg][label] = pd.DataFrame()
                continue
            df = pd.read_csv(path)
            df = df.set_index("metric").drop(columns=["algorithm"], errors="ignore")
            dfs[alg][label] = df

    features = list(dfs[alg_name[0]][dim_labels[0]].columns)
    feature_labels = [r"$\mathtt{" + ("num_sinks" if f == "num_sink" else f).replace("_", r"\_") + r"}$" for f in features]
    all_metrics = list(dfs[alg_name[0]][dim_labels[0]].index)
    metrics = [m for m in metrics if m in all_metrics] if metrics is not None else all_metrics

    x = np.arange(len(features))
    n_dims = len(dim_labels)
    dim_palette = ["#0072B2", "#009E73","#E69F00"]
    width = 0.8 / n_dims


    rows = [
        (metric, alg)
        for metric in metrics
        for alg in alg_name
    ]
    n_rows = len(rows)
    n_cols = 2
    n_plot_rows = (n_rows + 1) // 2

    fig, axes_2d = plt.subplots(n_plot_rows, n_cols,
                                figsize=(max(20, len(features) * 1.8), 5 * n_plot_rows),
                                sharex="col")
    axes_2d = np.atleast_2d(axes_2d)
    axes = list(axes_2d.flatten())
    for ax in axes[n_rows:]:
        ax.set_visible(False)

    for ax, (metric, alg) in zip(axes, rows):
        for di, dim in enumerate(dim_labels):
            if dfs[alg][dim].empty or metric not in dfs[alg][dim].index:
                continue
            corrs = dfs[alg][dim].loc[metric, features].values.astype(float)
            offset = (di - (n_dims - 1) / 2) * width
            ax.bar(x + offset, corrs,
                   width=width,
                   color=dim_palette[di],
                   edgecolor="black",
                   linewidth=0.6,
                   label=dim)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel(r"$\rho$ ($\mathtt{" + metric.replace("_", r"\_") + r"}$)", fontsize=27)
        ax.set_title(alg, fontsize=21)
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(axis="y", labelsize=22)
        sns.despine(ax=ax)

    for ax in axes_2d[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=26)


    handles, lbls = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        seen.setdefault(l, h)
    axes[0].legend(seen.values(), seen.keys(), loc="lower right", fontsize=22)

    title_algs = " / ".join(alg_name)
    plt.tight_layout()

    if save:
        fname = "_".join(alg_name)
        plt.savefig(f"{save_dir}/corplot_alg{fname}.pdf", dpi=150, bbox_inches="tight")
        plt.savefig(f"{save_dir}/corplot_alg{fname}.svg", dpi=150, bbox_inches="tight")
        print(f"saved: {save_dir}/corplot_alg{fname}.pdf")
    else:
        plt.show()
