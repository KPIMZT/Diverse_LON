"""
lon_viz.py
==============
LON visualization on [0,1]^2 design space (D=2 only).
Covers both isotropic (fixed-center) and genome-based LON construction.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import torch
from msg_landscape import MSGLandscapeIso


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_overlap(
    pos: dict,
    min_dist: float = 0.025,
    max_iter: int = 50,
) -> dict:
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)
    N = len(nodes)
    for _ in range(max_iter):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                diff = coords[i] - coords[j]
                d = np.linalg.norm(diff) + 1e-12
                if d < min_dist:
                    push = (min_dist - d) / 2.0 * diff / d
                    coords[i] += push
                    coords[j] -= push
                    moved = True
        coords = coords.clip(0.0, 1.0)
        if not moved:
            break
    return {n: (float(coords[k, 0]), float(coords[k, 1])) for k, n in enumerate(nodes)}


def _get_node_pos(G: nx.DiGraph) -> dict:
    pos = {}
    for i in G.nodes:
        nd = G.nodes[i]
        if "x" in nd and "y" in nd:
            pos[i] = (nd["x"], nd["y"])
        elif "pos" in nd and len(nd["pos"]) >= 2:
            pos[i] = (nd["pos"][0], nd["pos"][1])
        else:
            pos[i] = (0.5, 0.5)
    return pos


def _contour_on_ax(
    ax: plt.Axes,
    msg: MSGLandscapeIso,
    resolution: int = 500,
    cmap: str = "viridis",
    alpha_fill: float = 0.55,
) -> None:
    X_np, Y_np, Z_np = msg.grid_eval(resolution)
    Z_np = np.clip(Z_np, 0.0, 1.0)
    ax.contourf(X_np, Y_np, Z_np, cmap=cmap, alpha=alpha_fill, vmax=1.0)
    ax.contour(X_np, Y_np, Z_np, colors="white", linewidths=0.4, alpha=0.6)


def _draw_lon_on_ax(
    ax: plt.Axes,
    G: nx.DiGraph,
    orig_pos: dict,
    disp_pos: dict,
    r: float,
    node_size: float,
    sink_size: float,
) -> None:
    sink_nodes = {i for i in G.nodes if G.out_degree(i) == 0}

    all_nodes = list(G.nodes)
    all_fit = np.array([G.nodes[i]["fitness"] for i in all_nodes], dtype=float)
    if len(all_fit) == 1:
        ranks = np.array([1.0])
    else:
        ranks = np.argsort(np.argsort(all_fit)).astype(float)
        ranks /= (len(ranks) - 1)
    rank_map = {i: ranks[k] for k, i in enumerate(all_nodes)}


    for i in G.nodes:
        ox, oy = orig_pos[i]
        dx, dy = disp_pos[i]
        if abs(ox - dx) > 1e-6 or abs(oy - dy) > 1e-6:
            ax.plot(
                [ox, dx],
                [oy, dy],
                color="white",
                linewidth=0.5,
                alpha=0.4,
                linestyle="--",
                zorder=2,
            )


    node_r = np.sqrt(node_size) / 1000.0
    for u, v, data in G.edges(data=True):
        x0, y0 = disp_pos[u]
        x1, y1 = disp_pos[v]
        prob = data.get("prob", 0.0)
        imputed = data.get("imputed", False)
        ddx, ddy = x1 - x0, y1 - y0
        dist = np.hypot(ddx, ddy) + 1e-12
        ux, uy = ddx / dist, ddy / dist
        xs, ys = x0 + ux * node_r, y0 + uy * node_r
        xe, ye = x1 - ux * node_r, y1 - uy * node_r
        if imputed:
            ax.annotate(
                "",
                xy=(xe, ye),
                xytext=(xs, ys),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="red",
                    lw=0.7,
                    linestyle="dashed",
                    mutation_scale=10,
                ),
            )
        else:
            ax.annotate(
                "",
                xy=(xe, ye),
                xytext=(xs, ys),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="white",
                    lw=1.5 + 3.0 * prob,
                    mutation_scale=12,
                ),
            )


    normal_nodes = [i for i in G.nodes if i not in sink_nodes]
    s_nodes = list(sink_nodes)
    if normal_nodes:
        ax.scatter(
            [disp_pos[i][0] for i in normal_nodes],
            [disp_pos[i][1] for i in normal_nodes],
            c=[rank_map[i] for i in normal_nodes],
            cmap="cool",
            vmin=0.0,
            vmax=1.0,
            s=node_size,
            zorder=5,
            marker="o",
            edgecolors="white",
            linewidths=1,
        )
    if s_nodes:
        ax.scatter(
            [disp_pos[i][0] for i in s_nodes],
            [disp_pos[i][1] for i in s_nodes],
            c=[rank_map[i] for i in s_nodes],
            cmap="cool",
            vmin=0.0,
            vmax=1.0,
            s=sink_size,
            zorder=6,
            marker="*",
            edgecolors="white",
            linewidths=1,
        )


# ── main visualization: isotropic (fixed centers) ───────────────────────────
def visualize_design_space_LON(
    G: nx.DiGraph,
    genome: torch.Tensor,
    means: torch.Tensor,
    M: int,
    r: float,
    device: str,
    seed: Optional[int],
    resolution: int = 500,
    cmap: str = "viridis",
    node_size: float = 1100,
    sink_size: float = 1100,
    min_node_dist: float = 0.025,
    figsize: Tuple[float, float] = (14.0, 13.0),
    visualize_3d_flag: bool = False,
    z_scale: float = 1.0,
    save_path: Optional[str] = None,

):

    genome = genome.to(device)
    means = means.to(device)


    msg = MSGLandscapeIso(genome, means, M)

    if visualize_3d_flag:
        visualize_3d(msg, resolution=300, cmap=cmap, z_scale=z_scale)

    orig_pos = _get_node_pos(G)
    disp_pos = _resolve_overlap(orig_pos, min_dist=min_node_dist)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 23,
                          "mathtext.fontset": "stix"})

    fig, ax = plt.subplots(figsize=figsize)

    _contour_on_ax(ax, msg, resolution=resolution, cmap=cmap, alpha_fill=0.55)

    means_np = means.cpu().numpy()
    ax.scatter(
        means_np[:, 0],
        means_np[:, 1],
        s=8,
        color="white",
        alpha=0.2,
        zorder=2,
    )

    _draw_lon_on_ax(ax, G, orig_pos, disp_pos, r, node_size, sink_size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("$x_1$", fontsize=75, fontfamily="Times New Roman")
    ax.set_ylabel("$x_2$", fontsize=75, fontfamily="Times New Roman")
    plt.tick_params(labelsize=62)
    ax.tick_params(axis="x", pad=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    plt.show()




def visualize_3d(
    msg: MSGLandscapeIso,
    G: Optional[nx.DiGraph] = None,
    title: str = None,
    resolution: int = 100,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (9.0, 7.0),
    elev: float = 30.0,
    azim: float = -60.0,
    z_scale: float = 1.0,
    node_z_offset: float = 0.05,
    save_path: Optional[str] = None,
) -> None:

    Xg, Yg, Z_np = msg.grid_eval(resolution)
    Z_np = Z_np * z_scale

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Xg, Yg, Z_np, cmap=cmap, edgecolor="none", alpha=0.85)

    if G is not None:
        sink_nodes   = {i for i in G.nodes if G.out_degree(i) == 0}
        global_sink  = max(sink_nodes, key=lambda i: G.nodes[i]["fitness"]) if sink_nodes else None
        normal_sinks = [i for i in sink_nodes if i != global_sink]
        local_optima = [i for i in G.nodes if i not in sink_nodes]

        def _pos(i):
            nd = G.nodes[i]
            return nd["pos"][0], nd["pos"][1], nd["fitness"] * z_scale + node_z_offset


        all_nodes = local_optima + normal_sinks + ([global_sink] if global_sink is not None else [])
        all_fit = np.array([G.nodes[i]["fitness"] for i in all_nodes], dtype=float)
        if len(all_fit) == 1:
            ranks = np.array([1.0])
        else:
            ranks = np.argsort(np.argsort(all_fit)).astype(float)
            ranks /= (len(ranks) - 1)
        rank_map = {i: ranks[k] for k, i in enumerate(all_nodes)}


        if local_optima:
            xs, ys, zs = zip(*[_pos(i) for i in local_optima])
            rs = [rank_map[i] for i in local_optima]
            ax.scatter(xs, ys, zs, c=rs, cmap="cool", vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidths=0.5,
                       s=90, marker="o", depthshade=False, label="Node")


        if normal_sinks:
            xs, ys, zs = zip(*[_pos(i) for i in normal_sinks])
            rs = [rank_map[i] for i in normal_sinks]
            ax.scatter(xs, ys, zs, c=rs, cmap="cool", vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidths=0.8,
                       s=220, marker="*", depthshade=False, label="Sink")


        if global_sink is not None:
            x, y, z = _pos(global_sink)
            ax.scatter([x], [y], [z], c=[rank_map[global_sink]], cmap="cool", vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidths=0.8,
                       s=220, marker="*", depthshade=False, zorder=100, label="Global")

    else:
        lo, lf, go, gf = msg.find_optima()
        all_fit = np.concatenate([lf.cpu().numpy(), gf.cpu().numpy()])
        if len(all_fit) == 1:
            ranks = np.array([1.0])
        else:
            ranks = np.argsort(np.argsort(all_fit)).astype(float)
            ranks /= (len(ranks) - 1)
        lo_ranks = ranks[:len(lf)]
        go_ranks = ranks[len(lf):]

        if len(lo) > 0:
            ax.scatter(lo[:, 0].cpu(), lo[:, 1].cpu(), lf.cpu() * z_scale + node_z_offset,
                       c=lo_ranks, cmap="cool", vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidths=0.8,
                       s=220, marker="*", depthshade=False, label="Sink")
        if len(go) > 0:
            ax.scatter(go[:, 0].cpu(), go[:, 1].cpu(), gf.cpu() * z_scale + node_z_offset,
                       c=go_ranks, cmap="cool", vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidths=0.8,
                       s=220, marker="*", depthshade=False, label="Global")

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    plt.show()


