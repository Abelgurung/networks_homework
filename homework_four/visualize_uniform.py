"""Visualize the optimal capacity matrices for uniform hose-model traffic.

For every (n, d) pair drawn from the ``N_VALUES`` x ``D_VALUES`` grid
configured at the top of this file (optionally restricted to d < n) we build
the uniform traffic matrix T_ij = d / (n-1) (zero diagonal), solve the joint
topology-design / multi-commodity-flow MILP from ``topology_design.py`` and
draw the resulting integer link-multiplicity matrix c[u, v] in two ways:

* a directed-graph view with curved arrows whose width / label encodes the
  number of parallel unit-capacity links chosen between two nodes,
* a heat-map of the same matrix.

The full grid is written to ``GRAPH_FIGURE_NAME`` (graph view) and
``HEATMAP_FIGURE_NAME`` (heat-map view) next to this file.

Run with the venv interpreter:

    .\.venv\Scripts\python.exe visualize_uniform.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from topology_design import solve_topology


# ---------------------------------------------------------------------------
# CONFIG -- edit the values below to control the experiment grid.
# ---------------------------------------------------------------------------

# Node counts to sweep (rows of the output figure).
N_VALUES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

# Per-node degree bounds to sweep (columns of the output figure).
D_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

# Only solve cases where d < n.  Set to False to also include d >= n
# (which Gurobi will still solve, though parallel-link saturation makes them
# uninteresting).
REQUIRE_D_LESS_THAN_N: bool = True

# If True, first try to find a *rotationally symmetric* optimum
# (Cayley graph on Z_n: c[u,v] depends only on (v-u) mod n).  When the
# symmetric optimum is strictly worse than the unconstrained optimum we
# print a notification and fall back to the asymmetric solution.
TRY_ROTATIONAL_SYMMETRY: bool = True

# Tolerance for "matches the asymmetric optimum" comparison on lambda.
SYMMETRY_LAMBDA_TOLERANCE: float = 1e-6

# Output filenames (written next to this script).
GRAPH_FIGURE_NAME: str = "uniform_capacity_matrices.png"
HEATMAP_FIGURE_NAME: str = "uniform_capacity_heatmaps.png"

# Pop the matplotlib windows after saving.  Set to False for headless runs.
SHOW_PLOTS: bool = False

# ---------------------------------------------------------------------------


def uniform_traffic(n: int, d: int) -> np.ndarray:
    """Hose-uniform traffic matrix: T_ij = d/(n-1), zero diagonal."""
    T = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T, 0.0)
    return T


def solve_with_symmetry_preference(T: np.ndarray, n: int, d: int) -> dict:
    """Solve the MILP, preferring a rotationally symmetric optimum.

    Strategy
    --------
    1. Solve the unconstrained MILP to obtain the asymmetric optimum
       lambda*_async.
    2. If ``TRY_ROTATIONAL_SYMMETRY`` is True, additionally solve the MILP
       with rotational-symmetry constraints to obtain lambda*_sym.
    3. If lambda*_sym is within ``SYMMETRY_LAMBDA_TOLERANCE`` of
       lambda*_async, return the symmetric solution and tag it as
       ``mode = "symmetric"``.
    4. Otherwise print a notification and return the asymmetric solution
       tagged ``mode = "asymmetric"``.
    """
    res_async = solve_topology(T, d=d)
    lam_async = res_async["lambda"]

    if not TRY_ROTATIONAL_SYMMETRY:
        res_async["mode"] = "asymmetric"
        return res_async

    res_sym = solve_topology(T, d=d, enforce_rotational_symmetry=True)
    lam_sym = res_sym["lambda"]

    if (lam_sym is not None and lam_async is not None
            and lam_sym >= lam_async - SYMMETRY_LAMBDA_TOLERANCE):
        res_sym["mode"] = "symmetric"
        res_sym["lambda_async"] = lam_async
        return res_sym

    if lam_sym is None:
        print(f"  [notice] no rotationally symmetric solution exists for "
              f"n = {n}, d = {d}; falling back to asymmetric.")
    else:
        print(f"  [notice] no rotationally symmetric solution achieves the "
              f"optimum for n = {n}, d = {d}: lambda*_sym = {lam_sym:.6f} < "
              f"lambda*_async = {lam_async:.6f}.  Falling back to asymmetric.")

    res_async["mode"] = "asymmetric"
    res_async["lambda_sym"] = lam_sym
    return res_async


def circular_positions(n: int, radius: float = 1.0) -> dict[int, np.ndarray]:
    """Evenly spaced positions on a unit circle, node 0 at the top."""
    pos = {}
    for i in range(n):
        angle = np.pi / 2 - 2 * np.pi * i / n
        pos[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    return pos


def _title_for(n: int, d: int, lam: float | None, mode: str | None) -> str:
    title = f"n = {n}, d = {d}"
    if lam is not None:
        title += f"   $\\lambda^* = {lam:.4f}$"
    if mode is not None:
        title += f"\n[{mode}]"
    return title


def draw_capacity_graph(ax: plt.Axes, capacity: np.ndarray, n: int, d: int,
                        lam: float | None, mode: str | None = None) -> None:
    """Draw the directed multigraph induced by ``capacity`` on ``ax``."""
    pos = circular_positions(n)

    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    node_radius = 0.16
    for i, p in pos.items():
        ax.add_patch(plt.Circle(p, node_radius, color="#1f77b4",
                                ec="black", zorder=3))
        ax.text(p[0], p[1], str(i), color="white", ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=4)

    max_cap = max(int(capacity.max()), 1)

    for u in range(n):
        for v in range(n):
            if u == v or capacity[u, v] == 0:
                continue
            cap = int(capacity[u, v])
            p_u, p_v = pos[u], pos[v]

            direction = p_v - p_u
            length = np.linalg.norm(direction)
            unit = direction / length
            start = p_u + unit * node_radius
            end = p_v - unit * node_radius

            curvature = 0.18 if capacity[v, u] > 0 else 0.0

            width = 0.8 + 2.2 * (cap / max_cap)

            arrow = FancyArrowPatch(
                start, end,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={curvature}",
                mutation_scale=12 + 4 * cap,
                linewidth=width,
                color="#444",
                zorder=2,
            )
            ax.add_patch(arrow)

            mid = 0.5 * (start + end)
            normal = np.array([-unit[1], unit[0]])
            label_pos = mid + normal * (curvature * length * 0.65 + 0.06)
            ax.text(label_pos[0], label_pos[1], str(cap),
                    fontsize=9, color="#b00",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85),
                    zorder=5)

    ax.set_title(_title_for(n, d, lam, mode), fontsize=11)


def draw_capacity_heatmap(ax: plt.Axes, capacity: np.ndarray, n: int, d: int,
                          lam: float | None, mode: str | None = None) -> None:
    im = ax.imshow(capacity, cmap="Blues", vmin=0, vmax=d)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("v (dst)")
    ax.set_ylabel("u (src)")
    for u in range(n):
        for v in range(n):
            val = capacity[u, v]
            color = "white" if val > d / 2 else "black"
            ax.text(v, u, str(val), ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title(_title_for(n, d, lam, mode), fontsize=11)
    return im


def main() -> None:
    if REQUIRE_D_LESS_THAN_N:
        cases = [(n, d) for n in N_VALUES for d in D_VALUES if d < n]
    else:
        cases = [(n, d) for n in N_VALUES for d in D_VALUES]

    if not cases:
        raise SystemExit(
            f"No (n, d) pairs to solve. N_VALUES={N_VALUES}, D_VALUES={D_VALUES}, "
            f"REQUIRE_D_LESS_THAN_N={REQUIRE_D_LESS_THAN_N}.")

    print(f"Solving uniform-traffic MILP for {len(cases)} cases "
          f"(rotational symmetry preference: {TRY_ROTATIONAL_SYMMETRY}):")
    results = {}
    for n, d in cases:
        T = uniform_traffic(n, d)
        print(f"  n = {n}, d = {d} ... ", end="", flush=True)
        res = solve_with_symmetry_preference(T, n=n, d=d)
        results[(n, d)] = res
        print(f"lambda* = {res['lambda']:.6f}  [{res['mode']}]")

    n_cols = len(D_VALUES)
    n_rows = len(N_VALUES)

    def make_grid(figsize):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                 squeeze=False)
        for r, n in enumerate(N_VALUES):
            for c, d in enumerate(D_VALUES):
                if (n, d) in results:
                    continue
                ax = axes[r][c]
                ax.axis("off")
                reason = "d >= n" if REQUIRE_D_LESS_THAN_N and d >= n else "skipped"
                ax.set_title(f"n = {n}, d = {d}\n({reason})",
                             fontsize=10, color="gray")
        return fig, axes

    out_dir = Path(__file__).resolve().parent

    # Graph view
    fig_g, axes_g = make_grid((4.0 * n_cols, 4.0 * n_rows))
    for (n, d), res in results.items():
        r = N_VALUES.index(n)
        c = D_VALUES.index(d)
        draw_capacity_graph(axes_g[r][c], res["capacity"], n, d,
                            res["lambda"], res.get("mode"))
    fig_g.suptitle("Optimal capacity matrices for uniform traffic "
                   "(arrow label / width = number of parallel unit links)",
                   fontsize=14)
    fig_g.tight_layout(rect=(0, 0, 1, 0.97))
    graph_path = out_dir / GRAPH_FIGURE_NAME
    fig_g.savefig(graph_path, dpi=150)
    print(f"Saved {graph_path}")

    # Heat-map view
    fig_h, axes_h = make_grid((3.4 * n_cols, 3.4 * n_rows))
    for (n, d), res in results.items():
        r = N_VALUES.index(n)
        c = D_VALUES.index(d)
        draw_capacity_heatmap(axes_h[r][c], res["capacity"], n, d,
                              res["lambda"], res.get("mode"))
    fig_h.suptitle("Optimal capacity matrices c[u, v] for uniform traffic",
                   fontsize=14)
    fig_h.tight_layout(rect=(0, 0, 1, 0.97))
    heatmap_path = out_dir / HEATMAP_FIGURE_NAME
    fig_h.savefig(heatmap_path, dpi=150)
    print(f"Saved {heatmap_path}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
