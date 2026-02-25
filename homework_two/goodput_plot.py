from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

# Modern, distinctive color palette (avoids generic matplotlib look)
PALETTE = [
    "#1f77b4", "#e74c3c", "#2ecc71", "#9b59b6",
    "#f39c12", "#3498db", "#e91e63", "#00bcd4",
]


def load_data(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot goodput time series from goodput_measure.py output",
    )
    parser.add_argument(
        "input", nargs="?", default="generated_data/goodput_data.json",
        help="JSON data file (default goodput_data.json)",
    )
    parser.add_argument(
        "-o", "--output", default="plots/goodput_plot.png",
        help="Output image file (default goodput_plot.png)",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    if not data:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    # Clean, modern style (try seaborn style, fall back to default)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    fig, (ax_plot, ax_table) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Application-Layer Goodput (bytes ACK'd via getsockopt)",
        fontsize=15, fontweight="600", y=0.98,
    )
    summary_rows: list[list[str]] = []

    for i, entry in enumerate(data):
        samples = entry.get("goodput_samples", [])
        if not samples:
            continue

        label = entry["server_label"]
        if len(label) > 50:
            label = label[:47] + "..."

        t_vals = [s["t"] for s in samples]
        gp_mbps = [s["goodput_bps"] / 1e6 for s in samples]

        color = PALETTE[i % len(PALETTE)]
        ax_plot.plot(
            t_vals, gp_mbps,
            label=label, color=color, linewidth=2.0, alpha=0.9,
            solid_capstyle="round",
        )

        gp_arr = np.array([s["goodput_bps"] for s in samples])
        summary_rows.append([
            label,
            f"{np.min(gp_arr) / 1e6:.2f}",
            f"{np.median(gp_arr) / 1e6:.2f}",
            f"{np.mean(gp_arr) / 1e6:.2f}",
            f"{np.percentile(gp_arr, 95) / 1e6:.2f}",
        ])

    ax_plot.set_facecolor("white")
    ax_plot.set_xlabel("Time (s)", fontsize=11, fontweight="500")
    ax_plot.set_ylabel("Goodput (Mbps)", fontsize=11, fontweight="500")
    ax_plot.legend(
        fontsize=8, loc="upper right", framealpha=0.95,
        edgecolor="#e0e0e0", fancybox=True,
    )
    ax_plot.grid(True, alpha=0.4, linestyle="--")
    ax_plot.set_xlim(left=0)
    ax_plot.set_ylim(bottom=0)
    ax_plot.tick_params(axis="both", labelsize=9)
    for spine in ax_plot.spines.values():
        spine.set_color("#cccccc")

    # ---- Summary table ------------------------------------------------
    ax_table.axis("off")
    ax_table.set_facecolor("white")
    col_labels = [
        "Destination", "Min (Mbps)", "Median (Mbps)",
        "Avg (Mbps)", "P95 (Mbps)",
    ]

    if summary_rows:
        table = ax_table.table(
            cellText=summary_rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#e0e0e0")
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="600", fontsize=9)
            else:
                cell.set_facecolor("white")
            if col == 0:
                cell.set_text_props(ha="left")
                cell.set_width(0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        args.output, dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    print(f"Plot saved to {args.output}")

    # Also print the summary to stdout
    print(
        f"\n{'Destination':<50} "
        f"{'Min':>10} {'Median':>10} {'Avg':>10} {'P95':>10}"
    )
    print("-" * 95)
    for row in summary_rows:
        print(
            f"{row[0]:<50} "
            f"{row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}"
        )
    print()


if __name__ == "__main__":
    main()
