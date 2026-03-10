from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "cwnd": "#1f77b4",
    "rtt": "#e74c3c",
    "loss": "#9b59b6",
    "goodput": "#2ecc71",
    "scatter": "#3498db",
}


def load_data(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def clean_numeric(values: Iterable[object]) -> np.ndarray:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            cleaned.append(np.nan)
        else:
            cleaned.append(float(value))
    return np.array(cleaned, dtype=float)


def choose_representative(data: list[dict], server_label: str | None) -> dict:
    valid_entries = [entry for entry in data if entry.get("tcp_stats_samples")]
    if not valid_entries:
        raise ValueError("No TCP stats samples found in the input file.")

    if server_label:
        for entry in valid_entries:
            label = entry.get("server_label", "")
            if label == server_label or server_label.lower() in label.lower():
                return entry
        raise ValueError(f"No destination matched '{server_label}'.")

    def score(entry: dict) -> tuple[float, int]:
        samples = entry.get("tcp_stats_samples", [])
        goodput = clean_numeric(sample.get("goodput_bps") for sample in samples)
        return (float(np.nanmedian(goodput)), len(samples))

    return max(valid_entries, key=score)


def delta_series(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values

    out = np.full_like(values, np.nan)
    prev = np.nan
    for idx, value in enumerate(values):
        if np.isnan(value):
            continue
        if np.isnan(prev):
            out[idx] = 0.0
        else:
            out[idx] = max(value - prev, 0.0)
        prev = value
    return out


def choose_loss_proxy(samples: list[dict]) -> tuple[np.ndarray, str]:
    total_retrans = clean_numeric(sample.get("total_retrans") for sample in samples)
    lost = clean_numeric(sample.get("lost") for sample in samples)
    retransmits = clean_numeric(sample.get("retransmits") for sample in samples)

    candidates = [
        (delta_series(total_retrans), "Loss proxy: delta total retrans"),
        (delta_series(lost), "Loss proxy: delta lost"),
        (retransmits, "Loss proxy: retransmits"),
    ]

    for series, label in candidates:
        finite = series[np.isfinite(series)]
        if finite.size:
            return series, label

    return np.zeros(len(samples), dtype=float), "Loss proxy"


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def split_output_paths(output: str) -> tuple[str, str]:
    root, ext = os.path.splitext(output)
    if not ext:
        ext = ".pdf"
    return (f"{root}_timeseries{ext}", f"{root}_scatter{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TCP stats plots for tracing results",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="generated_data/tcp_stats.json",
        help="Input JSON file from tcp_stats_measure.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plots/tcp_stats_plots.pdf",
        help="Base output PDF path (default: plots/tcp_stats_plots.pdf)",
    )
    parser.add_argument(
        "--server-label",
        default=None,
        help="Exact or partial server label to plot; default picks a representative destination",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    if not data:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    selected = choose_representative(data, args.server_label)
    samples = selected["tcp_stats_samples"]
    label = selected["server_label"]

    t = clean_numeric(sample.get("elapsed_sec") for sample in samples)
    goodput_mbps = clean_numeric(sample.get("goodput_bps") for sample in samples) / 1e6
    cwnd_kib = clean_numeric(sample.get("snd_cwnd_bytes") for sample in samples) / 1024
    rtt_ms = clean_numeric(sample.get("srtt_us") for sample in samples) / 1000
    loss_proxy, loss_label = choose_loss_proxy(samples)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    timeseries_output, scatter_output = split_output_paths(args.output)
    os.makedirs(os.path.dirname(timeseries_output) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"TCP Stats Time Series\n{label}", fontsize=15, fontweight="600")

    axes[0, 0].plot(t, cwnd_kib, color=PALETTE["cwnd"], linewidth=2.0)
    axes[0, 0].set_title("snd_cwnd")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Congestion Window (KiB)")
    style_axes(axes[0, 0])

    axes[0, 1].plot(t, rtt_ms, color=PALETTE["rtt"], linewidth=2.0)
    axes[0, 1].set_title("RTT")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("SRTT (ms)")
    style_axes(axes[0, 1])

    axes[1, 0].plot(t, loss_proxy, color=PALETTE["loss"], linewidth=2.0)
    axes[1, 0].set_title("Loss Proxy")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel(loss_label)
    style_axes(axes[1, 0])

    axes[1, 1].plot(t, goodput_mbps, color=PALETTE["goodput"], linewidth=2.0)
    axes[1, 1].set_title("Throughput")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Goodput (Mbps)")
    style_axes(axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(timeseries_output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"TCP Stats Relationships\n{label}", fontsize=15, fontweight="600")

    axes[0].scatter(cwnd_kib, goodput_mbps, color=PALETTE["scatter"], alpha=0.8, s=28)
    axes[0].set_title("snd_cwnd vs Goodput")
    axes[0].set_xlabel("Congestion Window (KiB)")
    axes[0].set_ylabel("Goodput (Mbps)")
    style_axes(axes[0])

    axes[1].scatter(rtt_ms, goodput_mbps, color=PALETTE["scatter"], alpha=0.8, s=28)
    axes[1].set_title("RTT vs Goodput")
    axes[1].set_xlabel("SRTT (ms)")
    axes[1].set_ylabel("Goodput (Mbps)")
    style_axes(axes[1])

    axes[2].scatter(loss_proxy, goodput_mbps, color=PALETTE["scatter"], alpha=0.8, s=28)
    axes[2].set_title("Loss Proxy vs Goodput")
    axes[2].set_xlabel(loss_label)
    axes[2].set_ylabel("Goodput (Mbps)")
    style_axes(axes[2])

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(scatter_output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Representative destination: {label}")
    print(f"Time-series plot saved to {timeseries_output}")
    print(f"Scatter plot saved to {scatter_output}")


if __name__ == "__main__":
    main()
