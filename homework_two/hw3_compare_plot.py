#!/usr/bin/env python3
"""
Assignment 3 — Comparison plots for Custom Algorithm, CUBIC, and RENO.

Reads the per-algorithm JSON outputs from tcp_stats_measure.py and produces
three overlay plots (throughput, RTT, loss) with one line per algorithm.

Usage:
  python3 hw3_compare_plot.py
  python3 hw3_compare_plot.py --algo generated_data/algo.json \
                              --cubic generated_data/cubic.json \
                              --reno generated_data/reno.json \
                              -o plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ALGO_STYLES = {
    "algo":  {"color": "#2ecc71", "label": "Custom (algo)"},
    "cubic": {"color": "#3498db", "label": "CUBIC"},
    "reno":  {"color": "#e74c3c", "label": "RENO"},
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


def choose_representative(data: list[dict]) -> dict:
    valid = [e for e in data if e.get("tcp_stats_samples")]
    if not valid:
        raise ValueError("No TCP stats samples found.")

    def score(entry: dict) -> tuple[float, int]:
        samples = entry["tcp_stats_samples"]
        goodput = clean_numeric(s.get("goodput_bps") for s in samples)
        return (float(np.nanmedian(goodput)), len(samples))

    return max(valid, key=score)


def delta_series(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.full_like(values, np.nan)
    prev = np.nan
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        out[i] = 0.0 if np.isnan(prev) else max(v - prev, 0.0)
        prev = v
    return out


def choose_loss_proxy(samples: list[dict]) -> np.ndarray:
    total_retrans = clean_numeric(s.get("total_retrans") for s in samples)
    lost = clean_numeric(s.get("lost") for s in samples)
    retransmits = clean_numeric(s.get("retransmits") for s in samples)

    for series in [delta_series(total_retrans), delta_series(lost), retransmits]:
        if np.isfinite(series).any():
            return series

    return np.zeros(len(samples), dtype=float)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def extract_series(data: list[dict]) -> dict:
    rep = choose_representative(data)
    samples = rep["tcp_stats_samples"]
    return {
        "label": rep["server_label"],
        "t": clean_numeric(s.get("elapsed_sec") for s in samples),
        "throughput_mbps": clean_numeric(s.get("goodput_bps") for s in samples) / 1e6,
        "rtt_ms": clean_numeric(s.get("srtt_us") for s in samples) / 1000,
        "loss": choose_loss_proxy(samples),
    }


def make_plot(
    algo_series: dict[str, dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    for name in ["algo", "cubic", "reno"]:
        if name not in algo_series:
            continue
        s = algo_series[name]
        style = ALGO_STYLES[name]
        ax.plot(
            s["t"], s[metric],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            alpha=0.85,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="600")
    ax.legend(framealpha=0.9)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for Assignment 3 (algo vs CUBIC vs RENO)",
    )
    parser.add_argument(
        "--algo", default="generated_data/algo.json",
        help="JSON data for custom algorithm (default: generated_data/algo.json)",
    )
    parser.add_argument(
        "--cubic", default="generated_data/cubic.json",
        help="JSON data for CUBIC (default: generated_data/cubic.json)",
    )
    parser.add_argument(
        "--reno", default="generated_data/reno.json",
        help="JSON data for RENO (default: generated_data/reno.json)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="plots",
        help="Output directory for PDF plots (default: plots)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sources = {"algo": args.algo, "cubic": args.cubic, "reno": args.reno}
    algo_series: dict[str, dict] = {}

    for name, path in sources.items():
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skipping {name}.", file=sys.stderr)
            continue
        data = load_data(path)
        if not data:
            print(f"Warning: {path} is empty, skipping {name}.", file=sys.stderr)
            continue
        series = extract_series(data)
        algo_series[name] = series
        print(f"Loaded {name}: representative server = {series['label']}")

    if not algo_series:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    make_plot(
        algo_series,
        metric="throughput_mbps",
        ylabel="Goodput (Mbps)",
        title="Throughput Comparison: Custom Algorithm vs CUBIC vs RENO",
        output_path=os.path.join(args.output_dir, "opt1_throughput.pdf"),
    )

    make_plot(
        algo_series,
        metric="rtt_ms",
        ylabel="Smoothed RTT (ms)",
        title="RTT Comparison: Custom Algorithm vs CUBIC vs RENO",
        output_path=os.path.join(args.output_dir, "opt1_rtt.pdf"),
    )

    make_plot(
        algo_series,
        metric="loss",
        ylabel="Loss (delta retransmits per interval)",
        title="Loss Comparison: Custom Algorithm vs CUBIC vs RENO",
        output_path=os.path.join(args.output_dir, "opt1_loss.pdf"),
    )

    print("\nDone. Plots ready for inclusion in the report.")


if __name__ == "__main__":
    main()
