#!/usr/bin/env python3
"""
Plot ML model cwnd predictions (Q3).

Reads per-destination prediction CSVs produced by ml_model and the raw
tcp_stats.json trace data, then generates comparison PDFs showing the
full actual snd_cwnd timeseries (train + test) alongside the predicted
cwnd starting at the test split.

Usage:
  python3 ml_plot.py [--input-dir DIR] [--tcp-stats FILE] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 5
TRAIN_RATIO = 0.8


def sanitize_label(label: str) -> str:
    """Apply the same sanitisation as ml_model.cpp to produce a filename stem."""
    s = label.replace('"', "")
    s = s.replace(" ", "_").replace(".", "_").replace(":", "_")
    return s


def load_tcp_stats(path: str) -> dict[str, list[float]]:
    """Load tcp_stats.json and return {sanitized_label: [cwnd_bytes, ...]}."""
    with open(path) as f:
        data = json.load(f)

    traces: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    for entry in data:
        server_label = entry.get("server_label", "")
        samples = entry.get("tcp_stats_samples", [])
        if not samples:
            continue
        cwnd_series = [
            float(s.get("snd_cwnd_bytes") or 0) for s in samples
        ]
        key = sanitize_label(server_label)
        traces[key] = cwnd_series
        labels[key] = server_label
    return traces, labels


def load_prediction_csv(path: str):
    """Load a prediction CSV.  Returns (times, actual, predicted).

    Predicted entries may be None for training-phase rows.
    """
    times: list[int] = []
    actual: list[float] = []
    predicted: list[float | None] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(int(row["Time"]))
            actual.append(float(row["Actual_CWND"]))
            pred_str = row.get("Predicted_CWND", "").strip()
            predicted.append(float(pred_str) if pred_str else None)

    return times, actual, predicted


def build_full_trace(
    raw_cwnd: list[float],
    pred_times: list[int],
    pred_actual: list[float],
    pred_predicted: list[float | None],
):
    """Combine raw trace cwnd with prediction data.

    The prediction CSV uses *windowed* indices (0 = first sample with a
    full 5-step history, i.e. raw trace index WINDOW_SIZE).  We convert
    everything to raw trace indices so the plot includes all data points
    from the very beginning of the trace.

    Returns (train_t, train_cwnd, test_t, test_cwnd, test_pred, split_raw_idx).
    All cwnd values are in raw bytes.
    """
    n_raw = len(raw_cwnd)
    n_windowed = n_raw - WINDOW_SIZE
    if n_windowed <= 0:
        return None
    split_idx = int(n_windowed * TRAIN_RATIO)

    # Training portion: raw indices 0 .. WINDOW_SIZE + split_idx - 1
    # This includes the initial WINDOW_SIZE points that the ML model
    # cannot predict on (no history yet) plus the windowed training set.
    train_end = WINDOW_SIZE + split_idx
    train_t = list(range(train_end))
    train_cwnd = [raw_cwnd[i] for i in train_t]

    # Test portion from prediction CSV.
    # Convert windowed indices to raw indices: raw = WINDOW_SIZE + windowed
    has_predictions = any(p is not None for p in pred_predicted)
    if has_predictions:
        test_t = [WINDOW_SIZE + t for t, p in zip(pred_times, pred_predicted) if p is not None]
        test_cwnd = [a for a, p in zip(pred_actual, pred_predicted) if p is not None]
        test_pred = [p for p in pred_predicted if p is not None]
    else:
        test_t = [WINDOW_SIZE + t for t in pred_times]
        test_cwnd = list(pred_actual)
        test_pred = [float(p) for p in pred_predicted]

    return train_t, train_cwnd, test_t, test_cwnd, test_pred, train_end


def plot_cwnd(
    train_t, train_cwnd,
    test_t, test_cwnd, test_pred,
    split_idx: int,
    title: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        train_t, train_cwnd,
        color="#1f77b4", linewidth=1.5, solid_capstyle="round",
        label="Actual (Train)",
    )
    ax.plot(
        test_t, test_cwnd,
        color="#ff7f0e", linewidth=1.5, solid_capstyle="round",
        label="Actual (Test)",
    )
    ax.plot(
        test_t, test_pred,
        color="#2ca02c", linewidth=1.5, linestyle="--",
        label="Predicted (Test)",
    )
    ax.axvline(
        x=split_idx - 0.5, color="#d62728", linestyle="--", alpha=0.8,
        label="Train/Test split",
    )

    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("snd_cwnd (bytes)", fontsize=11)
    ax.set_title(f"CWND Time Series \u2014 {title}", fontsize=13, fontweight="600")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.35, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_prediction_only(
    times, actual, predicted,
    title: str,
    output_path: str,
) -> None:
    """Fallback when no raw trace data is available."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    t = np.array(times, dtype=float)
    act = np.array(actual)
    pred_arr = np.array([float(p) for p in predicted])

    ax.plot(t, act, color="#ff7f0e", linewidth=1.5, label="Actual (Test)")
    ax.plot(t, pred_arr, color="#2ca02c", linewidth=1.5, linestyle="--", label="Predicted (Test)")

    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("snd_cwnd (bytes)", fontsize=11)
    ax.set_title(f"CWND Time Series \u2014 {title}", fontsize=13, fontweight="600")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.35, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ML cwnd predictions for Assignment 2 Q3",
    )
    parser.add_argument(
        "--input-dir", default="generated_data/predictions",
        help="Directory containing prediction CSVs (default: generated_data/predictions)",
    )
    parser.add_argument(
        "--tcp-stats", default="generated_data/tcp_stats.json",
        help="Path to tcp_stats.json for full trace data (default: generated_data/tcp_stats.json)",
    )
    parser.add_argument(
        "--output-dir", default="plots",
        help="Output directory for PDF plots (default: plots)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join("ML_model", "*.csv")))
    if not csv_files:
        print("No prediction CSVs found.", file=sys.stderr)
        sys.exit(1)

    # Load raw trace data for training portion
    raw_traces: dict[str, list[float]] = {}
    raw_labels: dict[str, str] = {}
    if os.path.isfile(args.tcp_stats):
        raw_traces, raw_labels = load_tcp_stats(args.tcp_stats)
        print(f"Loaded {len(raw_traces)} raw traces from {args.tcp_stats}")
    else:
        print(f"Warning: {args.tcp_stats} not found; plots will lack training data.")

    print(f"Plotting {len(csv_files)} prediction traces...")
    for csv_path in csv_files:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(args.output_dir, f"cwnd_{basename}.pdf")

        times, actual, predicted = load_prediction_csv(csv_path)
        if len(times) < 2:
            print(f"  Skipping {basename}: too few data points")
            continue

        # Try to find matching raw trace for training data
        matched_key = None
        for key in raw_traces:
            if key == basename or basename in key or key in basename:
                matched_key = key
                break

        display_title = raw_labels.get(matched_key, basename.replace("_", " "))

        if matched_key and matched_key in raw_traces:
            result = build_full_trace(
                raw_traces[matched_key], times, actual, predicted,
            )
            if result is not None:
                train_t, train_cwnd, test_t, test_cwnd, test_pred, split_idx = result
                plot_cwnd(
                    train_t, train_cwnd,
                    test_t, test_cwnd, test_pred,
                    split_idx, display_title, output_path,
                )
                continue

        # Fallback: prediction CSV has train data embedded (new format)
        has_train = any(p is None for p in predicted)
        if has_train:
            train_t = [t for t, p in zip(times, predicted) if p is None]
            train_cwnd = [a for a, p in zip(actual, predicted) if p is None]
            test_t = [t for t, p in zip(times, predicted) if p is not None]
            test_cwnd = [a for a, p in zip(actual, predicted) if p is not None]
            test_pred = [p for p in predicted if p is not None]
            split_idx = max(train_t) + 1 if train_t else times[0]
            plot_cwnd(
                train_t, train_cwnd,
                test_t, test_cwnd, test_pred,
                split_idx, display_title, output_path,
            )
        else:
            plot_prediction_only(times, actual, predicted, display_title, output_path)


if __name__ == "__main__":
    main()
