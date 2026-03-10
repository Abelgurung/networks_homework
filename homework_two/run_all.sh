#!/bin/bash
set -e

NUM_SERVERS="${NUM_SERVERS:-10}"
DURATION="${DURATION:-60}"
INTERVAL="${INTERVAL:-0.2}"

echo "============================================"
echo " Assignment 2 — Full Experiment Pipeline"
echo " Servers: $NUM_SERVERS  Duration: ${DURATION}s  Interval: ${INTERVAL}s"
echo "============================================"

mkdir -p generated_data plots

# ── Q1: Application-Layer Goodput ──────────────────────────────────
echo ""
echo "=== Q1: Goodput Measurement ==="
python3 goodput_measure.py --auto \
    -n "$NUM_SERVERS" -t "$DURATION" -i "$INTERVAL" \
    -o generated_data/goodput_data.json

echo ""
echo "=== Q1: Goodput Plotting ==="
python3 goodput_plot.py generated_data/goodput_data.json -o plots/goodput_plot.pdf

# ── Q2: TCP Stats Tracing ─────────────────────────────────────────
echo ""
echo "=== Q2: TCP Stats Measurement ==="
python3 tcp_stats_measure.py --auto \
    -n "$NUM_SERVERS" -t "$DURATION" -i "$INTERVAL" \
    -o generated_data/tcp_stats.csv \
    --json-output generated_data/tcp_stats.json

echo ""
echo "=== Q2: TCP Stats Plotting ==="
python3 tcp_stats_plot.py generated_data/tcp_stats.json -o plots/tcp_stats_plots.pdf

# ── Q3: ML Model Training & Prediction ────────────────────────────
echo ""
echo "=== Q3: Compiling ML Model ==="
g++ -O2 -std=c++17 -o ml_model ML_model/ml_model.cpp
mkdir -p generated_data/predictions

echo ""
echo "=== Q3: Training ML Model ==="
./ml_model generated_data/tcp_stats.csv generated_data/predictions

echo ""
echo "=== Q3: Plotting ML Predictions ==="
python3 ml_plot.py --input-dir generated_data/predictions --output-dir plots

echo ""
echo "============================================"
echo " Pipeline complete."
echo " Data   → generated_data/"
echo " Plots  → plots/"
echo "============================================"
