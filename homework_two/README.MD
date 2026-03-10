# Homework Two

Measure TCP goodput, trace TCP socket statistics, and train an ML model for congestion window prediction during iperf3 tests. Uses `getsockopt(TCP_INFO / TCP_CONNECTION_INFO)` to extract kernel-level stats at regular intervals. Results are saved as JSON/CSV and plotted as PDFs.

## Quick Start (Docker)

```bash
cd networks_homework/homework_two
docker build -t cs536-hw2 .
docker run --rm cs536-hw2
```

To capture output on the host:

```bash
docker run --rm \
  -v "$(pwd)/output_plots:/app/plots" \
  -v "$(pwd)/output_data:/app/generated_data" \
  cs536-hw2
```

Environment variables: `NUM_SERVERS` (default 10), `DURATION` (default 60), `INTERVAL` (default 0.2).

## Prerequisites (without Docker)

- Python 3 with `numpy`, `matplotlib` (`pip install -r requirements.txt`)
- `g++` with C++17 support (for the ML model)
- Linux or WSL2 (required for `TCP_INFO`)

## Run Everything

```bash
bash run_all.sh
```

Runs Q1 goodput measurement and plotting, Q2 TCP stats measurement and plotting, and Q3 ML model training and prediction plotting. Output goes to `generated_data/` and `plots/`.

---

## Q1 — Application-Layer Goodput Measurement

**Generate measurement data**

```bash
python goodput_measure.py --auto -n 5 -t 30 -i 0.2
```

Runs tests against 5 random public servers for 30 s each, sampling goodput every 0.2 s. Output: `generated_data/goodput_data.json`.

**Generate the plot**

```bash
python goodput_plot.py
```

Produces `plots/goodput_plot.pdf` (or `.png` if run without `-o`).

| Flag | Description |
|------|-------------|
| `--auto` | Auto-select random public iperf3 servers |
| `-n N` | Number of servers to test (default: 10) |
| `-t SECS` | Duration per server in seconds (default: 10) |
| `-i SECS` | Goodput sampling interval (default: 0.2) |
| `-o FILE` | Output JSON path (default: `generated_data/goodput_data.json`) |

**Single server:** `python goodput_measure.py <hostname> -p 5201 -t 10`

**Plot options:** `python goodput_plot.py [input.json] -o plots/output.png`

---

## Q2 — TCP Stats Tracing During Transfer

Extracts TCP socket statistics at the same sampling interval as goodput. Per sample it records: `snd_cwnd`, `srtt`, loss signal (retransmits), goodput, `rttvar`, pacing rate, bytes acked/sent, delivery rate, min RTT, and more.

Supports both Linux (`TCP_INFO`) and macOS (`TCP_CONNECTION_INFO`).

**Generate TCP stats data**

```bash
python tcp_stats_measure.py --auto -n 5 -t 30 -i 0.2
```

Outputs:
- `generated_data/tcp_stats.csv` — flat CSV with one row per sample
- `generated_data/tcp_stats.json` — structured JSON grouped by server

**Generate Q2(b) PDF plots**

```bash
python tcp_stats_plot.py
```

Produces two PDF files for one representative destination chosen automatically from the JSON data:
- `plots/tcp_stats_plots_timeseries.pdf`
- `plots/tcp_stats_plots_scatter.pdf`

**Plot options:** `python tcp_stats_plot.py [input.json] -o plots/output.pdf --server-label "partial label"`

**Single server:**

```bash
python tcp_stats_measure.py <hostname> -p 5201 -t 10 -i 0.2
```

| Flag | Description |
|------|-------------|
| `--auto` | Auto-select random public iperf3 servers |
| `-n N` | Number of servers to test (default: 10) |
| `-t SECS` | Duration per server in seconds (default: 10) |
| `-i SECS` | Sampling interval (default: 0.2) |
| `-o FILE` | Output CSV path (default: `generated_data/tcp_stats.csv`) |
| `--json-output FILE` | Output JSON path (default: `generated_data/tcp_stats.json`) |

### CSV Columns

| Column | Description |
|--------|-------------|
| `server_label` | Server identifier |
| `timestamp` | Unix epoch timestamp |
| `elapsed_sec` | Seconds since test start |
| `interval_sec` | Duration of this sample interval |
| `goodput_bps` | Application-layer goodput (bits/sec) |
| `snd_cwnd_bytes` | Congestion window (bytes) |
| `srtt_us` | Smoothed RTT (μs) |
| `rttvar_us` | RTT variance (μs) |
| `retransmits` | Current retransmit count (Linux) / retransmit bytes (macOS) |
| `total_retrans` | Cumulative retransmits (count on Linux, bytes on macOS) |
| `lost` | Lost segments (Linux) / rx retransmit packets (macOS) |
| `pacing_rate_bps` | Pacing rate (bits/sec, Linux only) |
| `delivery_rate_bps` | Delivery rate estimate (bits/sec, Linux only) |
| `bytes_acked` | Cumulative bytes acknowledged |
| `bytes_sent_kernel` | Kernel-reported bytes sent |
| `bytes_sent_app` | Application-layer bytes written |
| `snd_ssthresh` | Slow-start threshold |
| `snd_mss` | Maximum segment size |
| `min_rtt_us` | Minimum observed RTT (μs, Linux only) |

---

## Q3 — ML Model + Congestion Window Prediction

A C++ linear regression model (`ML_model/ml_model.cpp`) trains on the TCP traces from Q2 to predict congestion window updates. A Python script (`ml_plot.py`) plots the predictions against actual cwnd.

**Compile and train**

```bash
g++ -O2 -std=c++17 -o ml_model ML_model/ml_model.cpp
mkdir -p generated_data/predictions
./ml_model generated_data/tcp_stats.csv generated_data/predictions
```

Reads `generated_data/tcp_stats.csv`, trains a per-destination linear model with a 5-step sliding window of (goodput, RTT, loss) features, and outputs prediction CSVs and weight files to `generated_data/predictions/`.

**Plot predictions**

```bash
python ml_plot.py --input-dir generated_data/predictions --output-dir plots
```

Produces per-destination PDFs (e.g. `plots/cwnd_195_181_162_195_5201_(Miami,_US).pdf`) showing actual cwnd (train + test) and predicted cwnd from the test split.

A standalone congestion window algorithm using the learned weights is in `ML_model/cwnd_algo.cpp`. See `ML_model/README.MD` for the mathematical formulation.

---

## Congestion Control Algorithm (`--cc` flag)

The iperf3 client supports selecting a TCP congestion control algorithm:

```bash
python iperf3_client.py <server> -p 5201 -t 10 --cc algo
```

This calls `setsockopt(TCP_CONGESTION)` on all sockets. Requires the algorithm to be loaded in the kernel (e.g. via `insmod`). See `homework3/README.md` for instructions on building and loading the custom kernel module.

---

## Project Structure

```
homework_two/
├── iperf3_client.py          # Pure-Python iperf3 TCP client
├── goodput_measure.py        # Q1: goodput measurement via getsockopt
├── goodput_plot.py           # Q1: goodput time series + summary table
├── tcp_stats_measure.py      # Q2: TCP stats extraction (cwnd, RTT, loss, ...)
├── tcp_stats_plot.py         # Q2: time-series and scatter plot generation
├── ml_plot.py                # Q3: cwnd prediction plotting
├── ML_model/
│   ├── ml_model.cpp          # Q3: linear regression model (C++)
│   ├── cwnd_algo.cpp         # Q3: standalone cwnd algorithm using learned weights
│   ├── README.MD             # Q3: mathematical formulation
│   ├── weights.txt           # Learned weights (one trace)
│   └── miami_US_weights.txt  # Learned weights (Miami trace)
├── homework3/                # Assignment 3 kernel module (do not modify)
│   ├── algo.c
│   ├── Makefile
│   └── README.md
├── run_all.sh                # Full pipeline (Q1 + Q2 + Q3)
├── Dockerfile                # Docker container (ubuntu:24.04)
├── requirements.txt          # Python dependencies
├── generated_data/           # Output data (JSON, CSV)
└── plots/                    # Output plots (PDF)
```
