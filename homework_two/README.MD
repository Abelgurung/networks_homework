# Homework Two

Measure TCP goodput and trace TCP socket statistics during iperf3 tests by polling `getsockopt(TCP_INFO / TCP_CONNECTION_INFO)` at regular intervals. Results are saved as JSON/CSV and can be plotted.

## Prerequisites

- Python 3
- iperf3 server(s) reachable on the network
- Dependencies: `numpy`, `matplotlib` (for plotting)

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

Produces `plots/goodput_plot.png`.

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
