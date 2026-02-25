# Homework Two — Application-Layer Goodput Measurement

Measure TCP goodput during iperf3 tests by polling `getsockopt(TCP_INFO)` at regular intervals. Results are saved as JSON and can be plotted.

## Prerequisites

- Python 3
- iperf3 server(s) reachable on the network
- Dependencies: `numpy`, `matplotlib` (for plotting)

## Quick Start

**1. Generate measurement data**

```bash
python goodput_measure.py --auto -n 5 -t 30 -i 0.2
```

This runs tests against 5 random public servers for 30 seconds each, sampling goodput every 0.2 seconds. Output is written to `generated_data/goodput_data.json`.

**2. Generate the plot**

```bash
python goodput_plot.py
```

Produces `plots/goodput_plot.png`.

## Options

| Flag | Description |
|------|-------------|
| `--auto` | Auto-select random public iperf3 servers (required for quick start) |
| `-n N` | Number of servers to test (default: 10) |
| `-t SECS` | Duration per server in seconds (default: 10) |
| `-i SECS` | Goodput sampling interval (default: 0.2) |
| `-o FILE` | Output JSON path (default: `generated_data/goodput_data.json`) |

**Single server:** `python goodput_measure.py <hostname> -p 5201 -t 10`

**Plot options:** `python goodput_plot.py [input.json] -o plots/output.png`
