# AllGather Benchmarks with PyTorch Gloo

This project implements and benchmarks three **AllGather** algorithms using **PyTorch Distributed** with the **Gloo** backend:

- **Ring AllGather**
- **Recursive Doubling AllGather**
- **Swing AllGather**

The code includes:

- `allgather_worker.py` — the algorithm implementations and a single-run benchmark entrypoint
- `bench_session_fork.py` — a Python benchmark harness for running multiple jobs inside one process group per world size
- `run_sweep.sh` — a convenience script that runs the full benchmark sweep and generates plots
- `allgather_bench/` — output directory for JSON results and plots

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

> Note: `torchrun` is installed as part of PyTorch.

## Files

```text
.
├── allgather_worker.py
├── bench_session_fork.py
├── run_sweep.sh
├── requirements.txt
└── allgather_bench/
    ├── allgather_results.json
    ├── allgather_vs_message_size.png
    └── allgather_vs_world_size.png
```

## Setup

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run a Single Benchmark

Use `torchrun` to launch one AllGather benchmark.

Example: **Ring AllGather**, **8 ranks**, **1 MB per rank**, **3 timed iterations**:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 allgather_worker.py \
  --algo ring \
  --msg-bytes 1048576 \
  --iters 3 \
  --result-file result_ring_8_1mb.json
```

### Supported algorithms

- `ring`
- `recursive_doubling`
- `swing`

### Arguments

- `--algo`: algorithm name
- `--msg-bytes`: message size **per rank** in bytes
- `--iters`: number of timed iterations
- `--result-file`: JSON file written by rank 0

### Example runs

Ring:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 allgather_worker.py \
  --algo ring --msg-bytes 65536 --iters 3 --result-file ring_4_64kb.json
```

Recursive Doubling:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 allgather_worker.py \
  --algo recursive_doubling --msg-bytes 1048576 --iters 3 --result-file rd_8_1mb.json
```

Swing:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 allgather_worker.py \
  --algo swing --msg-bytes 1048576 --iters 3 --result-file swing_8_1mb.json
```

## Run the Full Sweep

The provided `run_sweep.sh` script runs the benchmark sweep used for the included plots:

- **Message-size sweep** at **world size = 8**
  - 1 KB, 64 KB, 1 MB, 4 MB per rank
- **World-size sweep** at **message size = 1 MB per rank**
  - 2, 4, 8 ranks

Run it with:

```bash
chmod +x run_sweep.sh
./run_sweep.sh
```

This writes:

- `allgather_bench/allgather_results.json`
- `allgather_bench/allgather_vs_message_size.png`
- `allgather_bench/allgather_vs_world_size.png`

## Run the Python Harness Directly

`bench_session_fork.py` can run multiple jobs for a fixed world size in one shot.

Example for **world size 8**:

```bash
python bench_session_fork.py 8 \
  '[
    {"algorithm": "ring", "msg_bytes_per_rank": 1024},
    {"algorithm": "ring", "msg_bytes_per_rank": 65536},
    {"algorithm": "recursive_doubling", "msg_bytes_per_rank": 1048576},
    {"algorithm": "swing", "msg_bytes_per_rank": 1048576}
  ]' \
  session8.json
```

## Output Format

Each benchmark result is stored as JSON and looks like this:

```json
{
  "algorithm": "ring",
  "world_size": 8,
  "msg_bytes_per_rank": 1048576,
  "samples_ms": [192.95, 191.34, 193.02],
  "median_ms": 192.95,
  "min_ms": 191.34,
  "max_ms": 193.02
}
```

## Correctness Checking

Each run verifies that every gathered block contains the expected byte pattern for each rank. If the implementation is incorrect, the script raises an error instead of silently writing bad timings.

## Notes and Limitations

- **Recursive Doubling** and **Swing** are implemented for **power-of-two world sizes**.
- Message sizes are **per-rank payload sizes**, not total aggregate sizes.
- Larger message sizes may use substantial memory because each rank stores the full gathered output.
- `bench_session_fork.py` uses multiprocessing with `fork`, which is typically best on Linux/macOS. If you want the most portable path, prefer `torchrun` with `allgather_worker.py`.
- All timings are reported in **milliseconds** and use the **maximum rank time** per iteration via `all_reduce`, which is usually the right completion-time metric for collectives.

## Suggested Experiments

To extend the benchmarks, try:

- more message sizes, for example `1KB` to `16MB` or `32MB`
- more timed iterations, for example `--iters 5` or `--iters 10`
- more world sizes, such as `2, 4, 8, 16` if your machine has enough CPU and memory
- comparing against PyTorch's built-in `dist.all_gather` as a baseline
