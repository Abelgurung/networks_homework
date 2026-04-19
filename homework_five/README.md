# AllGather Benchmarks with PyTorch Gloo

This project implements and benchmarks three **AllGather** algorithms using **PyTorch Distributed** with the **Gloo** backend:

- **Ring AllGather**
- **Recursive Doubling AllGather**
- **Swing AllGather**

## Files

- `allgather_worker.py` — algorithm implementations and single-case benchmark runner
- `run_all.py` — Python driver that runs all benchmark cases and writes `allgather_results.json`
- `generate_plots.py` — reads the JSON results and generates the two plots
- `run.sh` — tiny wrapper that creates a virtualenv, installs requirements, sets the local Gloo env vars, and runs `run_all.py`
- `requirements.txt` — Python dependencies

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run everything:

```bash
./run.sh
```

This will:

1. create `.venv`
2. install the Python requirements
3. run all benchmark cases
4. write `allgather_bench/allgather_results.json`
5. generate:
   - `allgather_bench/allgather_vs_message_size.png`
   - `allgather_bench/allgather_vs_world_size.png`

## Why the launcher is configured this way

On some machines, especially macOS, `torchrun --standalone` may try to use a bad IPv6 hostname and hang during rendezvous. This bundle avoids that by explicitly using local IPv4 loopback:

- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=29500`
- `GLOO_SOCKET_IFNAME=lo0` on macOS, or `lo` on Linux

Those defaults are set automatically in `run.sh` and `run_all.py`.

## Run the Python driver directly

If your virtualenv is already set up, you can skip `run.sh` and run:

```bash
source .venv/bin/activate
python run_all.py
```

You can also override settings:

```bash
python run_all.py \
  --iters 5 \
  --world-sizes 2 4 8 \
  --message-sizes-bytes 1024 65536 1048576 4194304 \
  --fixed-msg-bytes 1048576
```

You can override the rendezvous settings too:

```bash
python run_all.py --master-port 29600 --gloo-ifname lo0
```

## Run a single benchmark manually

Example: ring AllGather, 8 ranks, 1 MB per rank:

```bash
source .venv/bin/activate
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=lo0   # use lo on Linux

torchrun \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  allgather_worker.py \
  --algo ring \
  --msg-bytes 1048576 \
  --iters 3 \
  --result-file ring_8_1mb.json
```

Supported values for `--algo`:

- `ring`
- `recursive_doubling`
- `swing`

## Output format

Each result JSON object looks like:

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

## Notes

- Recursive doubling and swing assume **power-of-two world sizes** in this benchmark setup.
- Message size is **per-rank payload size**.
- The default sweep uses:
  - message sizes: `1 KB`, `64 KB`, `1 MB`, `4 MB`
  - world sizes: `2`, `4`, `8`
- If port `29500` is already in use, rerun with a different port, for example:

```bash
MASTER_PORT=29600 ./run.sh
```
