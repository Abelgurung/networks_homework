#!/usr/bin/env python3
"""
Measure application-layer goodput during iperf3 tests by polling
getsockopt(TCP_INFO / TCP_CONNECTION_INFO) at regular intervals to
extract cumulative bytes acknowledged by the TCP stack.

Outputs a JSON file consumed by goodput_plot.py.

Usage:
  Single server:
    python3 goodput_measure.py <server_host> [-p PORT] [-t SECS] [-i 0.2]

  Auto-select n random servers from the public list:
    python3 goodput_measure.py --auto [-n N] [-t SECS] [-i 0.2]
"""

from __future__ import annotations

import argparse
import json
import platform
import select as _select
import socket
import struct
import sys
import time

from iperf3_client import (
    Iperf3Client,
    DEFAULT_PORT,
    DEFAULT_DURATION,
    DEFAULT_TCP_BLKSIZE,
    fetch_server_list,
    select_random_servers,
    format_bytes,
    format_bits,
    print_final_table,
    send_state,
    read_state,
    send_all,
    TEST_END,
    SERVER_TERMINATE,
    ACCESS_DENIED,
    SERVER_ERROR,
    STATE_NAMES,
    log,
)

# ---------------------------------------------------------------------------
# TCP stats extraction via getsockopt
# ---------------------------------------------------------------------------

_IS_LINUX = platform.system() == "Linux"
_IS_MACOS = platform.system() == "Darwin"

# macOS exposes a richer struct through TCP_CONNECTION_INFO (0x106)
_TCP_CONNECTION_INFO = 0x106


def get_bytes_acked(sock: socket.socket, app_bytes_written: int) -> int:
    """
    Return cumulative bytes ACK'd for *sock* using kernel TCP stats.

    Linux  - TCP_INFO  → tcpi_bytes_acked  (u64 @ offset 120)
    macOS  - TCP_CONNECTION_INFO → app_bytes_written − tcpi_snd_sbbytes
             (send-buffer bytes remaining, u32 @ offset 32)
    Other  - falls back to app_bytes_written (upper bound).
    """
    if _IS_LINUX:
        try:
            info = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)
            if len(info) >= 128:
                return struct.unpack_from("Q", info, 120)[0]
        except OSError:
            pass
        return app_bytes_written

    if _IS_MACOS:
        try:
            info = sock.getsockopt(
                socket.IPPROTO_TCP, _TCP_CONNECTION_INFO, 256
            )
            snd_sbbytes = struct.unpack_from("I", info, 32)[0]
            return max(app_bytes_written - snd_sbbytes, 0)
        except OSError:
            pass
        return app_bytes_written

    return app_bytes_written


# ---------------------------------------------------------------------------
# GoodputClient — iperf3 client with per-interval goodput recording
# ---------------------------------------------------------------------------

class GoodputClient(Iperf3Client):
    """Extends Iperf3Client to sample goodput via getsockopt at each interval."""

    def __init__(self, *args, interval: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval
        self.goodput_samples: list[dict] = []

    # Override the data-sending loop to insert getsockopt sampling.
    def _run_test(self) -> None:
        self.test_start_time = time.monotonic()
        self.interval_start_time = self.test_start_time
        self.total_bytes_sent = 0
        self.interval_bytes_sent = 0
        self.goodput_samples = []

        n_streams = len(self.data_socks)
        per_stream_bytes = [0] * n_streams

        end_time = self.test_start_time + self.duration
        next_report = self.test_start_time + self.interval

        ctrl_fd = self.ctrl_sock.fileno()

        prev_acked = sum(
            get_bytes_acked(self.data_socks[i], 0) for i in range(n_streams)
        )

        header = (
            f"[ ID] {'Interval':>15}  {'Transfer':>12}  "
            f"{'Bitrate':>16}  {'Goodput':>16}"
        )
        print(header)
        print("-" * len(header))

        while True:
            now = time.monotonic()
            if now >= end_time:
                break

            ready_r, _, _ = _select.select([ctrl_fd], [], [], 0)
            if ready_r:
                state = read_state(self.ctrl_sock)
                name = STATE_NAMES.get(state, str(state))
                if state == SERVER_TERMINATE:
                    self._log(f"Server terminated (state={name}).")
                    return
                if state == ACCESS_DENIED:
                    raise RuntimeError("Server denied access (busy).")
                if state in (SERVER_ERROR,):
                    raise RuntimeError(f"Server error (state={name}).")

            for i, sock in enumerate(self.data_socks):
                try:
                    send_all(sock, self.data_buf)
                    self.total_bytes_sent += self.blksize
                    self.interval_bytes_sent += self.blksize
                    per_stream_bytes[i] += self.blksize
                except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                    self._log(f"Data stream error: {exc}")
                    return

            now = time.monotonic()
            if now >= next_report:
                prev_acked = self._record_interval(
                    now, per_stream_bytes, prev_acked
                )
                next_report = now + self.interval

        now = time.monotonic()
        if self.interval_bytes_sent > 0:
            self._record_interval(now, per_stream_bytes, prev_acked)

    def _record_interval(
        self,
        now: float,
        per_stream_bytes: list[int],
        prev_acked: int,
    ) -> int:
        elapsed = now - self.interval_start_time
        total_elapsed = now - self.test_start_time

        cur_acked = sum(
            get_bytes_acked(self.data_socks[i], per_stream_bytes[i])
            for i in range(len(self.data_socks))
        )
        delta_acked = max(cur_acked - prev_acked, 0)

        if elapsed > 0:
            bitrate = (self.interval_bytes_sent * 8) / elapsed
            goodput_bps = (delta_acked * 8) / elapsed
        else:
            bitrate = goodput_bps = 0.0

        self.goodput_samples.append({
            "t": round(total_elapsed, 4),
            "goodput_bps": goodput_bps,
            "bytes_acked": delta_acked,
            "interval_sec": round(elapsed, 4),
        })

        print(
            f"[  0] {total_elapsed - elapsed:6.2f}-{total_elapsed:<6.2f} sec  "
            f"{format_bytes(self.interval_bytes_sent):>12}  "
            f"{format_bits(bitrate):>16}  "
            f"{format_bits(goodput_bps):>16}"
        )

        self.interval_bytes_sent = 0
        self.interval_start_time = now
        return cur_acked


# ---------------------------------------------------------------------------
# Multi-destination runner
# ---------------------------------------------------------------------------

def run_goodput_tests(
    n: int,
    duration: int,
    blksize: int,
    num_streams: int,
    interval: float,
    verbose: bool,
) -> list[dict]:
    full_list = fetch_server_list(verbose)
    pool = select_random_servers(full_list, len(full_list), verbose)

    results: list[dict] = []
    idx = 0

    while len(results) < n and idx < len(pool):
        srv = pool[idx]
        idx += 1

        host, port = srv["host"], srv["port"]
        label = f"{host}:{port}"
        if srv["site"]:
            label += f" ({srv['site']}, {srv['country']})"

        print(f"\n{'=' * 76}")
        print(
            f"[{len(results)+1}/{n}] Testing {label}  "
            f"(provider: {srv['provider']})"
        )
        print(f"{'=' * 76}")
        sys.stdout.flush()

        client = GoodputClient(
            server=host,
            port=port,
            duration=duration,
            blksize=blksize,
            num_streams=num_streams,
            verbose=verbose,
            interval=interval,
        )

        try:
            summary = client.run()
            summary["server_host"] = host
            summary["server_port"] = port
            summary["server_label"] = label
            summary["goodput_samples"] = client.goodput_samples
            results.append(summary)
            print(
                f"\n  => OK: {format_bytes(summary['bytes_sent'])} in "
                f"{summary['duration']:.2f}s "
                f"({format_bits(summary['bitrate_bps'])})"
            )
        except (ConnectionError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"  => SKIPPED ({exc})")
            log(f"Skipping {label}: {exc}", verbose)

    if len(results) < n:
        print(
            f"\nWarning: only completed {len(results)}/{n} tests "
            f"(exhausted {idx} servers from the pool)."
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Goodput measurement using iperf3 + getsockopt(TCP_INFO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 goodput_measure.py speedtest.wtnet.de -p 5200 -t 10\n"
            "  python3 goodput_measure.py --auto -n 5 -t 30 -i 0.2\n"
        ),
    )
    parser.add_argument(
        "server", nargs="?", default=None,
        help="iperf3 server hostname or IP (omit when using --auto)",
    )
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select random public servers")
    parser.add_argument("--num-servers", "-n", type=int, default=10,
                        help="Servers to test in --auto mode (default 10)")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", "-t", type=int, default=DEFAULT_DURATION,
                        help="Test duration per server in seconds")
    parser.add_argument("--blocksize", "-l", type=int,
                        default=DEFAULT_TCP_BLKSIZE)
    parser.add_argument("--streams", "-P", type=int, default=1)
    parser.add_argument(
        "--interval", "-i", type=float, default=0.2,
        help="Goodput sampling interval in seconds (default 0.2)",
    )
    parser.add_argument("--verbose", "-V", action="store_true")
    parser.add_argument(
        "--output", "-o", type=str, default="generated_data/goodput_data.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    if args.auto:
        results = run_goodput_tests(
            n=args.num_servers,
            duration=args.duration,
            blksize=args.blocksize,
            num_streams=args.streams,
            interval=args.interval,
            verbose=args.verbose,
        )
    elif args.server:
        client = GoodputClient(
            server=args.server,
            port=args.port,
            duration=args.duration,
            blksize=args.blocksize,
            num_streams=args.streams,
            verbose=args.verbose,
            interval=args.interval,
        )
        print(f"Connecting to host {args.server}, port {args.port}")
        sys.stdout.flush()
        try:
            summary = client.run()
        except (ConnectionError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            sys.exit(1)

        summary["server_host"] = args.server
        summary["server_port"] = args.port
        summary["server_label"] = f"{args.server}:{args.port}"
        summary["goodput_samples"] = client.goodput_samples
        results = [summary]
    else:
        parser.error("Provide a server hostname, or use --auto mode.")

    output = []
    for r in results:
        output.append({
            "server_label": r["server_label"],
            "server_host": r["server_host"],
            "server_port": r["server_port"],
            "bytes_sent": r["bytes_sent"],
            "duration": r["duration"],
            "bitrate_bps": r["bitrate_bps"],
            "goodput_samples": r["goodput_samples"],
        })

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nGoodput data saved to {args.output}")
    print_final_table(results)


if __name__ == "__main__":
    main()
