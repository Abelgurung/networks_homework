#!/usr/bin/env python3
"""
TCP Stats Tracing During Transfer (Question 2)

Periodically extracts TCP socket statistics (cwnd, RTT, loss signals, etc.)
alongside application-layer goodput while running iperf3 tests.

Outputs CSV and JSON for analysis.

Usage:
  Single server:
    python3 tcp_stats_measure.py <server_host> [-p PORT] [-t SECS] [-i 0.2]

  Auto-select n random servers:
    python3 tcp_stats_measure.py --auto [-n N] [-t SECS] [-i 0.2]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    read_state,
    send_all,
    send_state,
    TEST_END,
    SERVER_TERMINATE,
    ACCESS_DENIED,
    SERVER_ERROR,
    STATE_NAMES,
    log,
)

_IS_LINUX = platform.system() == "Linux"
_IS_MACOS = platform.system() == "Darwin"
_TCP_CONNECTION_INFO = 0x106

CSV_COLUMNS = [
    "server_label",
    "timestamp",
    "elapsed_sec",
    "interval_sec",
    "goodput_bps",
    "snd_cwnd_bytes",
    "srtt_us",
    "rttvar_us",
    "retransmits",
    "total_retrans",
    "lost",
    "pacing_rate_bps",
    "delivery_rate_bps",
    "bytes_acked",
    "bytes_sent_kernel",
    "bytes_sent_app",
    "snd_ssthresh",
    "snd_mss",
    "min_rtt_us",
]

# Fields returned by get_tcp_stats (subset of CSV_COLUMNS)
_STAT_KEYS = [
    "snd_cwnd_bytes", "srtt_us", "rttvar_us", "retransmits",
    "total_retrans", "lost", "pacing_rate_bps", "delivery_rate_bps",
    "bytes_acked", "bytes_sent_kernel", "snd_ssthresh", "snd_mss",
    "min_rtt_us",
]


# ---------------------------------------------------------------------------
# TCP stats extraction via getsockopt
# ---------------------------------------------------------------------------
#
# Linux  TCP_INFO offsets (uapi/linux/tcp.h):
#   retransmits     u8   @  2    current unacked retransmits
#   snd_mss         u32  @ 16
#   lost            u32  @ 32    lost segments
#   rtt (srtt)      u32  @ 68    smoothed RTT in μs
#   rttvar          u32  @ 72    RTT variance in μs
#   snd_ssthresh    u32  @ 76
#   snd_cwnd        u32  @ 80    cwnd in *segments*
#   total_retrans   u32  @100    cumulative retransmit count
#   pacing_rate     u64  @104    bytes/sec
#   bytes_acked     u64  @120
#   min_rtt         u32  @148    μs
#   delivery_rate   u64  @160    bytes/sec
#   bytes_sent      u64  @200
#
# macOS  TCP_CONNECTION_INFO (0x106) offsets (netinet/tcp.h):
#   snd_mss         u32  @ 16    (maxseg)
#   snd_ssthresh    u32  @ 20
#   snd_cwnd        u32  @ 24    in *bytes*
#   snd_sbbytes     u32  @ 32    send-buffer occupancy (bytes)
#   srtt            u32  @ 44    ms
#   rttvar          u32  @ 48    ms
#   txpackets       u64  @ 56    packed
#   txbytes         u64  @ 64    packed
#   txretransmitbytes u64 @ 72   packed  (best loss signal on macOS)
#   rxretransmitpkts  u64 @104   packed
# ---------------------------------------------------------------------------

def get_tcp_stats(sock: socket.socket, app_bytes_written: int) -> dict:
    """Return a dict of TCP-layer statistics from a live socket."""
    stats: dict = {k: None for k in _STAT_KEYS}

    if _IS_LINUX:
        try:
            info = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)
        except OSError:
            return stats

        n = len(info)
        snd_mss = 0
        if n >= 4:
            stats["retransmits"] = struct.unpack_from("B", info, 2)[0]
        if n >= 20:
            snd_mss = struct.unpack_from("I", info, 16)[0]
            stats["snd_mss"] = snd_mss
        if n >= 36:
            stats["lost"] = struct.unpack_from("I", info, 32)[0]
        if n >= 76:
            stats["srtt_us"] = struct.unpack_from("I", info, 68)[0]
            stats["rttvar_us"] = struct.unpack_from("I", info, 72)[0]
        if n >= 84:
            stats["snd_ssthresh"] = struct.unpack_from("I", info, 76)[0]
            cwnd_segs = struct.unpack_from("I", info, 80)[0]
            stats["snd_cwnd_bytes"] = cwnd_segs * snd_mss if snd_mss else cwnd_segs
        if n >= 104:
            stats["total_retrans"] = struct.unpack_from("I", info, 100)[0]
        if n >= 112:
            stats["pacing_rate_bps"] = struct.unpack_from("Q", info, 104)[0] * 8
        if n >= 128:
            stats["bytes_acked"] = struct.unpack_from("Q", info, 120)[0]
        if n >= 152:
            stats["min_rtt_us"] = struct.unpack_from("I", info, 148)[0]
        if n >= 168:
            stats["delivery_rate_bps"] = struct.unpack_from("Q", info, 160)[0] * 8
        if n >= 208:
            stats["bytes_sent_kernel"] = struct.unpack_from("Q", info, 200)[0]
        return stats

    if _IS_MACOS:
        try:
            info = sock.getsockopt(
                socket.IPPROTO_TCP, _TCP_CONNECTION_INFO, 256
            )
        except OSError:
            return stats

        n = len(info)
        if n >= 20:
            stats["snd_mss"] = struct.unpack_from("I", info, 16)[0]
        if n >= 24:
            stats["snd_ssthresh"] = struct.unpack_from("I", info, 20)[0]
        if n >= 28:
            stats["snd_cwnd_bytes"] = struct.unpack_from("I", info, 24)[0]
        if n >= 36:
            snd_sbbytes = struct.unpack_from("I", info, 32)[0]
            stats["bytes_acked"] = max(app_bytes_written - snd_sbbytes, 0)
        if n >= 48:
            stats["srtt_us"] = struct.unpack_from("I", info, 44)[0] * 1000
        if n >= 52:
            stats["rttvar_us"] = struct.unpack_from("I", info, 48)[0] * 1000
        if n >= 80:
            stats["bytes_sent_kernel"] = struct.unpack_from("<Q", info, 64)[0]
            retx_bytes = struct.unpack_from("<Q", info, 72)[0]
            stats["total_retrans"] = retx_bytes
        if n >= 112:
            stats["lost"] = struct.unpack_from("<Q", info, 104)[0]
        return stats

    # Unsupported platform fallback
    stats["bytes_acked"] = app_bytes_written
    return stats


# ---------------------------------------------------------------------------
# TcpStatsClient — iperf3 client with per-interval TCP stat + goodput tracing
# ---------------------------------------------------------------------------

class TcpStatsClient(Iperf3Client):
    """Extends Iperf3Client to sample TCP stats and goodput at each interval."""

    def __init__(self, *args, interval: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval
        self.samples: list[dict] = []

    def _run_test(self) -> None:
        self.test_start_time = time.monotonic()
        self.interval_start_time = self.test_start_time
        self.total_bytes_sent = 0
        self.interval_bytes_sent = 0
        self.samples = []

        n_streams = len(self.data_socks)
        per_stream_bytes = [0] * n_streams

        end_time = self.test_start_time + self.duration
        next_report = self.test_start_time + self.interval

        ctrl_fd = self.ctrl_sock.fileno()
        prev_acked = self._sum_bytes_acked(per_stream_bytes)

        header = (
            f"{'Interval':>15}  {'Transfer':>12}  {'Bitrate':>14}  "
            f"{'Goodput':>14}  {'cwnd':>10}  {'sRTT':>8}  {'Retx':>6}"
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
                prev_acked = self._record_sample(now, per_stream_bytes, prev_acked)
                next_report = now + self.interval

        now = time.monotonic()
        if self.interval_bytes_sent > 0:
            self._record_sample(now, per_stream_bytes, prev_acked)

    def _sum_bytes_acked(self, per_stream_bytes: list[int]) -> int:
        total = 0
        for i, sock in enumerate(self.data_socks):
            st = get_tcp_stats(sock, per_stream_bytes[i])
            acked = st.get("bytes_acked")
            total += acked if acked is not None else per_stream_bytes[i]
        return total

    def _record_sample(
        self,
        now: float,
        per_stream_bytes: list[int],
        prev_acked: int,
    ) -> int:
        elapsed = now - self.interval_start_time
        total_elapsed = now - self.test_start_time

        agg: dict = {k: None for k in _STAT_KEYS}
        total_acked = 0

        for i, sock in enumerate(self.data_socks):
            st = get_tcp_stats(sock, per_stream_bytes[i])
            acked = st.get("bytes_acked")
            total_acked += acked if acked is not None else per_stream_bytes[i]

            if len(self.data_socks) == 1:
                agg = st
            else:
                # Summable counters
                for key in ("bytes_acked", "total_retrans", "lost",
                            "bytes_sent_kernel", "retransmits"):
                    if st.get(key) is not None:
                        agg[key] = (agg.get(key) or 0) + st[key]
                # Sum cwnd across streams
                if st.get("snd_cwnd_bytes") is not None:
                    agg["snd_cwnd_bytes"] = (agg.get("snd_cwnd_bytes") or 0) + st["snd_cwnd_bytes"]
                # Take first non-None for per-connection metrics
                for key in ("srtt_us", "rttvar_us", "min_rtt_us",
                            "pacing_rate_bps", "delivery_rate_bps",
                            "snd_ssthresh", "snd_mss"):
                    if st.get(key) is not None and agg.get(key) is None:
                        agg[key] = st[key]

        delta_acked = max(total_acked - prev_acked, 0)
        goodput_bps = (delta_acked * 8 / elapsed) if elapsed > 0 else 0.0

        sample = {
            "timestamp": time.time(),
            "elapsed_sec": round(total_elapsed, 4),
            "interval_sec": round(elapsed, 4),
            "goodput_bps": goodput_bps,
            "bytes_sent_app": self.total_bytes_sent,
        }
        sample.update(agg)
        self.samples.append(sample)

        cwnd_s = _fmt_bytes(agg.get("snd_cwnd_bytes"))
        srtt_s = f"{agg['srtt_us'] / 1000:.1f}ms" if agg.get("srtt_us") else "n/a"
        retx_s = str(agg.get("total_retrans") if agg.get("total_retrans") is not None else "n/a")
        bitrate = (self.interval_bytes_sent * 8 / elapsed) if elapsed > 0 else 0
        print(
            f"{total_elapsed - elapsed:6.2f}-{total_elapsed:<6.2f} sec  "
            f"{format_bytes(self.interval_bytes_sent):>12}  "
            f"{format_bits(bitrate):>14}  "
            f"{format_bits(goodput_bps):>14}  "
            f"{cwnd_s:>10}  "
            f"{srtt_s:>8}  "
            f"{retx_s:>6}"
        )

        self.interval_bytes_sent = 0
        self.interval_start_time = now
        return total_acked


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Multi-destination runner
# ---------------------------------------------------------------------------

def run_tcp_stats_tests(
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

        print(f"\n{'=' * 80}")
        print(
            f"[{len(results) + 1}/{n}] Testing {label}  "
            f"(provider: {srv['provider']})"
        )
        print(f"{'=' * 80}")
        sys.stdout.flush()

        client = TcpStatsClient(
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
            summary["tcp_stats_samples"] = client.samples
            results.append(summary)
            print(
                f"\n  => OK: {format_bytes(summary['bytes_sent'])} in "
                f"{summary['duration']:.2f}s "
                f"({format_bits(summary['bitrate_bps'])}), "
                f"{len(client.samples)} TCP stat samples"
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
# Output serialization
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            label = r["server_label"]
            for sample in r["tcp_stats_samples"]:
                row = dict(sample)
                row["server_label"] = label
                writer.writerow(row)


def save_json(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    output = []
    for r in results:
        output.append({
            "server_label": r["server_label"],
            "server_host": r["server_host"],
            "server_port": r["server_port"],
            "bytes_sent": r["bytes_sent"],
            "duration": r["duration"],
            "bitrate_bps": r["bitrate_bps"],
            "platform": platform.system(),
            "tcp_stats_samples": r["tcp_stats_samples"],
        })
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCP stats tracing + goodput measurement using iperf3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 tcp_stats_measure.py speedtest.wtnet.de -p 5200 -t 10\n"
            "  python3 tcp_stats_measure.py --auto -n 5 -t 30 -i 0.2\n"
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
        help="Sampling interval in seconds (default 0.2)",
    )
    parser.add_argument("--verbose", "-V", action="store_true")
    parser.add_argument(
        "--output", "-o", type=str, default="generated_data/tcp_stats.csv",
        help="Output CSV file (default generated_data/tcp_stats.csv)",
    )
    parser.add_argument(
        "--json-output", type=str, default="generated_data/tcp_stats.json",
        help="Output JSON file (default generated_data/tcp_stats.json)",
    )

    args = parser.parse_args()

    if args.auto:
        results = run_tcp_stats_tests(
            n=args.num_servers,
            duration=args.duration,
            blksize=args.blocksize,
            num_streams=args.streams,
            interval=args.interval,
            verbose=args.verbose,
        )
    elif args.server:
        client = TcpStatsClient(
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
        summary["tcp_stats_samples"] = client.samples
        results = [summary]
    else:
        parser.error("Provide a server hostname, or use --auto mode.")

    os.makedirs("generated_data", exist_ok=True)

    save_csv(results, args.output)
    save_json(results, args.json_output)

    total_samples = sum(len(r["tcp_stats_samples"]) for r in results)
    print(f"\nTCP stats saved to {args.output} (CSV) and {args.json_output} (JSON)")
    print(f"Total samples: {total_samples}")
    print_final_table(results)


if __name__ == "__main__":
    main()
