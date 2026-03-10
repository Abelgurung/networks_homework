#!/usr/bin/env python3
"""
Modes:
  Single server:
    python3 iperf3_client.py <server_host> [--port PORT] [--duration SECS] ...

  Auto-select n random servers from the public list:
    python3 iperf3_client.py --auto [--num-servers N] [--duration SECS] ...
"""

from __future__ import annotations

import argparse
import json
import random
import select
import socket
import string
import struct
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# iperf3 protocol state constants (from src/iperf_api.h)
# ---------------------------------------------------------------------------
TEST_START      =  1
TEST_RUNNING    =  2
TEST_END        =  4
PARAM_EXCHANGE  =  9
CREATE_STREAMS  = 10
SERVER_TERMINATE = 11
CLIENT_TERMINATE = 12
EXCHANGE_RESULTS = 13
DISPLAY_RESULTS  = 14
IPERF_START     = 15
IPERF_DONE      = 16
ACCESS_DENIED   = -1
SERVER_ERROR    = -2

COOKIE_SIZE = 37  # 36 random chars + NUL

DEFAULT_TCP_BLKSIZE = 128 * 1024  # 128 KiB
DEFAULT_DURATION = 10
DEFAULT_PORT = 5201
CONNECT_TIMEOUT = 10
CTRL_READ_TIMEOUT = 30

STATE_NAMES = {
    TEST_START:      "TEST_START",
    TEST_RUNNING:    "TEST_RUNNING",
    TEST_END:        "TEST_END",
    PARAM_EXCHANGE:  "PARAM_EXCHANGE",
    CREATE_STREAMS:  "CREATE_STREAMS",
    SERVER_TERMINATE: "SERVER_TERMINATE",
    CLIENT_TERMINATE: "CLIENT_TERMINATE",
    EXCHANGE_RESULTS: "EXCHANGE_RESULTS",
    DISPLAY_RESULTS:  "DISPLAY_RESULTS",
    IPERF_START:     "IPERF_START",
    IPERF_DONE:      "IPERF_DONE",
    ACCESS_DENIED:   "ACCESS_DENIED",
    SERVER_ERROR:    "SERVER_ERROR",
}


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[iperf3-client] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def make_cookie() -> bytes:
    """Generate a 37-byte iperf3 cookie (36 random chars + NUL)."""
    chars = string.ascii_lowercase + "234567"
    cookie_str = "".join(random.choice(chars) for _ in range(COOKIE_SIZE - 1))
    return cookie_str.encode("ascii") + b"\x00"


def send_all(sock: socket.socket, data: bytes) -> None:
    """Send all bytes, retrying on partial writes."""
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError("Socket connection broken during send")
        total_sent += sent


def recv_exact(sock: socket.socket, nbytes: int, timeout: float = CTRL_READ_TIMEOUT) -> bytes:
    """Receive exactly *nbytes* from *sock*, or raise on timeout / EOF."""
    buf = bytearray()
    deadline = time.monotonic() + timeout
    while len(buf) < nbytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for {nbytes} bytes (got {len(buf)})"
            )
        ready, _, _ = select.select([sock], [], [], max(remaining, 0.1))
        if not ready:
            continue
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Server closed connection (got {len(buf)}/{nbytes} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def read_state(ctrl: socket.socket) -> int:
    """Read one signed-byte state value from the control socket."""
    data = recv_exact(ctrl, 1)
    return struct.unpack("!b", data)[0]


def send_state(ctrl: socket.socket, state: int) -> None:
    """Write one signed-byte state value to the control socket."""
    send_all(ctrl, struct.pack("!b", state))


def send_json(ctrl: socket.socket, obj: dict) -> None:
    """Send a JSON object preceded by a 4-byte network-order length."""
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    send_all(ctrl, struct.pack("!I", len(payload)))
    send_all(ctrl, payload)


def recv_json(ctrl: socket.socket) -> dict:
    """Receive a length-prefixed JSON object from the control socket."""
    length_bytes = recv_exact(ctrl, 4)
    length = struct.unpack("!I", length_bytes)[0]
    if length == 0 or length > 10 * 1024 * 1024:
        raise ValueError(f"Unexpected JSON payload length: {length}")
    payload = recv_exact(ctrl, length)
    return json.loads(payload.decode("utf-8"))


# ---------------------------------------------------------------------------
# Data-stream helpers
# ---------------------------------------------------------------------------

def fill_buffer(size: int) -> bytes:
    """Create a repeating-pattern data buffer (matches iperf3 convention)."""
    buf = bytearray(size)
    for i in range(size):
        buf[i] = ord("0") + (i % 10)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Interval reporting
# ---------------------------------------------------------------------------

def format_bytes(nbytes: float) -> str:
    """Return a human-friendly byte string."""
    for unit in ("Bytes", "KBytes", "MBytes", "GBytes", "TBytes"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PBytes"


def format_bits(bits_per_sec: float) -> str:
    """Return a human-friendly bits/sec string."""
    for unit in ("bits/sec", "Kbits/sec", "Mbits/sec", "Gbits/sec"):
        if abs(bits_per_sec) < 1000:
            return f"{bits_per_sec:.2f} {unit}"
        bits_per_sec /= 1000
    return f"{bits_per_sec:.2f} Tbits/sec"


# ---------------------------------------------------------------------------
# Main client logic
# ---------------------------------------------------------------------------

class Iperf3Client:
    """Pure-Python iperf3 TCP client."""

    def __init__(
        self,
        server: str,
        port: int = DEFAULT_PORT,
        duration: int = DEFAULT_DURATION,
        blksize: int = DEFAULT_TCP_BLKSIZE,
        num_streams: int = 1,
        cc_algo: str = "cubic",
        verbose: bool = False,
    ):
        self.server = server
        self.port = port
        self.duration = duration
        self.blksize = blksize
        self.num_streams = num_streams
        self.cc_algo = cc_algo
        self.verbose = verbose

        self.cookie: bytes = make_cookie()
        self.ctrl_sock: socket.socket | None = None
        self.data_socks: list[socket.socket] = []
        self.data_buf: bytes = fill_buffer(self.blksize)

        self.total_bytes_sent = 0
        self.interval_bytes_sent = 0
        self.test_start_time = 0.0
        self.interval_start_time = 0.0

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect_tcp(self) -> socket.socket:
        """Open a TCP connection to self.server:self.port with timeout."""
        sock = socket.create_connection(
            (self.server, self.port), timeout=CONNECT_TIMEOUT
        )
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_CONGESTION,
            self.cc_algo.encode("ascii"),
        )

        # read back actual congestion control in use
        actual_cc = sock.getsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_CONGESTION,
            32
        ).rstrip(b"\x00").decode("ascii", errors="ignore")

        print(f"[cc] requested={self.cc_algo} actual={actual_cc}")

        return sock

    def _open_control(self) -> None:
        """Establish the control connection and send the cookie."""
        self._log("Connecting control channel...")
        self.ctrl_sock = self._connect_tcp()
        local_addr, local_port = self.ctrl_sock.getsockname()[:2]
        remote_addr, remote_port = self.ctrl_sock.getpeername()[:2]
        print(
            f"[  0] local {local_addr} port {local_port} "
            f"connected to {remote_addr} port {remote_port}"
        )
        send_all(self.ctrl_sock, self.cookie)
        self._log("Control connection established, cookie sent.")

    def _open_data_stream(self) -> socket.socket:
        """Open one data stream and authenticate with the cookie."""
        sock = self._connect_tcp()
        send_all(sock, self.cookie)
        return sock

    # ------------------------------------------------------------------
    # Protocol phases
    # ------------------------------------------------------------------

    def _send_parameters(self) -> None:
        """Send the JSON test parameters to the server."""
        params = {
            "tcp": True,
            "omit": 0,
            "time": self.duration,
            "num": 0,
            "blockcount": 0,
            "parallel": self.num_streams,
            "len": self.blksize,
            "pacing_timer": 1000,
            "client_version": "3.16",
        }
        self._log(f"Sending parameters: {json.dumps(params)}")
        send_json(self.ctrl_sock, params)

    def _create_streams(self) -> None:
        """Open the data-stream TCP connections."""
        self._log(f"Creating {self.num_streams} data stream(s)...")
        for i in range(self.num_streams):
            sock = self._open_data_stream()
            self.data_socks.append(sock)
            self._log(f"  Stream {i} connected.")

    def _send_results(self) -> None:
        """Send client-side results to the server during EXCHANGE_RESULTS."""
        elapsed = time.monotonic() - self.test_start_time
        streams = []
        per_stream_bytes = self.total_bytes_sent // max(self.num_streams, 1)
        for i in range(self.num_streams):
            streams.append({
                "id": i + 1,
                "bytes": per_stream_bytes,
                "retransmits": -1,
                "jitter": 0,
                "errors": 0,
                "omitted_errors": 0,
                "packets": 0,
                "omitted_packets": 0,
                "start_time": 0,
                "end_time": elapsed,
            })
        results = {
            "cpu_util_total": 0,
            "cpu_util_user": 0,
            "cpu_util_system": 0,
            "sender_has_retransmits": -1,
            "streams": streams,
        }
        self._log("Sending client results...")
        send_json(self.ctrl_sock, results)

    def _recv_results(self) -> dict | None:
        """Receive server-side results."""
        try:
            server_results = recv_json(self.ctrl_sock)
            self._log("Received server results.")
            return server_results
        except Exception as exc:
            self._log(f"Warning: failed to receive server results: {exc}")
            return None

    # ------------------------------------------------------------------
    # Data transmission
    # ------------------------------------------------------------------

    def _run_test(self) -> None:
        """Continuously send data on all streams for self.duration seconds."""
        self.test_start_time = time.monotonic()
        self.interval_start_time = self.test_start_time
        self.total_bytes_sent = 0
        self.interval_bytes_sent = 0

        end_time = self.test_start_time + self.duration
        report_interval = 1.0
        next_report = self.test_start_time + report_interval

        ctrl_fd = self.ctrl_sock.fileno()

        print(
            f"[ ID] {'Interval':>15}  {'Transfer':>12}  {'Bitrate':>16}"
        )
        print("-" * 60)

        while True:
            now = time.monotonic()
            if now >= end_time:
                break

            ready_r, _, _ = select.select([ctrl_fd], [], [], 0)
            if ready_r:
                state = read_state(self.ctrl_sock)
                name = STATE_NAMES.get(state, str(state))
                if state == SERVER_TERMINATE:
                    self._log(f"Server terminated the test (state={name}).")
                    return
                elif state == ACCESS_DENIED:
                    raise RuntimeError("Server denied access (busy).")
                elif state in (SERVER_ERROR,):
                    raise RuntimeError(f"Server error (state={name}).")

            for sock in self.data_socks:
                try:
                    send_all(sock, self.data_buf)
                    self.total_bytes_sent += self.blksize
                    self.interval_bytes_sent += self.blksize
                except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                    self._log(f"Data stream error: {exc}")
                    return

            now = time.monotonic()
            if now >= next_report:
                elapsed_interval = now - self.interval_start_time
                if elapsed_interval > 0:
                    bitrate = (self.interval_bytes_sent * 8) / elapsed_interval
                    total_elapsed = now - self.test_start_time
                    print(
                        f"[  0] {total_elapsed - elapsed_interval:6.2f}-{total_elapsed:<6.2f} sec  "
                        f"{format_bytes(self.interval_bytes_sent):>12}  "
                        f"{format_bits(bitrate):>16}"
                    )
                self.interval_bytes_sent = 0
                self.interval_start_time = now
                next_report = now + report_interval

        now = time.monotonic()
        elapsed_interval = now - self.interval_start_time
        if elapsed_interval > 0 and self.interval_bytes_sent > 0:
            bitrate = (self.interval_bytes_sent * 8) / elapsed_interval
            total_elapsed = now - self.test_start_time
            print(
                f"[  0] {total_elapsed - elapsed_interval:6.2f}-{total_elapsed:<6.2f} sec  "
                f"{format_bytes(self.interval_bytes_sent):>12}  "
                f"{format_bits(bitrate):>16}"
            )

    # ------------------------------------------------------------------
    # Main state-machine driver
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute a full iperf3 test and return a summary dict.

        Raises on protocol or network errors.
        """
        try:
            self._open_control()
            return self._drive_state_machine()
        finally:
            self._cleanup()

    def _drive_state_machine(self) -> dict:
        """Read server state bytes and react until IPERF_DONE."""
        server_results = None

        while True:
            state = read_state(self.ctrl_sock)
            name = STATE_NAMES.get(state, str(state))
            self._log(f"Server → state {name} ({state})")

            if state == PARAM_EXCHANGE:
                self._send_parameters()

            elif state == CREATE_STREAMS:
                self._create_streams()

            elif state == TEST_START:
                self._log("Test starting...")

            elif state == TEST_RUNNING:
                self._log("Test running — sending data...")
                self._run_test()
                self._log("Duration elapsed — sending TEST_END.")
                send_state(self.ctrl_sock, TEST_END)

            elif state == EXCHANGE_RESULTS:
                self._send_results()
                server_results = self._recv_results()

            elif state == DISPLAY_RESULTS:
                self._print_summary(server_results)
                send_state(self.ctrl_sock, IPERF_DONE)
                self._log("Test complete — sent IPERF_DONE.")
                break

            elif state == IPERF_DONE:
                self._log("Test complete (IPERF_DONE).")
                break

            elif state == ACCESS_DENIED:
                raise RuntimeError(
                    "Server is busy running another test. Try again later."
                )

            elif state == SERVER_ERROR:
                err_bytes = recv_exact(self.ctrl_sock, 4)
                err_code = struct.unpack("!i", err_bytes)[0]
                errno_bytes = recv_exact(self.ctrl_sock, 4)
                errno_val = struct.unpack("!i", errno_bytes)[0]
                raise RuntimeError(
                    f"Server error: code={err_code}, errno={errno_val}"
                )

            elif state == SERVER_TERMINATE:
                self._log("Server terminated the test prematurely.")
                break

            else:
                self._log(f"Unknown state {state} — ignoring.")

        total_elapsed = time.monotonic() - self.test_start_time if self.test_start_time else 0
        return {
            "bytes_sent": self.total_bytes_sent,
            "duration": total_elapsed,
            "bitrate_bps": (self.total_bytes_sent * 8 / total_elapsed) if total_elapsed > 0 else 0,
            "server_results": server_results,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_summary(self, server_results: dict | None) -> None:
        elapsed = time.monotonic() - self.test_start_time if self.test_start_time else 0
        if elapsed > 0:
            bitrate = (self.total_bytes_sent * 8) / elapsed
        else:
            bitrate = 0

        print("-" * 60)
        print(
            f"[  0]  0.00-{elapsed:<6.2f} sec  "
            f"{format_bytes(self.total_bytes_sent):>12}  "
            f"{format_bits(bitrate):>16}  sender"
        )

        if server_results and "streams" in server_results:
            for stream in server_results["streams"]:
                srv_bytes = stream.get("bytes", 0)
                srv_end = stream.get("end_time", elapsed)
                srv_start = stream.get("start_time", 0)
                srv_dur = srv_end - srv_start if srv_end > srv_start else elapsed
                srv_bitrate = (srv_bytes * 8 / srv_dur) if srv_dur > 0 else 0
                print(
                    f"[  0]  {srv_start:.2f}-{srv_end:<6.2f} sec  "
                    f"{format_bytes(srv_bytes):>12}  "
                    f"{format_bits(srv_bitrate):>16}  receiver"
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        for sock in self.data_socks:
            try:
                sock.close()
            except OSError:
                pass
        self.data_socks.clear()
        if self.ctrl_sock:
            try:
                self.ctrl_sock.close()
            except OSError:
                pass
            self.ctrl_sock = None

    def _log(self, msg: str) -> None:
        log(msg, self.verbose)


# ---------------------------------------------------------------------------
# Server-list fetching & selection
# ---------------------------------------------------------------------------

SERVER_LIST_URL = "https://export.iperf3serverlist.net/listed_iperf3_servers.json"
SERVER_LIST_FETCH_TIMEOUT = 15  # seconds


def fetch_server_list(verbose: bool = False) -> list[dict]:
    """Download the public iperf3 server list JSON from iperf3serverlist.net."""
    log("Fetching server list from iperf3serverlist.net ...", verbose)
    req = urllib.request.Request(
        SERVER_LIST_URL,
        headers={"User-Agent": "iperf3-python-client/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=SERVER_LIST_FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to fetch server list from {SERVER_LIST_URL}: {exc}"
        ) from exc

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Server list is empty or malformed.")
    log(f"Fetched {len(data)} servers from the list.", verbose)
    return data


def parse_port(port_str: str) -> int:
    """
    Parse a PORT field like "5201", "5201-5209", or "9200-9240".
    Returns a single port — picks a random one from the range.
    """
    port_str = port_str.strip()
    if "-" in port_str:
        parts = port_str.split("-", 1)
        try:
            lo, hi = int(parts[0]), int(parts[1])
            return random.randint(lo, hi)
        except ValueError:
            pass
    try:
        return int(port_str)
    except ValueError:
        return DEFAULT_PORT


def select_random_servers(
    server_list: list[dict], n: int, verbose: bool = False
) -> list[dict]:
    """
    Shuffle the full server list and return the first *n* entries
    (as parsed dicts with 'host' and 'port' keys).

    The caller will iterate through them, skipping failures and drawing
    replacements from the remaining pool until *n* successful tests are
    completed or the pool is exhausted.
    """
    shuffled = list(server_list)
    random.shuffle(shuffled)

    parsed: list[dict] = []
    for entry in shuffled:
        host = entry.get("IP/HOST", "").strip()
        port_str = entry.get("PORT", str(DEFAULT_PORT))
        if not host:
            continue
        parsed.append({
            "host": host,
            "port": parse_port(port_str),
            "site": entry.get("SITE", ""),
            "country": entry.get("COUNTRY", ""),
            "provider": entry.get("PROVIDER", ""),
        })
    return parsed


def run_multi_destination(
    n: int,
    duration: int,
    blksize: int,
    num_streams: int,
    cc_algo: str,
    verbose: bool,
) -> list[dict]:
    """
    Fetch the public server list, pick *n* random destinations, and run
    a throughput test against each.  Servers that refuse, time out, or
    error are skipped and a replacement is drawn from the remaining pool.

    Returns a list of per-server result dicts.
    """
    full_list = fetch_server_list(verbose)
    pool = select_random_servers(full_list, len(full_list), verbose)

    results: list[dict] = []
    idx = 0  # pointer into the pool

    while len(results) < n and idx < len(pool):
        srv = pool[idx]
        idx += 1

        host, port = srv["host"], srv["port"]
        label = f"{host}:{port}"
        if srv["site"]:
            label += f" ({srv['site']}, {srv['country']})"

        print(f"\n{'='*70}")
        print(
            f"[{len(results)+1}/{n}] Testing {label} "
            f"  (provider: {srv['provider']})"
        )
        print(f"{'='*70}")
        sys.stdout.flush()

        client = Iperf3Client(
            server=host,
            port=port,
            duration=duration,
            blksize=blksize,
            num_streams=num_streams,
            cc_algo = cc_algo,
            verbose=verbose,
        )

        try:
            summary = client.run()
            summary["server_host"] = host
            summary["server_port"] = port
            summary["server_label"] = label
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


def print_final_table(results: list[dict]) -> None:
    """Print a summary table of all completed tests."""
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'#':>3}  {'Server':<40}  {'Duration':>8}  {'Sent':>12}  {'Bitrate':>16}"
    )
    print("-" * 90)
    for i, r in enumerate(results, 1):
        label = r.get("server_label", f"{r.get('server_host')}:{r.get('server_port')}")
        if len(label) > 40:
            label = label[:37] + "..."
        print(
            f"{i:>3}  {label:<40}  "
            f"{r['duration']:7.1f}s  "
            f"{format_bytes(r['bytes_sent']):>12}  "
            f"{format_bits(r['bitrate_bps']):>16}"
        )
    print("-" * 90)
    if results:
        avg_bps = sum(r["bitrate_bps"] for r in results) / len(results)
        print(f"     Average bitrate across {len(results)} servers: {format_bits(avg_bps)}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-Python iperf3 TCP throughput client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Single-server mode:\n"
            "  python3 iperf3_client.py speedtest.wtnet.de -p 5200 -t 10\n\n"
            "Auto-select n random servers from iperf3serverlist.net:\n"
            "  python3 iperf3_client.py --auto -n 10 -t 60\n"
        ),
    )
    parser.add_argument(
        "server", nargs="?", default=None,
        help="iperf3 server hostname or IP (omit when using --auto)"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-select random servers from the public iperf3 server list"
    )
    parser.add_argument(
        "--num-servers", "-n", type=int, default=10,
        help="Number of random servers to test in --auto mode (default 10)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=DEFAULT_PORT,
        help=f"Server port (default {DEFAULT_PORT}, ignored in --auto mode)"
    )
    parser.add_argument(
        "--duration", "-t", type=int, default=DEFAULT_DURATION,
        help=f"Test duration per server in seconds (default {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--blocksize", "-l", type=int, default=DEFAULT_TCP_BLKSIZE,
        help=f"Block size in bytes (default {DEFAULT_TCP_BLKSIZE})"
    )
    parser.add_argument(
        "--streams", "-P", type=int, default=1,
        help="Number of parallel streams (default 1)"
    )
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Enable verbose/debug output"
    )

    parser.add_argument(
        "--cc", default="cubic",
        help="TCP congestion control algorithm (cubic, reno, algo)"
    )

    args = parser.parse_args()

    # ---- Auto mode: fetch list and iterate over n random servers -----------
    if args.auto:
        results = run_multi_destination(
            n=args.num_servers,
            duration=args.duration,
            blksize=args.blocksize,
            num_streams=args.streams,
            verbose=args.verbose,
        )
        print_final_table(results)
        sys.exit(0 if results else 1)

    # ---- Single-server mode -----------------------------------------------
    if args.server is None:
        parser.error("Provide a server hostname, or use --auto mode.")

    client = Iperf3Client(
        server=args.server,
        port=args.port,
        duration=args.duration,
        blksize=args.blocksize,
        cc_algo=args.cc,        
        num_streams=args.streams,
        verbose=args.verbose,
    )

    print(f"Connecting to host {args.server}, port {args.port}")
    sys.stdout.flush()

    try:
        summary = client.run()
    except (ConnectionError, TimeoutError, RuntimeError, OSError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\niperf Done. Sent {format_bytes(summary['bytes_sent'])} "
        f"in {summary['duration']:.2f}s "
        f"({format_bits(summary['bitrate_bps'])} average)."
    )


if __name__ == "__main__":
    main()
