from __future__ import annotations

import csv
import os
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple


IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HOP_LINE_RE = re.compile(r"^\s*(\d+)\s+(.*)$")
RTT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ms\b")

# ---- Config (edit these) ----------------------------------------------------
CSV_PATH = os.path.join("data", "listed_iperf3_servers.csv")
NUM_DESTINATIONS = 5
MAX_HOPS = 30
WAIT_S = 2
PROBES_PER_HOP = 1

OUTPUT_CSV_PATH = "generated_data/question_2_traceroute_rtts.csv"


@dataclass(frozen=True)
class HopRtt:
    destination: str
    hop: int
    hop_ip: str
    rtt_ms_mean: float
    rtts_ms: Tuple[float, ...]


def _is_valid_ipv4(ip: str) -> bool:
    # Basic validation: dotted-quad with each octet 0..255
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return False
    return all(0 <= o <= 255 for o in octets)


def load_ipv4_candidates(csv_path: str) -> List[str]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # The file header is "IP/HOST"
        col = "IP/HOST"
        if reader.fieldnames and col not in reader.fieldnames:
            # fallback: use first column
            col = reader.fieldnames[0]

        ips: List[str] = []
        for row in reader:
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            m = IPV4_RE.search(raw)
            if not m:
                continue
            ip = m.group(0)
            if _is_valid_ipv4(ip):
                ips.append(ip)

    # De-duplicate while preserving order
    deduped = list(dict.fromkeys(ips))
    if not deduped:
        raise RuntimeError(f"No IPv4 addresses found in {csv_path}")
    return deduped


def pick_random(items: Sequence[str], count: int) -> List[str]:
    rng = random.Random()
    if count >= len(items):
        return list(items)
    return rng.sample(list(items), k=count)


def run_traceroute(dest_ip: str, max_hops: int, wait_s: int, probes: int) -> str:
    # macOS traceroute:
    # -n: numeric output (no DNS), easier to parse
    # -q: number of probes per hop
    # -w: wait time (seconds)
    # -m: max TTL / max hops
    cmd = [
        "traceroute",
        "-n",
        "-m",
        str(max_hops),
        "-w",
        str(wait_s),
        "-q",
        str(probes),
        dest_ip,
    ]

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "traceroute command not found. On macOS it should be available by default."
        ) from e

    return proc.stdout


def parse_traceroute_output(dest_ip: str, output: str) -> List[HopRtt]:
    hops: List[HopRtt] = []

    for line in output.splitlines():
        m = HOP_LINE_RE.match(line)
        if not m:
            continue

        hop_num = int(m.group(1))
        rest = m.group(2)

        # Skip fully non-responsive hops, usually like: "3  * * *"
        if "*" in rest and not IPV4_RE.search(rest):
            continue

        ip_m = IPV4_RE.search(rest)
        if not ip_m:
            continue
        hop_ip = ip_m.group(0)
        if not _is_valid_ipv4(hop_ip):
            continue

        rtts = tuple(float(x) for x in RTT_RE.findall(rest))
        if not rtts:
            # Some lines can include an IP but no RTT (rare); treat as non-responsive
            continue

        rtt_mean = sum(rtts) / len(rtts)
        hops.append(
            HopRtt(
                destination=dest_ip,
                hop=hop_num,
                hop_ip=hop_ip,
                rtt_ms_mean=rtt_mean,
                rtts_ms=rtts,
            )
        )

    return hops


def default_output_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(os.getcwd(), f"question_2_traceroute_rtts_{ts}.csv")


def write_csv(rows: Iterable[HopRtt], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "destination",
                "hop",
                "hop_ip",
                "rtt_ms_mean",
                "rtts_ms",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "destination": r.destination,
                    "hop": r.hop,
                    "hop_ip": r.hop_ip,
                    "rtt_ms_mean": f"{r.rtt_ms_mean:.3f}",
                    "rtts_ms": ";".join(f"{x:.3f}" for x in r.rtts_ms),
                }
            )


def main() -> int:
    ips = load_ipv4_candidates(CSV_PATH)
    sampled = pick_random(ips, count=NUM_DESTINATIONS)

    print(f"Loaded {len(ips)} IPv4 addresses from {CSV_PATH}")
    print(f"Random sample ({len(sampled)}): {', '.join(sampled)}")
    print("")

    all_rows: List[HopRtt] = []
    for i, dest in enumerate(sampled, start=1):
        print(f"[{i}/{len(sampled)}] traceroute to {dest} ...")
        out = run_traceroute(dest, max_hops=MAX_HOPS,
                             wait_s=WAIT_S, probes=PROBES_PER_HOP)
        rows = parse_traceroute_output(dest, out)
        all_rows.extend(rows)
        if rows:
            print(f"  responsive hops: {len(rows)} (showing up to 5)")
            for r in rows[:5]:
                print(
                    f"    hop {r.hop:>2} {r.hop_ip:<15} {r.rtt_ms_mean:.3f} ms")
        else:
            print(
                "  no responsive hops parsed (destination may be unreachable or filtered).")
        print("")

    out_path = OUTPUT_CSV_PATH or default_output_path()
    write_csv(all_rows, out_path)
    print(f"Wrote {len(all_rows)} hop RTT rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
