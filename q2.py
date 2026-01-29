# to run:
#
# python3 -m venv .venv
# source .venv/bin/activate (or .venv\Scripts\activate on Windows)
# pip install -r requirements.txt
# python3 q2.py list.txt
#

import sys
import argparse
import random
import subprocess
import platform
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Regex for parsing Windows tracert output
# Example: "  1    <1 ms    <1 ms    <1 ms  192.168.1.1"
# Example: "  2     5 ms     4 ms     5 ms  10.0.0.1"
# Example: " 10     *        *        *     Request timed out."
_WIN_TRACERT_RE = re.compile(
    r"^\s*(?P<hop>\d+)\s+"
    r"(?:(?P<t1>[\d<]+)\s+ms|\*)\s+"
    r"(?:(?P<t2>[\d<]+)\s+ms|\*)\s+"
    r"(?:(?P<t3>[\d<]+)\s+ms|\*)\s+"
    r"(?P<host>.*)$"
)

# Regex for parsing Linux traceroute output
# Example: " 1  192.168.1.1 (192.168.1.1)  0.123 ms  0.456 ms  0.789 ms"
_LINUX_TRACEROUTE_RE = re.compile(
    r"^\s*(?P<hop>\d+)\s+"
    r"(?P<host>[\S]+)\s+\((?P<ip>[\d\.]+)\)\s+"
    r"(?:(?P<t1>[\d\.]+)\s+ms|\*)\s+"
    r"(?:(?P<t2>[\d\.]+)\s+ms|\*)\s+"
    r"(?:(?P<t3>[\d\.]+)\s+ms|\*)"
)

def run_traceroute(target: str) -> str:
    """
    Runs traceroute/tracert to the target IP/hostname and returns the stdout output.
    """
    system = platform.system().lower()
    
    if "windows" in system:
        # Windows tracert
        # -d: Do not resolve addresses to hostnames (faster)
        # -w 1000: Wait 1000ms for each reply
        cmd = ["tracert", "-d", "-w", "1000", target]
    else:
        # Linux/Mac traceroute
        # -w 1: Wait 1 second
        # -n: Do not resolve addresses (faster)
        cmd = ["traceroute", "-n", "-w", "1", target]
        
    print(f"Running {' '.join(cmd)}...")
    try:
        # Capture output. text=True ensures we get string, not bytes.
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return result.stdout
    except FileNotFoundError:
        print(f"Error: Traceroute command not found. Ensure 'tracert' (Windows) or 'traceroute' (Linux/Mac) is installed.")
        return ""

def parse_traceroute_output(output: str) -> List[Dict]:
    """
    Parses traceroute output and returns a list of hops.
    Each hop is a dict: {'hop': int, 'rtt': float, 'address': str}
    RTT is the average of valid times for that hop.
    """
    hops = []
    lines = output.splitlines()
    
    for line in lines:
        # Try Windows regex
        m_win = _WIN_TRACERT_RE.match(line)
        if m_win:
            hop_num = int(m_win.group("hop"))
            times = []
            for t_group in ["t1", "t2", "t3"]:
                val = m_win.group(t_group)
                if val and val != "*":
                    if val.startswith("<"):
                        times.append(0.5) # Treat <1 ms as 0.5 ms
                    else:
                        times.append(float(val))
            
            host = m_win.group("host").strip()
            
            # If all timed out, we might skip or record as None
            if not times:
                continue # Skip non-responsive hops as per instructions ("filter out non-responsive hops")
                
            avg_rtt = sum(times) / len(times)
            hops.append({
                "hop": hop_num,
                "rtt": avg_rtt,
                "address": host
            })
            continue

        # Try Linux regex
        # Note: Linux output is more variable.
        # Simplified parsing for standard linux output:
        # " 1  1.2.3.4 (1.2.3.4)  1.0 ms  2.0 ms  3.0 ms"
        parts = line.split()
        if not parts:
            continue
            
        # Basic heuristic check if line starts with hop number
        if parts[0].isdigit():
            try:
                hop_num = int(parts[0])
                # Find ms values
                # Look for tokens ending in "ms" or just numbers followed by "ms"
                times = []
                # Linux output can have multiple * and times intermixed.
                # Just regex search for times in the line
                # "  1.234 ms"
                ms_matches = re.findall(r"([\d\.]+)\s+ms", line)
                
                for ms_val in ms_matches:
                    times.append(float(ms_val))
                
                if not times:
                    continue # Filter out non-responsive
                    
                avg_rtt = sum(times) / len(times)
                
                # Extract IP - usually in parens
                ip_match = re.search(r"\(([\d\.]+)\)", line)
                if ip_match:
                    address = ip_match.group(1)
                else:
                    # Maybe just the second token if no parens (if -n used)
                    address = parts[1]
                
                hops.append({
                    "hop": hop_num,
                    "rtt": avg_rtt,
                    "address": address
                })
            except (ValueError, IndexError):
                continue
                
    return hops

def plot_latency_breakdown(all_traces: Dict[str, List[Dict]], output_pdf: Path):
    """
    Plots a stacked bar chart showing the breakdown of latencies to each hop.
    """
    # Prepare data for DataFrame
    # Rows: Destinations
    # Columns: Hops (Hop 1, Hop 2, ...)
    # Values: Incremental latency
    
    data = []
    destinations = list(all_traces.keys())
    
    # Determine max hop count to align bars
    max_hops = 0
    for dest in destinations:
        if all_traces[dest]:
            max_hops = max(max_hops, all_traces[dest][-1]["hop"])
            
    # We need to build a matrix where rows are destinations and columns are hop segments
    # But since hops might be skipped or vary, it's tricky.
    # Approach: For each destination, calculate incremental latencies.
    # If Hop N is missing, we can't easily attribute latency to it specifically, 
    # but we can just use the available hops.
    # The assignment asks for "breakdown of latencies to each hop".
    # This implies we show the contribution of each hop to the total RTT.
    
    plot_data = {}
    
    for dest in destinations:
        hops = all_traces[dest]
        if not hops:
            continue
            
        sorted_hops = sorted(hops, key=lambda x: x['hop'])
        
        # Calculate incremental latencies
        prev_rtt = 0
        increments = {}
        
        for h in sorted_hops:
            hop_idx = h['hop']
            rtt = h['rtt']
            # Incremental latency
            inc = rtt - prev_rtt
            if inc < 0:
                inc = 0 # Should not happen physically, but possible with measurement noise
            
            increments[f"Hop {hop_idx}"] = inc
            prev_rtt = rtt
            
        plot_data[dest] = increments

    df = pd.DataFrame(plot_data).T.fillna(0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', stacked=True, ax=ax, legend=False)
    
    ax.set_title("Latency Breakdown per Hop")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Destination IP")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

def plot_hop_vs_rtt(all_traces: Dict[str, List[Dict]], output_pdf: Path):
    """
    Scatter plot: hop count vs rtt. Each data point corresponds to a destination ip address.
    """
    hop_counts = []
    rtts = []
    labels = []
    
    for dest, hops in all_traces.items():
        if not hops:
            continue
        
        # Last hop gives the total info
        last_hop = max(hops, key=lambda x: x['hop'])
        hop_counts.append(last_hop['hop'])
        rtts.append(last_hop['rtt'])
        labels.append(dest)
        
    if not hop_counts:
        print("No data available for Hop Count vs RTT plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hop_counts, rtts)
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (hop_counts[i], rtts[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
    ax.set_title("Hop Count vs RTT")
    ax.set_xlabel("Hop Count")
    ax.set_ylabel("RTT (ms)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run traceroute to random IPs and plot results.")
    parser.add_argument("input_file", type=Path, help="File containing list of IP addresses")
    args = parser.parse_args()
    
    if not args.input_file.exists():
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)
        
    # Read IPs
    with open(args.input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if len(lines) < 5:
        print("Error: Input file must contain at least 5 IP addresses.")
        sys.exit(1)
        
    # Shuffle all IPs to pick randomly without replacement
    random.shuffle(lines)
    
    all_traces = {}
    selected_ips = []
    
    print(f"Selecting 5 responsive IPs from {len(lines)} available...")
    
    for ip in lines:
        if len(selected_ips) >= 5:
            break
            
        print(f"Testing {ip}...")
        output = run_traceroute(ip)
        hops = parse_traceroute_output(output)
        
        # Check if we got any valid hops
        if hops:
            all_traces[ip] = hops
            selected_ips.append(ip)
            print(f"  -> Success: Found {len(hops)} hops.")
        else:
            print(f"  -> No responsive hops found. Skipping.")
            
        time.sleep(1) # Be nice
        
    if len(selected_ips) < 5:
        print(f"Warning: Only found {len(selected_ips)} responsive IPs out of the entire list.")
    
    print(f"Final selected IPs: {selected_ips}")
        
    # Generate plots
    # Check if we have any valid data
    valid_traces_count = sum(1 for t in all_traces.values() if t)
    if valid_traces_count == 0:
        print("Warning: No responsive hops found for any of the selected IPs. Skipping plot generation.")
        return

    # (b) Stacked bar chart
    try:
        plot_latency_breakdown(all_traces, Path("latency_breakdown.pdf"))
    except Exception as e:
        print(f"Error generating latency breakdown plot: {e}")
    
    # (c) Scatter plot
    try:
        plot_hop_vs_rtt(all_traces, Path("hop_vs_rtt.pdf"))
    except Exception as e:
        print(f"Error generating hop vs RTT plot: {e}")
    
    print("Plots generated (if data was available): latency_breakdown.pdf, hop_vs_rtt.pdf")

if __name__ == "__main__":
    main()
