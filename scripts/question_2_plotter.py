from __future__ import annotations

import csv
import os
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_latency_breakdown(csv_path: Path, output_pdf: Path):
    """
    Plot a stacked bar chart showing the breakdown of latencies to each hop,
    corresponding to each of the five chosen destination IP addresses.
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Group by destination and sort by hop number
    destinations = df['destination'].unique()
    
    # Prepare data for stacked bar chart
    # Each destination will be a bar, each hop will be a segment
    plot_data = {}
    total_latencies = {}
    
    for dest in destinations:
        dest_data = df[df['destination'] == dest].sort_values('hop')
        
        if dest_data.empty:
            continue
        
        # Calculate incremental latencies (difference between consecutive hops)
        prev_rtt = 0
        increments = {}
        
        for _, row in dest_data.iterrows():
            hop_num = int(row['hop'])
            rtt = float(row['rtt_ms_mean'])
            
            # Incremental latency added by this hop
            inc = rtt - prev_rtt
            if inc < 0:
                inc = 0  # Handle measurement noise
            
            increments[f"Hop {hop_num}"] = inc
            prev_rtt = rtt
        
        plot_data[dest] = increments
        total_latencies[dest] = prev_rtt  # Final RTT is the total
    
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(plot_data).T.fillna(0)
    
    # Sort columns by hop number for better visualization
    hop_cols = sorted(plot_df.columns, key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0)
    plot_df = plot_df[hop_cols]
    
    # Sort destinations by total latency (ascending) for easier comparison
    totals = plot_df.sum(axis=1)
    plot_df = plot_df.loc[totals.sort_values().index]
    
    # Create the plot with normal background - sized for single-column LaTeX
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Use a sequential colormap that's colorblind-friendly
    # 'viridis' or 'plasma' work well, but 'tab20' gives more distinct colors
    num_hops = len(hop_cols)
    colors = plt.cm.tab20(range(num_hops)) if num_hops <= 20 else plt.cm.viridis(np.linspace(0, 1, num_hops))
    
    # Create stacked bar chart
    plot_df.plot(kind='bar', stacked=True, ax=ax, width=0.75, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add total latency labels on top of each bar
    totals = plot_df.sum(axis=1)
    for i, (dest, total) in enumerate(totals.items()):
        ax.text(i, total + max(totals) * 0.02, f'{total:.1f}ms', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Styling improvements - appropriate fonts for single-column layout
    ax.set_title("Latency Breakdown per Hop by Destination", fontsize=11, fontweight='bold', pad=12)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_xlabel("Destination IP Address", fontsize=10)
    
    # Legend for single-column layout
    ax.legend(title="Hop Number", bbox_to_anchor=(1.02, 1), loc='upper left', 
              fontsize=8, title_fontsize=9, framealpha=0.9, ncol=1)
    
    # Improve x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Keep x and y axis lines visible and prominent
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0, top=max(totals) * 1.15)
    
    # Tight layout for single-column fit
    plt.tight_layout()
    
    # Save to PDF with high quality
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_pdf}")


def plot_hop_count_vs_rtt(csv_path: Path, output_pdf: Path):
    """
    Plot a scatter plot showing hop count vs RTT.
    Each data point corresponds to a destination IP address.
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Group by destination and get the final hop count and RTT
    destinations = df['destination'].unique()
    
    hop_counts = []
    rtts = []
    dest_labels = []
    
    for dest in destinations:
        dest_data = df[df['destination'] == dest].sort_values('hop')
        
        if dest_data.empty:
            continue
        
        # Get the maximum hop number (hop count)
        max_hop = int(dest_data['hop'].max())
        
        # Get the RTT at the final hop (total RTT to destination)
        final_rtt = float(dest_data[dest_data['hop'] == max_hop]['rtt_ms_mean'].iloc[0])
        
        hop_counts.append(max_hop)
        rtts.append(final_rtt)
        dest_labels.append(dest)
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Assign colors to each destination
    num_destinations = len(dest_labels)
    colors = plt.cm.tab10(range(num_destinations)) if num_destinations <= 10 else plt.cm.tab20(range(num_destinations))
    color_map = dict(zip(dest_labels, colors))
    
    # Create scatter plot with color coding - plot each destination separately for legend
    for hop, rtt, dest in zip(hop_counts, rtts, dest_labels):
        ax.scatter(hop, rtt, s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                  c=[color_map[dest]], label=dest)
    
    # Add legend in bottom right corner (remove duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 0.0), loc='lower right', 
              fontsize=8, framealpha=0.9, title='Destination IP', title_fontsize=9)
    
    # Styling
    ax.set_title("Hop Count vs RTT by Destination", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Hop Count", fontsize=10)
    ax.set_ylabel("RTT (ms)", fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Keep x and y axis lines visible and prominent
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    
    # Set reasonable axis limits
    ax.set_xlim(left=min(hop_counts) - 1, right=max(hop_counts) + 1)
    ax.set_ylim(bottom=0, top=max(rtts) * 1.1)
    
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to PDF with high quality
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_pdf}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    csv_path = project_root / "generated_data" / "question_2_traceroute_rtts.csv"
    plots_dir = project_root / "plots"
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(exist_ok=True)
    
    output_pdf_latency = plots_dir / "latency_breakdown.pdf"
    output_pdf_scatter = plots_dir / "hop_count_vs_rtt.pdf"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    plot_latency_breakdown(csv_path, output_pdf_latency)
    plot_hop_count_vs_rtt(csv_path, output_pdf_scatter)


if __name__ == "__main__":
    main()
