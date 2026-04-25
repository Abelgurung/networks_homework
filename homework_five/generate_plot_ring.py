import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_vs_message_size(results, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["msg_bytes_per_rank"] / 1024 for r in results],
        [r["median_ms"] for r in results],
        marker="o",
        label="Ring Topology",
    )

    plt.xscale("log", base=2)
    plt.xlabel("Message size per rank (KiB)")
    plt.ylabel("Completion time (ms)")
    plt.title("AllGather in Ring")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plot.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate ring benchmark plots from JSON results."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to a directory of JSON files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the PNG plots",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix == ".json":
            with open(f) as f:
                results.append(json.load(f))
    results.sort(key=lambda e: e["msg_bytes_per_rank"])
    print(results)

    plot_vs_message_size(results, output_dir)


if __name__ == "__main__":
    main()
