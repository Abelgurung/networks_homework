import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def flatten_results(obj):
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(flatten_results(item))
        return out
    raise TypeError(f"Unsupported JSON root type: {type(obj)!r}")


def load_results(path: Path):
    text = path.read_text().strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        results = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            results.extend(flatten_results(json.loads(line)))
        return results

    return flatten_results(json.loads(text))


def validate_results(results):
    required = {"algorithm", "world_size", "msg_bytes_per_rank", "median_ms"}
    for i, r in enumerate(results):
        missing = required - set(r.keys())
        if missing:
            raise ValueError(f"Result {i} is missing keys: {sorted(missing)}")


def plot_vs_message_size(results, output_dir: Path, fixed_world_size: int, algorithms):
    plt.figure(figsize=(8, 5))
    plotted = False
    for algo in algorithms:
        subset = sorted(
            [r for r in results if r["world_size"] == fixed_world_size and r["algorithm"] == algo],
            key=lambda x: x["msg_bytes_per_rank"],
        )
        if not subset:
            continue
        plt.plot(
            [r["msg_bytes_per_rank"] / 1024 for r in subset],
            [r["median_ms"] for r in subset],
            marker="o",
            label=algo.replace("_", " ").title(),
        )
        plotted = True

    if not plotted:
        raise ValueError(f"No results found for fixed world size {fixed_world_size}")

    plt.xscale("log", base=2)
    plt.xlabel("Message size per rank (KB)")
    plt.ylabel("Completion time (ms)")
    plt.title(f"Broadcast on Gloo vs message size (world size = {fixed_world_size})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "broadcast_vs_message_size.png", dpi=180)
    plt.close()


def plot_vs_world_size(results, output_dir: Path, fixed_msg_bytes: int, algorithms):
    plt.figure(figsize=(8, 5))
    plotted = False
    xticks = sorted({r["world_size"] for r in results if r["msg_bytes_per_rank"] == fixed_msg_bytes})
    for algo in algorithms:
        subset = sorted(
            [r for r in results if r["msg_bytes_per_rank"] == fixed_msg_bytes and r["algorithm"] == algo],
            key=lambda x: x["world_size"],
        )
        if not subset:
            continue
        plt.plot(
            [r["world_size"] for r in subset],
            [r["median_ms"] for r in subset],
            marker="o",
            label=algo.replace("_", " ").title(),
        )
        plotted = True

    if not plotted:
        raise ValueError(f"No results found for fixed message size {fixed_msg_bytes} bytes")

    if xticks:
        plt.xticks(xticks)
    plt.xlabel("Number of ranks")
    plt.ylabel("Completion time (ms)")
    plt.title(f"Broadcast on Gloo vs number of ranks (message size = {fixed_msg_bytes // 1024} KB per rank)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "broadcast_vs_world_size.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Broadcast benchmark plots from JSON or JSONL results.")
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL results file")
    parser.add_argument("--output-dir", default="broadcast_bench", help="Directory to write the PNG plots")
    parser.add_argument("--fixed-world-size", type=int, default=8, help="World size used for the message-size plot")
    parser.add_argument("--fixed-msg-bytes", type=int, default=1048576, help="Message size per rank used for the world-size plot")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["binary_tree", "binomial_tree"],
        help="Algorithms to include in the plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_path)
    validate_results(results)

    combined_path = output_dir / "broadcast_results.json"
    combined_path.write_text(json.dumps(results, indent=2))

    plot_vs_message_size(results, output_dir, args.fixed_world_size, args.algorithms)
    plot_vs_world_size(results, output_dir, args.fixed_msg_bytes, args.algorithms)

    print(f"Wrote {combined_path}")
    print(f"Wrote {output_dir / 'broadcast_vs_message_size.png'}")
    print(f"Wrote {output_dir / 'broadcast_vs_world_size.png'}")


if __name__ == "__main__":
    main()
