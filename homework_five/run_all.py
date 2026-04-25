import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None):
    print('Running:', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def default_ifname() -> str:
    return 'lo0' if platform.system() == 'Darwin' else 'lo'


def main():
    parser = argparse.ArgumentParser(description='Run all AllGather benchmarks and generate plots.')
    parser.add_argument('--out-dir', default='allgather_bench')
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--world-sizes', nargs='+', type=int, default=[2, 4, 8, 16])
    parser.add_argument('--message-sizes-bytes', nargs='+', type=int, default=[1024, 4096, 16384, 65536, 262144, 1048576, 2097152])
    parser.add_argument('--fixed-world-size', type=int, default=8)
    parser.add_argument('--fixed-msg-bytes', type=int, default=1048576)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=29500)
    parser.add_argument('--gloo-ifname', default=None, help='Network interface for Gloo, e.g. lo (Linux) or lo0 (macOS)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cases_dir = out_dir / 'cases'
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ['ring', 'recursive_doubling', 'swing']

    torchrun = shutil.which('torchrun')
    if torchrun is None:
        launcher_prefix = [sys.executable, '-m', 'torch.distributed.run']
    else:
        launcher_prefix = [torchrun]

    child_env = os.environ.copy()
    child_env.setdefault('MASTER_ADDR', args.master_addr)
    child_env.setdefault('MASTER_PORT', str(args.master_port))
    child_env.setdefault('GLOO_SOCKET_IFNAME', args.gloo_ifname or default_ifname())
    child_env.setdefault('OMP_NUM_THREADS', '1')
    child_env.setdefault('MKL_NUM_THREADS', '1')

    print(f"Using MASTER_ADDR={child_env['MASTER_ADDR']}")
    print(f"Using MASTER_PORT={child_env['MASTER_PORT']}")
    print(f"Using GLOO_SOCKET_IFNAME={child_env['GLOO_SOCKET_IFNAME']}")

    def run_case(world_size: int, algorithm: str, msg_bytes: int):
        result_file = cases_dir / f'{algorithm}_ws{world_size}_msg{msg_bytes}.json'
        cmd = launcher_prefix + [
            f'--nproc_per_node={world_size}',
            f'--master_addr={child_env["MASTER_ADDR"]}',
            f'--master_port={child_env["MASTER_PORT"]}',
            'allgather_worker.py',
            '--algo', algorithm,
            '--msg-bytes', str(msg_bytes),
            '--iters', str(args.iters),
            '--result-file', str(result_file),
        ]
        run(cmd, env=child_env)

    for msg_bytes in args.message_sizes_bytes:
        for algorithm in algorithms:
            run_case(args.fixed_world_size, algorithm, msg_bytes)

    for world_size in args.world_sizes:
        for algorithm in algorithms:
            if world_size == args.fixed_world_size and args.fixed_msg_bytes in args.message_sizes_bytes:
                continue
            run_case(world_size, algorithm, args.fixed_msg_bytes)

    results = []
    for path in sorted(cases_dir.glob('*.json')):
        data = json.loads(path.read_text())
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    results_file = out_dir / 'allgather_results.json'
    results_file.write_text(json.dumps(results, indent=2))
    print(f'Wrote {results_file}')

    run([
        sys.executable,
        'generate_plots.py',
        '--input', str(results_file),
        '--output-dir', str(out_dir),
        '--fixed-world-size', str(args.fixed_world_size),
        '--fixed-msg-bytes', str(args.fixed_msg_bytes),
    ], env=child_env)


if __name__ == '__main__':
    main()
