import argparse, json, math, os, statistics, time
from pathlib import Path
import torch
import torch.distributed as dist
import datetime

# number of ranks should be a power of 2
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _wait_all(reqs):
    for req in reqs:
        req.wait()


def binary_tree_broadcast(buf: torch.Tensor, world_size: int, rank: int, root: int = 0) -> torch.Tensor:
    # Virtual rank with root remapped to 0. Tree: parent = (v-1)//2, children = 2v+1, 2v+2.
    vrank = (rank - root) % world_size
    # Receive from parent (unless we are the root)
    if vrank != 0:
        parent = ((vrank - 1) // 2 + root) % world_size
        ops = [dist.P2POp(dist.irecv, buf, parent)]
        _wait_all(dist.batch_isend_irecv(ops))
    # Sending the same buffer to both children is safe (both ops only read from it).
    ops = []

    #children
    left_v = 2 * vrank + 1
    right_v = 2 * vrank + 2

    if left_v < world_size:
        left = (left_v + root) % world_size
        ops.append(dist.P2POp(dist.isend, buf, left))
    if right_v < world_size:
        right = (right_v + root) % world_size
        ops.append(dist.P2POp(dist.isend, buf, right))
    if ops:
        _wait_all(dist.batch_isend_irecv(ops))
    return buf


def binomial_tree_broadcast(buf: torch.Tensor, world_size: int, rank: int, root: int = 0) -> torch.Tensor:
    vrank = (rank - root) % world_size
    num_steps = max(1, math.ceil(math.log2(world_size))) if world_size > 1 else 0
    mask = 1
    for k in range(num_steps):
        if vrank < mask:
            peer_v = vrank + mask
            if peer_v < world_size:
                peer = (peer_v + root) % world_size
                ops = [dist.P2POp(dist.isend, buf, peer)]
                _wait_all(dist.batch_isend_irecv(ops))
        elif vrank < (mask << 1):
            peer_v = vrank - mask
            peer = (peer_v + root) % world_size
            ops = [dist.P2POp(dist.irecv, buf, peer)]
            _wait_all(dist.batch_isend_irecv(ops))
        mask <<= 1
    return buf


def run_algo(algo, buf, world_size, rank, root):
    if algo == 'binary_tree':
        return binary_tree_broadcast(buf, world_size, rank, root)
    if algo == 'binomial_tree':
        return binomial_tree_broadcast(buf, world_size, rank, root)
    raise ValueError(algo)

"""
Expected result: every rank holds the root's fill value. Non-root ranks pre-fill
with a sentinel so a no-op implementation (which would leave the sentinel in
place) fails verification.
"""

ROOT_FILL = 0x5A
NONROOT_SENTINEL = 0xFF


def make_local(msg_bytes: int, rank: int, root: int) -> torch.Tensor:
    value = ROOT_FILL if rank == root else NONROOT_SENTINEL
    return torch.full((msg_bytes,), value, dtype=torch.uint8)


def verify(buf: torch.Tensor, msg_bytes: int):
    exp = torch.full((msg_bytes,), ROOT_FILL, dtype=torch.uint8)
    if not torch.equal(buf, exp):
        raise RuntimeError('broadcast result does not match root fill pattern')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', default='binary_tree', choices=['binary_tree', 'binomial_tree'])
    ap.add_argument('--msg-bytes', type=int, default=1024)
    ap.add_argument('--iters', type=int, default=1)
    ap.add_argument('--root', type=int, default=0)
    ap.add_argument('--result-file', default=None)
    args = ap.parse_args()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    torch.set_num_threads(1)

    if args.result_file is None:
        args.result_file = f'broadcast_{args.algo}_{args.msg_bytes}_{args.root}_{rank}.json'

    print(
        f"Rank {rank}: initializing with "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
        f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}",
        flush=True,
    )

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )

    print(f"Rank {rank}: initialized", flush=True)

    # Warm-up run (outside the timing loop)
    buf = make_local(args.msg_bytes, rank, args.root)
    buf = run_algo(args.algo, buf, world_size, rank, args.root)
    verify(buf, args.msg_bytes)
    dist.barrier()

    samples = []
    for _ in range(args.iters):
        # Reset buffer each iteration so non-root ranks start from the sentinel
        # and verification catches missed receives.
        buf = make_local(args.msg_bytes, rank, args.root)
        dist.barrier()
        t0 = time.perf_counter()
        buf = run_algo(args.algo, buf, world_size, rank, args.root)
        dist.barrier()
        elapsed = (time.perf_counter() - t0) * 1000.0
        verify(buf, args.msg_bytes)
        # Report worst-rank completion time, matching allgather_worker.py
        t = torch.tensor([elapsed], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(float(t.item()))

    if rank == 0:
        result = {
            'algorithm': args.algo,
            'world_size': world_size,
            'msg_bytes_per_rank': args.msg_bytes,
            'root': args.root,
            'samples_ms': samples,
            'median_ms': statistics.median(samples),
            'min_ms': min(samples),
            'max_ms': max(samples),
        }
        Path(args.result_file).write_text(json.dumps(result))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
