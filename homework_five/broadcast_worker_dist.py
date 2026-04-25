import argparse, base64, json, math, os, statistics, time
from pathlib import Path
import torch
import torch.distributed as dist
import datetime

# number of ranks should be a power of 2
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def make_store(rank: int, world_size: int) -> dist.TCPStore:
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    print(
        f"Rank {rank}: connecting to TCPStore at "
        f"MASTER_ADDR={master_addr} MASTER_PORT={master_port}",
        flush=True,
    )
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=datetime.timedelta(seconds=30),
    )
    print(f"Rank {rank}: connected", flush=True)
    return store


def store_send_tensor(store: dist.TCPStore, tensor: torch.Tensor, src: int, dst: int, tag: str):
    key = f"tensor/{tag}/{src}/{dst}"
    payload = base64.b64encode(tensor.contiguous().numpy().tobytes()).decode("ascii")
    store.set(key, payload)


def store_recv_tensor(store: dist.TCPStore, tensor: torch.Tensor, src: int, dst: int, tag: str):
    key = f"tensor/{tag}/{src}/{dst}"
    payload = bytearray(base64.b64decode(store.get(key)))
    tensor.copy_(torch.frombuffer(payload, dtype=tensor.dtype).view_as(tensor))


def store_barrier(store: dist.TCPStore, rank: int, world_size: int, label: str):
    store.set(f"barrier/{label}/{rank}", "1")
    store.wait([f"barrier/{label}/{r}" for r in range(world_size)])


def store_max(store: dist.TCPStore, rank: int, world_size: int, label: str, value: float) -> float:
    keys = [f"max/{label}/{r}" for r in range(world_size)]
    store.set(keys[rank], str(value))
    store.wait(keys)
    result_key = f"max/{label}/result"
    if rank == 0:
        max_value = max(float(store.get(key).decode("utf-8")) for key in keys)
        store.set(result_key, str(max_value))
    store.wait([result_key])
    return float(store.get(result_key).decode("utf-8"))


def binary_tree_broadcast(
    store: dist.TCPStore,
    buf: torch.Tensor,
    world_size: int,
    rank: int,
    op_name: str,
    root: int = 0,
) -> torch.Tensor:
    # Virtual rank with root remapped to 0. Tree: parent = (v-1)//2, children = 2v+1, 2v+2.
    vrank = (rank - root) % world_size
    # Receive from parent (unless we are the root)
    if vrank != 0:
        parent = ((vrank - 1) // 2 + root) % world_size
        store_recv_tensor(store, buf, parent, rank, f"{op_name}/binary")
    # Sending the same buffer to both children is safe (both ops only read from it).
    #children
    left_v = 2 * vrank + 1
    right_v = 2 * vrank + 2

    if left_v < world_size:
        left = (left_v + root) % world_size
        store_send_tensor(store, buf, rank, left, f"{op_name}/binary")
    if right_v < world_size:
        right = (right_v + root) % world_size
        store_send_tensor(store, buf, rank, right, f"{op_name}/binary")
    return buf


def binomial_tree_broadcast(
    store: dist.TCPStore,
    buf: torch.Tensor,
    world_size: int,
    rank: int,
    op_name: str,
    root: int = 0,
) -> torch.Tensor:
    vrank = (rank - root) % world_size
    num_steps = max(1, math.ceil(math.log2(world_size))) if world_size > 1 else 0
    mask = 1
    for k in range(num_steps):
        if vrank < mask:
            peer_v = vrank + mask
            if peer_v < world_size:
                peer = (peer_v + root) % world_size
                store_send_tensor(store, buf, rank, peer, f"{op_name}/binomial/{k}")
        elif vrank < (mask << 1):
            peer_v = vrank - mask
            peer = (peer_v + root) % world_size
            store_recv_tensor(store, buf, peer, rank, f"{op_name}/binomial/{k}")
        mask <<= 1
    return buf


def run_algo(algo, store, buf, world_size, rank, root, op_name):
    if algo == 'binary_tree':
        return binary_tree_broadcast(store, buf, world_size, rank, op_name, root)
    if algo == 'binomial_tree':
        return binomial_tree_broadcast(store, buf, world_size, rank, op_name, root)
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

    store = make_store(rank, world_size)

    # Warm-up run (outside the timing loop)
    buf = make_local(args.msg_bytes, rank, args.root)
    buf = run_algo(args.algo, store, buf, world_size, rank, args.root, "warmup")
    verify(buf, args.msg_bytes)
    store_barrier(store, rank, world_size, "warmup")

    samples = []
    for i in range(args.iters):
        # Reset buffer each iteration so non-root ranks start from the sentinel
        # and verification catches missed receives.
        buf = make_local(args.msg_bytes, rank, args.root)
        store_barrier(store, rank, world_size, f"iter-{i}-pre")
        t0 = time.perf_counter()
        buf = run_algo(args.algo, store, buf, world_size, rank, args.root, f"iter-{i}")
        store_barrier(store, rank, world_size, f"iter-{i}-post")
        elapsed = (time.perf_counter() - t0) * 1000.0
        verify(buf, args.msg_bytes)
        # Report worst-rank completion time, matching allgather_worker.py.
        samples.append(store_max(store, rank, world_size, f"iter-{i}", elapsed))

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


if __name__ == '__main__':
    main()
