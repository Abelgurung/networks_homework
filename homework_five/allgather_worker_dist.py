import argparse, base64, json, math, os, statistics, time
from pathlib import Path
import torch
import torch.distributed as dist
import datetime

# number of ranks should be a power of 2
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

# offset seq for swing
def swing_rho(step: int) -> int:
    return (1 - ((-2) ** (step + 1))) // 3


def swing_peer(rank: int, step: int, world_size: int) -> int:
    r = swing_rho(step)
    return (rank + r) % world_size if rank % 2 == 0 else (rank - r) % world_size


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

# the layout is:
# bytes 0 .. m-1 belong to rank 0
# bytes m .. 2m-1 belong to rank 1
# ...
# bytes (p-1)m .. pm-1 belong to rank p-1

def ring_allgather(
    store: dist.TCPStore,
    local: torch.Tensor,
    world_size: int,
    rank: int,
    op_name: str,
) -> torch.Tensor:
    m = local.numel()
    out = torch.empty((world_size, m), dtype=local.dtype)
    out[rank].copy_(local)
    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size # at every step, it sends out what it learned in the last step
        recv_idx = (rank - step - 1) % world_size
        tag = f"{op_name}/ring/{step}"
        store_send_tensor(store, out[send_idx].contiguous(), rank, right, tag)
        store_recv_tensor(store, out[recv_idx], left, rank, tag)
    return out.reshape(-1)


def recursive_doubling_allgather(
    store: dist.TCPStore,
    local: torch.Tensor,
    world_size: int,
    rank: int,
    op_name: str,
) -> torch.Tensor:
    assert is_power_of_two(world_size)
    m = local.numel()
    out = torch.empty((world_size, m), dtype=local.dtype)
    out[rank].copy_(local)
    for k in range(int(math.log2(world_size))):
        group = 1 << k
        peer = rank ^ group
        send_start = (rank // group) * group
        recv_start = (peer // group) * group
        send_t = out[send_start:send_start+group].contiguous().reshape(-1)
        recv_t = out[recv_start:recv_start+group].reshape(-1)
        tag = f"{op_name}/recursive_doubling/{k}"
        store_send_tensor(store, send_t, rank, peer, tag)
        store_recv_tensor(store, recv_t, peer, rank, tag)
    return out.reshape(-1)


def swing_allgather(
    store: dist.TCPStore,
    local: torch.Tensor,
    world_size: int,
    rank: int,
    op_name: str,
) -> torch.Tensor:
    assert is_power_of_two(world_size)
    m = local.numel()
    out = torch.empty((world_size, m), dtype=local.dtype)
    out[rank].copy_(local)
    known = [rank]
    logp = int(math.log2(world_size))
    for phase in range(logp):
        step = logp - 1 - phase # reverse step order relative to phase
        peer = swing_peer(rank, step, world_size) # choose a peer
        send_ids = torch.tensor(sorted(known), dtype=torch.int64) # send the list of known block IDs
        recv_ids = torch.empty(len(known), dtype=torch.int64)
        send_blocks = torch.cat([out[i].contiguous() for i in send_ids.tolist()], dim=0) # send the actual concatenated payload for those IDs
        recv_blocks = torch.empty_like(send_blocks)
        ids_tag = f"{op_name}/swing/ids/{phase}"
        blocks_tag = f"{op_name}/swing/blocks/{phase}"
        store_send_tensor(store, send_ids, rank, peer, ids_tag)
        store_send_tensor(store, send_blocks, rank, peer, blocks_tag)
        store_recv_tensor(store, recv_ids, peer, rank, ids_tag)
        store_recv_tensor(store, recv_blocks, peer, rank, blocks_tag)
        ids = recv_ids.tolist()
        for j, bid in enumerate(ids):
            out[bid].copy_(recv_blocks[j*m:(j+1)*m])
        known = sorted(set(known).union(ids))
    return out.reshape(-1)


def run_algo(algo, store, local, world_size, rank, op_name):
    if algo == 'ring':
        return ring_allgather(store, local, world_size, rank, op_name)
    if algo == 'recursive_doubling':
        return recursive_doubling_allgather(store, local, world_size, rank, op_name)
    if algo == 'swing':
        return swing_allgather(store, local, world_size, rank, op_name)
    raise ValueError(algo)


def verify(gathered, msg_bytes, world_size):
    out = gathered.view(world_size, msg_bytes)
    for r in range(world_size):
        exp = torch.full((msg_bytes,), r % 251, dtype=torch.uint8)
        if not torch.equal(out[r], exp):
            raise RuntimeError(f'bad block {r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', default='ring')
    ap.add_argument('--msg-bytes', type=int, default=1024)
    ap.add_argument('--iters', type=int, default=1)
    ap.add_argument('--result-file', default=None)
    args = ap.parse_args()
    rank = int(os.environ['RANK']) # process index
    world_size = int(os.environ['WORLD_SIZE']) # number of processes
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    torch.set_num_threads(1)

    if args.result_file is None:
        args.result_file = f'allgather_{args.algo}_{args.msg_bytes}_{rank}.json'

    store = make_store(rank, world_size)
    local = torch.full((args.msg_bytes,), rank % 251, dtype=torch.uint8) # Each rank’s local message is a one-dimensional byte tensor of length msg_bytes
    gathered = run_algo(args.algo, store, local, world_size, rank, "warmup")
    verify(gathered, args.msg_bytes, world_size)
    store_barrier(store, rank, world_size, "warmup")
    samples = []
    for i in range(args.iters):
        store_barrier(store, rank, world_size, f"iter-{i}-pre")
        t0 = time.perf_counter()
        gathered = run_algo(args.algo, store, local, world_size, rank, f"iter-{i}")
        store_barrier(store, rank, world_size, f"iter-{i}-post")
        elapsed = (time.perf_counter() - t0) * 1000.0
        verify(gathered, args.msg_bytes, world_size)
        samples.append(store_max(store, rank, world_size, f"iter-{i}", elapsed))

    if rank == 0:
        result = {
            'algorithm': args.algo,
            'world_size': world_size,
            'msg_bytes_per_rank': args.msg_bytes,
            'samples_ms': samples,
            'median_ms': statistics.median(samples),
            'min_ms': min(samples),
            'max_ms': max(samples),
        }
        Path(args.result_file).write_text(json.dumps(result))

    print("finishing")
    store_barrier(store, rank, world_size, "shutdown")

if __name__ == '__main__':
    main()
