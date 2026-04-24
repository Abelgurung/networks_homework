import argparse, json, math, os, statistics, time
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


def _wait_all(reqs):
    for req in reqs:
        req.wait()

# the layout is:
# bytes 0 .. m-1 belong to rank 0
# bytes m .. 2m-1 belong to rank 1
# ...
# bytes (p-1)m .. pm-1 belong to rank p-1

def ring_allgather(local: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    m = local.numel()
    out = torch.empty((world_size, m), dtype=local.dtype)
    out[rank].copy_(local)
    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size # at every step, it sends out what it learned in the last step
        recv_idx = (rank - step - 1) % world_size
        ops = [
            dist.P2POp(dist.isend, out[send_idx].contiguous(), right, tag=1000 + step), #point to point communication - nonblocking send and rcv
            dist.P2POp(dist.irecv, out[recv_idx], left, tag=1000 + step),
        ]
        _wait_all(dist.batch_isend_irecv(ops))
    return out.reshape(-1)


def recursive_doubling_allgather(local: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
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
        ops = [
            dist.P2POp(dist.isend, send_t, peer, tag=2000 + k),
            dist.P2POp(dist.irecv, recv_t, peer, tag=2000 + k),
        ]
        _wait_all(dist.batch_isend_irecv(ops))
    return out.reshape(-1)


def swing_allgather(local: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
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
        ops = [
            dist.P2POp(dist.isend, send_ids, peer, tag=3000 + phase),
            dist.P2POp(dist.irecv, recv_ids, peer, tag=3000 + phase),
            dist.P2POp(dist.isend, send_blocks, peer, tag=4000 + phase),
            dist.P2POp(dist.irecv, recv_blocks, peer, tag=4000 + phase),
        ]
        _wait_all(dist.batch_isend_irecv(ops))
        ids = recv_ids.tolist()
        for j, bid in enumerate(ids):
            out[bid].copy_(recv_blocks[j*m:(j+1)*m])
        known = sorted(set(known).union(ids))
    return out.reshape(-1)


def run_algo(algo, local, world_size, rank):
    if algo == 'ring':
        return ring_allgather(local, world_size, rank)
    if algo == 'recursive_doubling':
        return recursive_doubling_allgather(local, world_size, rank)
    if algo == 'swing':
        return swing_allgather(local, world_size, rank)
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
    local = torch.full((args.msg_bytes,), rank % 251, dtype=torch.uint8) # Each rank’s local message is a one-dimensional byte tensor of length msg_bytes
    gathered = run_algo(args.algo, local, world_size, rank)
    verify(gathered, args.msg_bytes, world_size)
    dist.barrier()
    samples = []
    for _ in range(args.iters):
        dist.barrier()
        t0 = time.perf_counter()
        gathered = run_algo(args.algo, local, world_size, rank)
        dist.barrier()
        elapsed = (time.perf_counter() - t0) * 1000.0
        verify(gathered, args.msg_bytes, world_size)
        t = torch.tensor([elapsed], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(float(t.item()))
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
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
