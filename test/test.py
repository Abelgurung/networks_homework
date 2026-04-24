import os
import datetime
import socket
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print("hostname:", socket.gethostname(), flush=True)
print("rank:", rank, flush=True)
print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"), flush=True)
print("MASTER_PORT:", os.environ.get("MASTER_PORT"), flush=True)
print("GLOO_SOCKET_IFNAME:", os.environ.get("GLOO_SOCKET_IFNAME"), flush=True)
print(f"Rank {rank}: starting init...", flush=True)

dist.init_process_group(
    backend="gloo",
    init_method="env://",
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(seconds=20),
)

print(f"Rank {rank}: initialized", flush=True)

x = torch.tensor([rank + 1.0])
print(f"Rank {rank}: before all_reduce {x}", flush=True)

dist.all_reduce(x, op=dist.ReduceOp.SUM)

print(f"Rank {rank}: after all_reduce {x}", flush=True)

dist.destroy_process_group()