import os
import datetime
import socket
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]

# Must be set before init_process_group
os.environ.setdefault("GLOO_SOCKET_IFNAME", "en0")
os.environ.setdefault("TORCH_GLOO_LAZY_INIT", "1")

print("hostname:", socket.gethostname(), flush=True)
print("rank:", rank, flush=True)
print("world_size:", world_size, flush=True)
print("master:", f"{master_addr}:{master_port}", flush=True)
print("GLOO_SOCKET_IFNAME:", os.environ["GLOO_SOCKET_IFNAME"], flush=True)
print("TORCH_GLOO_LAZY_INIT:", os.environ["TORCH_GLOO_LAZY_INIT"], flush=True)

print(f"Rank {rank}: starting init...", flush=True)

dist.init_process_group(
    backend="gloo",
    init_method=f"tcp://{master_addr}:{master_port}",
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