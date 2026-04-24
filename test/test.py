import os
import sys
import datetime
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_addr = os.environ["MASTER_ADDR"]
master_port = int(os.environ["MASTER_PORT"])

my_number = float(sys.argv[1])

print(f"Rank {rank}: connecting to TCPStore...", flush=True)

store = dist.TCPStore(
    host_name=master_addr,
    port=master_port,
    world_size=world_size,
    is_master=(rank == 0),
    timeout=datetime.timedelta(seconds=20),
)

print(f"Rank {rank}: connected", flush=True)

# Each rank writes its number.
store.set(f"number_{rank}", str(my_number))

# Rank 0 waits for all numbers and sums them.
if rank == 0:
    total = 0.0
    for r in range(world_size):
        value = float(store.get(f"number_{r}").decode("utf-8"))
        print(f"Rank 0: got number_{r} = {value}", flush=True)
        total += value

    print("SUM =", total, flush=True)
else:
    print(f"Rank {rank}: sent {my_number}", flush=True)