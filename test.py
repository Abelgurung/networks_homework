import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=60),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    # Test 1: all_reduce
    x = torch.tensor([rank], dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Test 2: all_gather
    gathered = [torch.zeros(1) for _ in range(world_size)]
    my_value = torch.tensor([rank], dtype=torch.float32)
    dist.all_gather(gathered, my_value)

    print(
        f"host={hostname} "
        f"rank={rank}/{world_size} "
        f"all_reduce_sum={x.item()} "
        f"all_gather={[int(t.item()) for t in gathered]}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()