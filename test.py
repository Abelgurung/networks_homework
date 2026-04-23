import os
import sys
import torch
import torch.distributed as dist

# Each machine passes in one number
my_number = float(sys.argv[1])

dist.init_process_group(
    backend="gloo",
    init_method="env://",
)

x = torch.tensor([my_number])

# Sum x across both machines
dist.all_reduce(x, op=dist.ReduceOp.SUM)

rank = dist.get_rank()

if rank == 0:
    print("Sum =", x.item())

dist.destroy_process_group()