
import torch
import torch.distributed as dist
import os
import penny_cpp
from penny.utils import bench_kineto, initialize_distributed
from torch.profiler import profile, ProfilerActivity
from triton.testing import do_bench

initialize_distributed()
rank = dist.get_rank()
world_size = dist.get_world_size()
nnodes = int(os.getenv("NNODES"))
local_size = world_size//nnodes
local_rank = dist.get_rank() % local_size

num = 2**20
atol = 1e-2
rtol = 1e-4
for packet_size in [8, 16, 32, 64, 512]:
    for block_size in [256, 512, 1024]:
# for packet_size in [8]:
#     for block_size in [256]:
        if num%(packet_size * block_size * world_size) != 0:
            continue
        configuration = f"{packet_size=} {block_size=}"

        data = torch.randn(num).to("cuda").to(torch.float16)
        data2 = data.clone()

        dist.all_reduce(data)
        penny_cpp.all_reduce(data2, packet_size, block_size, nnodes)

        if not torch.allclose(data, data2, atol=atol, rtol=rtol) and rank == 0:
            idx = torch.isclose(data, data2, atol=atol, rtol=rtol) 
            num_missed =  idx.logical_not().sum() / idx.nelement()
            print(f"failed {configuration=} {rank=}, {num_missed=} {data.mean()}, {data2.mean()}")
            print(data[idx.logical_not()])
            print(data2[idx.logical_not()])

        nccl_time =  bench_kineto(lambda: dist.all_reduce(data), kernel_names="nccl")
        penny_time = bench_kineto(lambda: penny_cpp.all_reduce(data2, packet_size, block_size, nnodes), kernel_names="all_reduce")


        if rank == 0:
            recv_bytes = data2.nelement() * data2.element_size() * (world_size-1)//world_size * 2
            print(f"{configuration=} nccl time: {nccl_time*1e6:.2f}us, bandwidth {recv_bytes / 1e9 / nccl_time :.2f} GB/s  penny_time: {penny_time*1e6:.2f}, bandwidth {recv_bytes / 1e9 / penny_time :.2f} GB")
