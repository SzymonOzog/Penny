import torch
import os
import torch.distributed as dist
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

packet_size = 512
block_size = 1024
num = 2**26
for gpu_offset in range(8):
    if num%(packet_size * block_size) != 0:
        continue

    if rank < world_size//2:
        src = ((rank + gpu_offset)%local_size + local_size)%world_size
    else:
        src = ((rank//local_size) * local_size + (local_rank - gpu_offset)%local_size + local_size)%world_size
    configuration = f"{gpu_offset=} {src=}"

    data = torch.randn(num).to("cuda").to(torch.float16)
    data2 = data.clone()
    data_r = torch.FloatTensor([0,] * num).to("cuda").to(torch.float16)

    ops = [dist.P2POp(dist.isend, data, src),
           dist.P2POp(dist.irecv, data_r, src)]
    if rank >= world_size//(2):
        ops = list(reversed(ops))

    dist.batch_isend_irecv(ops)
    torch.cuda.synchronize()

    penny_cpp.exchange(data2, packet_size, block_size, src)
    if not (data_r==data2).all():
        if rank == 0:
            idx = data_r == data2
            num_missed =  idx.logical_not().sum() / idx.nelement()
            print(f"failed {configuration=} {rank=}, {num_missed=}")

    nccl_time =  bench_kineto(lambda: dist.batch_isend_irecv(ops), kernel_name="nccl")
    penny_time = bench_kineto(lambda: penny_cpp.exchange(data2, packet_size, block_size, src), "exchange")


    if rank == 0:
        recv_bytes = data2.nelement() * data2.element_size() * 2
        print(f"{configuration=} nccl time: {nccl_time:.2f}us, bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  penny_time: {penny_time:.2f}, bandwidth {recv_bytes / 1e3 / penny_time :.2f} GB")
