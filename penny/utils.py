import torch
import torch.distributed as dist
import penny_cpp
import os
from typing import Optional, Union

# Adapted from: https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/utils.py
def bench_kineto(fn, kernel_name: str, num_tests: int = 30, suppress_kineto_output: bool = False,
                 barrier_comm_profiling: bool = True):
    # Profile
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for i in range(2):
            # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
            if barrier_comm_profiling:
                lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                lhs @ rhs
                dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
            for _ in range(num_tests):
                # Alocate big tensor to clear cache
                x = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                fn()
            prof.step()

    # Parse the profiling table
    times = []
    for e in prof.profiler.function_events:
        # HACK remove the sync allreduce from lines
        if kernel_name in e.name and "AllReduce_Sum_f32" not in e.name:
            times.append(e.device_time_total)
    # Remove outliers
    # TODO investigate where are they coming from
    median = list(sorted(times))[len(times)//2]
    times = [t for t in times if 0.5 < t/median < 1.5]
    return sum(times)/len(times)


def initialize_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES"))
    local_size = world_size//nnodes
    local_rank = dist.get_rank() % local_size

    torch.cuda.set_device(local_rank)
    nvshmem_uid = penny_cpp.get_unique_id()

    nvshmem_uids = [None, ] * world_size
    dist.all_gather_object(nvshmem_uids, nvshmem_uid)
    penny_cpp.init_with_uid(nvshmem_uids[0], dist.get_rank(), world_size)
