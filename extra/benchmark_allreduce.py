import argparse
import torch
import torch.distributed as dist
import os
import penny_cpp
from torch.profiler import profile, ProfilerActivity, record_function
from penny.utils import bench_kineto, initialize_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=2**28,
                        help="Total number of elements to reduce")
    parser.add_argument("--packet-sizes", type=int, nargs="+",
                        default=[8, 16, 32, 64, 512],
                        help="List of packet sizes to test")
    parser.add_argument("--block-sizes", type=int, nargs="+",
                        default=[256, 512, 1024],
                        help="List of block sizes to test")
    parser.add_argument("--atol", type=float, default=1e-1,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--profile-mode", type=str, default="info",
                        help="How do you want to profile (info|file|none)")

    args = parser.parse_args()

    initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES", "1"))
    local_size = world_size // nnodes
    local_rank = rank % local_size

    def run_benchmark():
        for packet_size in args.packet_sizes:
            for block_size in args.block_sizes:
                if args.num % (packet_size * block_size * world_size) != 0:
                    continue
                configuration = f"{packet_size=} {block_size=} {args.num=}"

                data = torch.randn(args.num, device="cuda", dtype=torch.float16)
                data = torch.ones(args.num, device="cuda", dtype=torch.float16)
                data2 = data.clone()

                with record_function(configuration):
                    dist.all_reduce(data)
                    penny_cpp.all_reduce(data2, packet_size, block_size, nnodes)

                if not torch.allclose(data, data2, atol=args.atol, rtol=args.rtol) and rank == 0:
                    idx = torch.isclose(data, data2, atol=args.atol, rtol=args.rtol)
                    num_missed = idx.logical_not().sum() / idx.nelement()
                    print(f"failed {configuration=} {rank=}, {num_missed=} {data.mean()}, {data2.mean()}")
                    print(data[idx.logical_not()])
                    print(data2[idx.logical_not()])

                if args.profile_mode == "info":
                    nccl_time = bench_kineto(lambda: dist.all_reduce(data), kernel_names="nccl")
                    penny_time = bench_kineto(lambda: penny_cpp.all_reduce(data2, packet_size, block_size, nnodes),
                                              kernel_names="all_reduce")

                    if rank == 0:
                        recv_bytes = data2.nelement() * data2.element_size() * (world_size - 1) // world_size * 2
                        print(f"{configuration=} nccl time: {nccl_time*1e6:.2f}us, "
                              f"bandwidth {recv_bytes / 1e9 / nccl_time :.2f} GB/s  "
                              f"penny_time: {penny_time*1e6:.2f}us, "
                              f"bandwidth {recv_bytes / 1e9 / penny_time :.2f} GB/s")

    if args.profile_mode == "file" and rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            run_benchmark()
        prof.export_chrome_trace("trace_penny.json")
    else:
        run_benchmark()


if __name__ == "__main__":
    main()

