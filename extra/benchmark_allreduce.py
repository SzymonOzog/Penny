import argparse
import torch
import torch.distributed as dist
import os
import penny_cpp
from penny.utils import bench_kineto, initialize_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=2**20,
                        help="Total number of elements to reduce")
    parser.add_argument("--packet-sizes", type=int, nargs="+",
                        default=[8, 16, 32, 64, 512],
                        help="List of packet sizes to test")
    parser.add_argument("--block-sizes", type=int, nargs="+",
                        default=[256, 512, 1024],
                        help="List of block sizes to test")
    parser.add_argument("--atol", type=float, default=3e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for correctness check")

    args = parser.parse_args()

    initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES", "1"))
    local_size = world_size // nnodes
    local_rank = rank % local_size

    for packet_size in args.packet_sizes:
        for block_size in args.block_sizes:
            if args.num % (packet_size * block_size * world_size) != 0:
                continue
            configuration = f"{packet_size=} {block_size=}"

            data = torch.randn(args.num, device="cuda", dtype=torch.float16)
            data2 = data.clone()

            dist.all_reduce(data)
            penny_cpp.all_reduce(data2, packet_size, block_size, nnodes)

            if not torch.allclose(data, data2, atol=args.atol, rtol=args.rtol) and rank == 0:
                idx = torch.isclose(data, data2, atol=args.atol, rtol=args.rtol)
                num_missed = idx.logical_not().sum() / idx.nelement()
                print(f"failed {configuration=} {rank=}, {num_missed=} {data.mean()}, {data2.mean()}")
                print(data[idx.logical_not()])
                print(data2[idx.logical_not()])

            nccl_time = bench_kineto(lambda: dist.all_reduce(data), kernel_names="nccl")
            penny_time = bench_kineto(lambda: penny_cpp.all_reduce(data2, packet_size, block_size, nnodes),
                                      kernel_names="all_reduce")

            if rank == 0:
                recv_bytes = data2.nelement() * data2.element_size() * (world_size - 1) // world_size * 2
                print(f"{configuration=} nccl time: {nccl_time*1e6:.2f}us, "
                      f"bandwidth {recv_bytes / 1e9 / nccl_time :.2f} GB/s  "
                      f"penny_time: {penny_time*1e6:.2f}us, "
                      f"bandwidth {recv_bytes / 1e9 / penny_time :.2f} GB/s")


if __name__ == "__main__":
    main()

