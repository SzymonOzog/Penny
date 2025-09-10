import argparse
import torch
import torch.distributed as dist
import os
import penny_cpp
from torch.profiler import profile, ProfilerActivity, record_function
from penny.utils import bench_kineto, initialize_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-sizes", type=int, nargs="+",
                        default=[2, 8, 32, 64, 128, 256, 512, 2048, 4096, 8192],
                        help="List of packet sizes to test")
    parser.add_argument("--block-sizes", type=int, nargs="+",
                        default=[32, 128, 256, 512, 1024],
                        help="List of block sizes to test")
    parser.add_argument("--atol", type=float, default=1e-1,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--profile-mode", type=str, default="info",
                        help="How do you want to profile (info|verbose|file|none)")
    parser.add_argument("--algo", type=int, default=0,
                        help="What type of algo to use, 0 == ring, 1 == tree")
    parser.add_argument("--range", type=int, default=1,
                        help="How many sizes to profile")
    parser.add_argument("--start-pow", type=int, default=28,
                        help="What power of 2 size to start from")

    args = parser.parse_args()

    initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES", "1"))
    local_size = world_size // nnodes
    local_rank = rank % local_size

    def run_benchmark():
        for pow in range(args.start_pow, args.start_pow + args.range):
            num = 2 ** pow
            best_time = float("inf")
            best_configuration = None
            for packet_size in args.packet_sizes:
                for block_size in args.block_sizes:
                    rings = 1 if nnodes == 1 else local_size
                    if pow > 23 and packet_size < 32:
                        continue
                    if num % (packet_size * block_size * world_size * rings) != 0 and args.algo == 0:
                        continue
                    if num % (packet_size * block_size * local_size) != 0 and args.algo == 1:
                        continue
                    configuration = f"{packet_size=} {block_size=} {num=}, {rings=}"

                    data = torch.randn(num, device="cuda", dtype=torch.float16)
                    # data = torch.ones(num, device="cuda", dtype=torch.float16)
                    data2 = data.clone()
                    recv_bytes = data2.nelement() * data2.element_size() * (world_size - 1) // world_size * 2
                    handle = penny_cpp.all_reduce_create(data2, packet_size, block_size, nnodes, rings)

                    with record_function(configuration):
                        dist.all_reduce(data)
                        penny_cpp.all_reduce_run(handle)

                    if not torch.allclose(data, data2, atol=args.atol, rtol=args.rtol) and rank == 0:
                        idx = torch.isclose(data, data2, atol=args.atol, rtol=args.rtol)
                        num_missed = idx.logical_not().sum() / idx.nelement()
                        print(f"failed {configuration=} {rank=}, {num_missed=} {data.mean()}, {data2.mean()}")
                        print(data[idx.logical_not()])
                        print(data2[idx.logical_not()])

                    if args.profile_mode == "info" or args.profile_mode == "verbose":
                        penny_time = bench_kineto(lambda: penny_cpp.all_reduce_run(handle),
                                                  kernel_names="all_reduce")
                        nccl_time = bench_kineto(lambda: dist.all_reduce(data), kernel_names="AllReduce_Sum_f16")
                        if penny_time < best_time:
                            best_time = penny_time
                            best_configuration = configuration

                        if rank == 0 and args.profile_mode == "verbose":
                            print(f"{configuration=} nccl time: {nccl_time*1e6:.2f}us, "
                                  f"bandwidth {recv_bytes / 1e9 / nccl_time :.2f} GB/s  "
                                  f"penny_time: {penny_time*1e6:.2f}us, "
                                  f"bandwidth {recv_bytes / 1e9 / penny_time :.2f} GB/s")
                    penny_cpp.all_reduce_destroy(handle)

            if rank == 0 and args.profile_mode == "info" and best_configuration is None:
                print(f"no configuration found for {num=}")
            elif rank == 0 and args.profile_mode == "info":
                print(f"{best_configuration=} nccl time: {nccl_time*1e6:.2f}us, "
                      f"bandwidth {recv_bytes / 1e9 / nccl_time :.2f} GB/s  "
                      f"penny_time time: {best_time*1e6:.2f}us, "
                      f"bandwidth {recv_bytes / 1e9 / best_time :.2f} GB/s")

    if args.profile_mode == "file" and rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            run_benchmark()
        prof.export_chrome_trace("trace_penny.json")
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
