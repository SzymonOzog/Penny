import argparse
import torch
import torch.distributed as dist
import os
import penny_cpp
from torch.profiler import profile, ProfilerActivity, record_function
from penny.utils import bench_kineto, initialize_distributed
os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = "32"
os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-sizes", type=int, nargs="+",
                        default=[2, 8, 16, 32, 64, 128, 256, 512, 2048],
                        help="List of packet sizes to test")
    parser.add_argument("--block-sizes", type=int, nargs="+",
                        default=[32, 128, 256, 512, 1024],
                        help="List of block sizes to test")
    parser.add_argument("--profile-mode", type=str, default="info",
                        help="How do you want to profile (info|verbose|file|none)")
    parser.add_argument("--mode", type=str, default="intranode",
                        help="Exchange mode (intranode|internode)")
    parser.add_argument("--range", type=int, default=1,
                        help="How many sizes to profile")
    parser.add_argument("--start-pow", type=int, default=20,
                        help="What power of 2 size to start from")
    parser.add_argument("--num-tests", type=int, default=3,
                        help="How many tests in a row to run")

    args = parser.parse_args()

    initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES", "1"))
    local_size = world_size // nnodes
    local_rank = rank % local_size
    # float16
    elem_size = 2

    def run_benchmark():
        for pow in range(args.start_pow, args.start_pow + args.range):
            num = 2 ** pow
            best_time = float("inf")
            best_configuration = None
            for packet_size in args.packet_sizes:
                for block_size in args.block_sizes:
                    if pow > 23 and packet_size < 32:
                        continue
                    if (num * elem_size) % (packet_size * block_size) != 0:
                        continue

                    if args.mode == "internode":
                        src = (rank + local_size) % world_size
                    else:
                        if local_rank % 2 == 0:
                            src = (rank // local_size) * local_size + (local_rank + 1) % local_size
                        else:
                            src = (rank // local_size) * local_size + (local_rank - 1) % local_size
                    configuration = f"mode={args.mode} {packet_size=} {block_size=} {num=} {src=}"

                    data = torch.empty(num, device="cuda", dtype=torch.float16).normal_(mean=0, std=0.1)
                    data2 = data.clone()
                    data_r = torch.zeros(num, device="cuda", dtype=torch.float16)
                    recv_bytes = data2.nelement() * data2.element_size() * 2

                    ops = [dist.P2POp(dist.isend, data, src),
                           dist.P2POp(dist.irecv, data_r, src)]
                    if args.mode == "internode" and rank >= world_size // 2:
                        ops = list(reversed(ops))
                    if args.mode == "intranode" and rank % 2:
                        ops = list(reversed(ops))

                    for _ in range(args.num_tests):
                        # Reset data for each test
                        data2.copy_(data)
                        data_r.zero_()

                        with record_function(configuration):
                            dist.batch_isend_irecv(ops)
                            torch.cuda.synchronize()
                            penny_cpp.exchange(data2, packet_size, block_size, src)

                        if not (data2 == data_r).all() and rank == 0:
                            idx = (data2 == data_r)
                            num_missed = idx.logical_not().sum() / idx.nelement()
                            print(f"failed {configuration=} {rank=}, {num_missed=} {data_r.mean()}, {data2.mean()}")
                            print(data_r[idx.logical_not()])
                            print(data2[idx.logical_not()])

                    if args.profile_mode == "info" or args.profile_mode == "verbose":
                        penny_time = bench_kineto(lambda: penny_cpp.exchange(data2, packet_size, block_size, src),
                                                  kernel_name="exchange")
                        nccl_time = bench_kineto(lambda: dist.batch_isend_irecv(ops), kernel_name="nccl")

                        if penny_time < best_time:
                            best_time = penny_time
                            best_configuration = configuration

                        if rank == 0 and args.profile_mode == "verbose":
                            print(f"{configuration=} nccl time: {nccl_time:.2f}us, "
                                  f"bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  "
                                  f"penny_time: {penny_time:.2f}us, "
                                  f"bandwidth {recv_bytes / 1e3 / penny_time :.2f} GB/s")

            if rank == 0 and args.profile_mode == "info" and best_configuration is None:
                print(f"no configuration found for {num=}")
            elif rank == 0 and args.profile_mode == "info":
                print(f"{best_configuration=} nccl time: {nccl_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  "
                      f"penny_time time: {best_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / best_time :.2f} GB/s")

    if args.profile_mode == "file" and rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            run_benchmark()
        prof.export_chrome_trace("trace_penny_exchange.json")
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
