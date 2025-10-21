import argparse
import torch
import torch.distributed as dist
import os
import penny_cpp
from torch.profiler import profile, ProfilerActivity, record_function
from penny.utils import bench_kineto, initialize_distributed
os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = "128"
os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-sizes", type=int, nargs="+",
                        default=[2, 8, 16, 32, 64, 128, 256, 512, 2048],
                        help="List of packet sizes to test")
    parser.add_argument("--block-sizes", type=int, nargs="+",
                        default=[32, 128, 256, 512, 1024],
                        help="List of block sizes to test")
    parser.add_argument("--atol", type=float, default=1e-1,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--profile-mode", type=str, default="info",
                        help="How do you want to profile (info|verbose|file|none)")
    parser.add_argument("--algo", type=int, default=0,
                        help="What type of algo to use, 0 == ring, 1 == tree")
    parser.add_argument("--range", type=int, default=1,
                        help="How many sizes to profile")
    parser.add_argument("--start-pow", type=int, default=28,
                        help="What power of 2 size to start from")
    parser.add_argument("--num-tests", type=int, default=3,
                        help="How many tests in a row to run")
    parser.add_argument("--bench_custom", type=bool, default=False,
                        help="Do we want to also bench against vLLM custom all reduce(long init time)")

    args = parser.parse_args()

    initialize_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES", "1"))
    local_size = world_size // nnodes
    local_rank = rank % local_size
    if nnodes == 1 and args.bench_custom:
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        group = dist.new_group(list(range(world_size)), backend="gloo") 
        custom_ar = CustomAllreduce(group, device=local_rank, max_size = 2**(args.start_pow + args.range))
    else:
        custom_ar = None
    # float16
    elem_size = 2
    penny_bandwidth = []
    penny_times = []

    def run_benchmark():
        for pow in range(args.start_pow, args.start_pow + args.range):
            num = 2 ** pow
            best_time = float("inf")
            best_configuration = None
            packet_sizes = args.packet_sizes
            n_routes = [1, 2, 4, 8, 16, 32] if args.algo in [3, 4] else [1]
            for packet_size in packet_sizes:
                for block_size in args.block_sizes:
                    for routes in n_routes:
                        if pow > 23 and packet_size < 32:
                            continue
                        if args.algo == 0 and (num * elem_size) % (packet_size * block_size * world_size * routes) != 0:
                            continue
                        if args.algo == 1 and (num * elem_size) % (packet_size * block_size * routes) != 0:
                            continue
                        if args.algo == 2:
                            # packet_size = (num*elem_size)//block_size
                            if int((num*elem_size)/(packet_size*block_size)) != (num*elem_size)//(packet_size*block_size) or \
                                    (num*elem_size)//(packet_size*block_size) == 0:
                                continue
                        if args.algo == 3:
                            if int((num*elem_size)/(packet_size*block_size*world_size)) != (num*elem_size)//(packet_size*block_size*world_size) or \
                                    (num*elem_size)//(packet_size*block_size*world_size) == 0:
                                continue
                        if args.algo in [2, 3]:
                            reduce_size = (packet_size*block_size)//(world_size*routes)
                            if packet_size < 1 or reduce_size == 0 or reduce_size%16 != 0 or (routes == 32 and block_size == 1024):
                                continue

                        configuration = f"{packet_size=} {block_size=} {num=}, {routes=}"

                        data = torch.empty(num, device="cuda", dtype=torch.float16).normal_(mean=0, std=0.1)
                        mul = [(i*world_size + num)//num for i in range(num)]
                        data = torch.ones(num, device="cuda", dtype=torch.float16) * torch.tensor(mul).to(data.device)
                        data2 = data.clone()
                        data3 = data.clone()
                        penny_out = torch.empty_like(data);
                        recv_bytes = 2 * data2.nelement() * data2.element_size()
                        handle = penny_cpp.all_reduce_create(data2, packet_size, block_size, nnodes, routes, args.algo)

                        for _ in range(args.num_tests):
                            #avoid stacking errors
                            data2.copy_(data)
                            data3.copy_(data)

                            with record_function(configuration):
                                dist.all_reduce(data)
                                penny_cpp.all_reduce_run(handle, penny_out)
                                if custom_ar is not None:
                                    data3 = custom_ar.all_reduce(data3)


                            if not torch.allclose(data, penny_out, atol=args.atol, rtol=args.rtol) and rank == 0:
                                idx = torch.isclose(data, penny_out, atol=args.atol, rtol=args.rtol)
                                num_missed = idx.logical_not().sum() / idx.nelement()
                                print(f"failed {configuration=} {rank=}, {num_missed=} {data.mean()}, {penny_out.mean()}")
                                print(data[idx.logical_not()][:10])
                                print(penny_out[idx.logical_not()][:10])
                                print(data[:10])
                                print(penny_out[:10])

                            if custom_ar is not None and not torch.allclose(data, data3, atol=args.atol, rtol=args.rtol) and rank == 0:
                                idx = torch.isclose(data, data3, atol=args.atol, rtol=args.rtol)
                                num_missed = idx.logical_not().sum() / idx.nelement()
                                print(f"failed {configuration=} {rank=}, {num_missed=} {data.mean()}, {data3.mean()}")
                                print(data[idx.logical_not()][:10])
                                print(data3[idx.logical_not()][:10])
                                print(data[:10])
                                print(data3[:10])

                        if args.profile_mode == "info" or args.profile_mode == "verbose":
                            penny_time = bench_kineto(lambda: penny_cpp.all_reduce_run(handle, penny_out),
                                                      kernel_name="all_reduce")
                            nccl_time = bench_kineto(lambda: dist.all_reduce(data), kernel_name="AllReduce_Sum_f16")
                            if custom_ar is not None:
                                custom_time = bench_kineto(lambda: custom_ar.all_reduce(data3),
                                                          kernel_name="reduce")
                            else:
                                custom_time = 1

                            if penny_time < best_time:
                                best_time = penny_time
                                best_configuration = configuration

                            if rank == 0 and args.profile_mode == "verbose":
                                print(f"{configuration=} nccl time: {nccl_time:.2f}us, "
                                      f"bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  "
                                      f"penny_time: {penny_time:.2f}us, "
                                      f"bandwidth {recv_bytes / 1e3 / penny_time :.2f} GB/s "
                                      f"custom_time: {custom_time:.2f}us, "
                                      f"bandwidth {recv_bytes / 1e3 / custom_time :.2f} GB/s"
                                      )
                        penny_cpp.all_reduce_destroy(handle)

            if rank == 0 and args.profile_mode == "info" and best_configuration is None:
                print(f"no configuration found for {num=}")
            elif rank == 0 and args.profile_mode == "info":
                print(f"{best_configuration=} nccl time: {nccl_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  "
                      f"penny_time time: {best_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / best_time :.2f} GB/s "
                      f"custom_time: {custom_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / custom_time :.2f} GB/s"
                      )
                penny_times.append(best_time)
                penny_bandwidth.append(recv_bytes / 1e3 / best_time)
        if(rank == 0):
            print(penny_bandwidth)
            print(penny_times)

    if args.profile_mode == "file" and rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            run_benchmark()
        prof.export_chrome_trace("trace_penny.json")
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
