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
    parser.add_argument("--atol", type=float, default=1e-1,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--profile-mode", type=str, default="info",
                        help="How do you want to profile (info|verbose|file|none)")
    parser.add_argument("--range", type=int, default=1,
                        help="How many sizes to profile")
    parser.add_argument("--start-pow", type=int, default=28,
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
    from penny.custom_all_reduce import CustomAllreduce
    group = dist.new_group(list(range(world_size)), backend="gloo") 
    custom_ar = CustomAllreduce(group, device=local_rank, max_size = 2**(args.start_pow + args.range), nvshmem_registered=True)
    # float16
    elem_size = 2
    penny_bandwidth = []
    penny_times = []

    def run_benchmark():
        for pow in range(args.start_pow, args.start_pow + args.range):
            num = 2 ** pow
            in_num = num//world_size
            data = torch.empty(in_num, device="cuda", dtype=torch.float16).normal_(mean=0, std=1) + 1
            out = torch.empty(num, device="cuda", dtype=torch.float16)
            custom_out = torch.empty(num, device="cuda", dtype=torch.float16)

            data = torch.ones(in_num, device="cuda", dtype=torch.float16) * rank

            data2 = data.clone()
            recv_bytes = 2 * out.nelement() * out.element_size()


            for _ in range(args.num_tests):
                data2.copy_(data)

                dist.all_gather_into_tensor(out, data)
                custom_out = custom_ar.all_gather(data2, out=custom_out)


                if not torch.allclose(out, custom_out, atol=args.atol, rtol=args.rtol):
                    idx = torch.isclose(out, custom_out, atol=args.atol, rtol=args.rtol)
                    num_missed = idx.logical_not().sum() / idx.nelement()
                    print(f"failed {rank=}, {num_missed=} {out.mean()}, {custom_out.mean()}")
                    # print(out[idx.logical_not()][:10])
                    # print(custom_out[idx.logical_not()][:10])
                    # print(out[:10])
                    # print(custom_out[:10])

            if args.profile_mode == "info" or args.profile_mode == "verbose":

                nccl_time = bench_kineto(lambda: dist.all_gather_into_tensor(out, data), kernel_name="AllGather")
                custom_time = bench_kineto(lambda: custom_ar.all_gather(data2, out=custom_out),
                                          kernel_name="cross_device")

                if rank == 0 and args.profile_mode == "verbose":
                    print(f"nccl time: {nccl_time:.2f}us, "
                          f"bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  "
                          f"penny_time: {custom_time:.2f}us, "
                          f"bandwidth {recv_bytes / 1e3 / custom_time :.2f} GB/s"
                          )

            if rank == 0 and args.profile_mode == "info":
                print(f"nccl time: {nccl_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / nccl_time :.2f} GB/s  "
                      f"penny_time: {custom_time:.2f}us, "
                      f"bandwidth {recv_bytes / 1e3 / custom_time :.2f} GB/s"
                      )
                penny_times.append(custom_time)
                penny_bandwidth.append(recv_bytes / 1e3 / custom_time)
        if(rank == 0):
            print(penny_bandwidth)
            print(penny_times)

    if rank == 0:
        print("running benchmark")
    if args.profile_mode == "file" and rank == 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            run_benchmark()
        prof.export_chrome_trace("trace_penny.json")
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
