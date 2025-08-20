
import torch
import torch.distributed as dist
import penny_cpp
from penny.utils import bench_kineto, initialize_distributed
from torch.profiler import profile, ProfilerActivity
from triton.testing import do_bench

initialize_distributed()
rank = dist.get_rank()
world_size = dist.get_world_size()
nnodes = os.getenv("NNODES")
local_size = world_size//nnodes
local_rank = dist.get_rank() % local_size

num = 2**18
for mode in ["internode", "intranode"]:
    for packet_size in [2, 8, 16, 32, 64, 512]:
        for block_size in [32, 256, 512, 1024]:
            if num%(packet_size * block_size) != 0:
                continue

            if mode == "internode":
                src = (rank + local_size)%world_size
            else: 
                if local_rank % 2 == 0:
                    src = (rank//local_size) * local_size + (local_rank + 1)%local_size
                else:
                    src = (rank//local_size) * local_size + (local_rank - 1)%local_size
            configuration = f"{mode=} {packet_size=} {block_size=} {src=}"

            data = torch.randn(num).to("cuda").to(torch.float16)
            data2 = data.clone()
            data_r = torch.FloatTensor([0,] * num).to("cuda").to(torch.float16)

            ops = [dist.P2POp(dist.isend, data, src),
                   dist.P2POp(dist.irecv, data_r, src)]
            if mode == "internode" and rank >= world_size//(2):
                ops = list(reversed(ops))
            if mode == "intranode" and rank%2:
                ops = list(reversed(ops))

            dist.batch_isend_irecv(ops)
            torch.cuda.synchronize()

            penny_cpp.exchange(data2, packet_size, block_size, src)
            if not (data_r==data2).all():
                if rank == 0:
                    idx = data_r == data2
                    num_missed =  idx.logical_not().sum() / idx.nelement()
                    print(f"failed {configuration=} {rank=}, {num_missed=}")

            nccl_time =  bench_kineto(lambda: dist.batch_isend_irecv(ops), kernel_names="nccl")
            penny_time = bench_kineto(lambda: penny_cpp.exchange(data2, packet_size, block_size, src), "exchange")


            if rank == 0:
                recv_bytes = data2.nelement() * data2.element_size() * 2
                print(f"{configuration=} nccl time: {nccl_time*1e6:.2f}us, bandwidth {recv_bytes / 1e9 / nccl_time :.2f} GB/s  penny_time: {penny_time*1e6:.2f}, bandwidth {recv_bytes / 1e9 / penny_time :.2f} GB")
