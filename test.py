import torch
import torch.distributed as dist
import penny_cpp

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
# TODO how do I get this
local_size = world_size//2
local_rank = dist.get_rank() % local_size

torch.cuda.set_device(local_rank)
nvshmem_uid = penny_cpp.get_unique_id()
print(f"{rank=} {local_rank=} {world_size=} local uuid {nvshmem_uid}")

nvshmem_uids = [None, ] * world_size
dist.all_gather_object(nvshmem_uids, nvshmem_uid)
print("initializing with uuid", nvshmem_uids[0])
penny_cpp.init_with_uid(nvshmem_uids[0], dist.get_rank(), world_size)
# penny_cpp.run_example()

data = torch.FloatTensor([1,] * 2**18).to("cuda").to(torch.float16)
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()


penny_cpp.all_reduce(data, world_size, local_size)
print(data)
assert value == world_size, f"Expected {world_size}, got {value}"
print("PyTorch NCCL is successful!")
