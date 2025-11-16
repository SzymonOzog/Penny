# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py
from contextlib import contextmanager
from typing import Optional, Union

import torch
import os
import torch.distributed as dist
from torch.distributed import ProcessGroup
import logging 
logger = logging.getLogger(__name__)
import penny_cpp as ops
#TODO is there a way to get this?
nnodes = int(os.getenv("NNODES"))


try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
            return torch.cuda.can_device_access_peer(rank, i)
        # if not gpu_p2p_access_check(rank, i):
        #     return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (inp.storage().nbytes() -
                                   inp.storage_offset() * inp.element_size()
                                   == inp.numel() * inp.element_size())

class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 max_size=8192 * 1024,
                 symm_mem_enabled=False,
                 nvshmem_registered=False) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        if not nvshmem_registered:
            world_size = dist.get_world_size()
            local_size = world_size//nnodes
            local_rank = dist.get_rank() % local_size

            torch.cuda.set_device(local_rank)
            nvshmem_uid = ops.get_unique_id()

            nvshmem_uids = [None, ] * world_size
            dist.all_gather_object(nvshmem_uids, nvshmem_uid)
            ops.init_with_uid(nvshmem_uids[0], dist.get_rank(), world_size)

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info("Custom allreduce is disabled because "
                        "of missing custom allreduce library")
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group.")

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group) // nnodes
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size, str(CustomAllreduce._SUPPORTED_WORLD_SIZES))
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        cuda_visible_devices = None
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(world_size))

        # physical_device_id = device_ids[device.index]
        # tensor = torch.tensor([physical_device_id],
        #                       dtype=torch.int,
        #                       device="cpu")
        # gather_list = [
        #     torch.tensor([0], dtype=torch.int, device="cpu")
        #     for _ in range(world_size)
        # ]
        # dist.all_gather(gather_list, tensor, group=self.group)
        # physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        # assert current_platform.is_cuda_alike()
        # fully_connected = current_platform.is_fully_connected(
        #     physical_device_ids)
        # if world_size > 2 and not fully_connected:
        #     logger.warning(
        #         "Custom allreduce is disabled because it's not supported on"
        #         " more than two PCIe-only GPUs. To silence this warning, "
        #         "specify disable_custom_all_reduce=True explicitly.")
        #     return
        # # test P2P capability, this checks software/cudaruntime support
        # # this is expensive to compute at the first time
        # # then we cache the result
        # # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        # if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
        #     logger.warning(
        #         "Custom allreduce is disabled because your platform lacks "
        #         "GPU P2P capability or P2P test failed. To silence this "
        #         "warning, specify disable_custom_all_reduce=True explicitly.")
        #     return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(ops.meta_size() + max_size,
                                                   group=group,
                                                   uncached=True)
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device=self.device)
        self.max_size = max_size
        self.rank = rank%world_size
        self.world_size = world_size
        self.fully_connected = True
        self._ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data, self.rank,
                                       self.fully_connected)
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the 
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        world_size = dist.get_world_size(group=self.group)
        local_size = world_size//nnodes
        rank = dist.get_rank(group=self.group)

        all_data = [[None, None]
                    for _ in range(dist.get_world_size(group=self.group))]
        all_data[rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        off = (rank//local_size)*local_size

        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i],
                                       src=rank,
                                       group=self.group,
                                       device="cpu")
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data[off :  off + local_size]]  # type: ignore
        offsets = [d[1] for d in all_data[off :  off + local_size]]  # type: ignore
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False

    def all_reduce(self,
                   inp: torch.Tensor,
                   *,
                   out: torch.Tensor = None,
                   registered: bool = False):
        """Performs an out-of-place all reduce.
        
        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)

        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank],
                           self.max_size)
        return out

    def custom_all_reduce(self, input: torch.Tensor, out = None) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, out=out, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                # TODO this should have smaller shape
                return torch.empty_like(out)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, out=out, registered=False)

    def reduce_scatter(self,
                   inp: torch.Tensor,
                   *,
                   out: torch.Tensor = None,
                   registered: bool = False):
        """Performs an out-of-place reduce scatter.
        
        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)

        if registered:
            ops.reduce_scatter(self._ptr, inp, out, 0, 0)
        else:
            ops.reduce_scatter(self._ptr, inp, out, self.buffer_ptrs[self.rank],
                           self.max_size)
        return out

    def custom_reduce_scatter(self, input: torch.Tensor, out = None) -> Optional[torch.Tensor]:
        """The main reduce scatter API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.reduce_scatter(input, out=out, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.reduce_scatter(input, out=out, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(size_in_bytes: int,
                             group: Optional[ProcessGroup] = None,
                             uncached: Optional[bool] = False) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        local_size = world_size//nnodes
        rank = dist.get_rank(group=group)
        local_rank = rank%local_size
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)
        off = (rank//local_size)*local_size
        handles = handles[off :  off + local_size]

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == local_rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(pointers: list[int],
                           group: Optional[ProcessGroup] = None,
                           rank: Optional[int] = None) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
