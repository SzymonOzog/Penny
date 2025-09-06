#include "bootstrap_device_host/nvshmem_uniqueid.h"
#include <pybind11/functional.h>
#include <torch/python.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <vector>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string>

void init_with_uid(pybind11::bytearray uid_py, int rank, int world_size)
{
    auto uid_str = uid_py.cast<std::string>();

    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_str.c_str(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_init_attr_t attr;
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void all_reduce_ring(half* buffer, int numel, int packet_size, int block_size, int nnodes, cudaStream_t stream);
void all_reduce_tree(half* buffer, int numel, int packet_size, int block_size, int nnodes, cudaStream_t stream);
void all_reduce_double_ring(half* buffer, int numel, int packet_size, int block_size, int nnodes, cudaStream_t stream);

void all_reduce_launcher(torch::Tensor& buffer, int packet_size, int block_size, int nnodes, int algo)
{
    if (algo == 0)
    {
    all_reduce_ring(static_cast<half*>(buffer.data_ptr()),
            buffer.numel(),
            packet_size,
            block_size,
            nnodes,
            at::cuda::getCurrentCUDAStream()
            );
    }
    else if (algo == 1)
    {
        all_reduce_tree(static_cast<half*>(buffer.data_ptr()),
                buffer.numel(),
                packet_size,
                block_size,
                nnodes,
                at::cuda::getCurrentCUDAStream()
                );
    }
    // all_reduce_double_ring(static_cast<half*>(buffer.data_ptr()),
    //         buffer.numel(),
    //         packet_size,
    //         block_size,
    //         nnodes,
    //         at::cuda::getCurrentCUDAStream()
    //         );
}

void exchange(torch::Tensor& buffer, int packet_size, int block_size, int peer);

pybind11::bytearray get_nvshmem_unique_id() 
{
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return {reinterpret_cast<const char*>(result.data()), result.size()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_unique_id", &get_nvshmem_unique_id);
    m.def("init_with_uid", &init_with_uid);
    m.def("all_reduce", &all_reduce_launcher);
    m.def("exchange", &exchange);
}
