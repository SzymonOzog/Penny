#include "bootstrap_device_host/nvshmem_uniqueid.h"
#include "common.h"
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
#include <cuda_fp16.h>
#include <cstdint>

void init_with_uid(pybind11::bytearray uid_py, int rank, int world_size)
{
    auto uid_str = uid_py.cast<std::string>();

    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_str.c_str(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_init_attr_t attr;
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

// Ring allreduce object lifecycle API
void* create_all_reduce(half* buffer, int numel, int packet_size, int block_size, int nnodes, int routes, AlgoType algo_type, cudaStream_t stream);
void destroy_all_reduce(void* all_reduce_obj);
void all_reduce(void* all_reduce_obj, half* output, cudaStream_t stream);
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
    m.def("exchange", &exchange);

    m.def("nvshmem_register", [] (torch::Tensor& buffer){
            nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * buffer.element_size());
            });
    m.def("nvshmem_unregister", [] (torch::Tensor& buffer){
            nvshmemx_buffer_unregister(buffer.data_ptr());
            });
    m.def("all_reduce_create", [](torch::Tensor& buffer, int packet_size, int block_size, int nnodes, int routes, int algo) {
            auto stream = at::cuda::getCurrentCUDAStream();
            assert(algo >= 0);
            assert(algo < 4);
            AlgoType t;
            if (algo == 0)
            {
                t = AlgoType::ring_standard;
            }
            else if (algo == 1)
            {
                t = AlgoType::ring_simple;
            }
            else if (algo == 2)
            {
                t = AlgoType::oneshot;
            }
            else if (algo == 3)
            {
                t = AlgoType::twoshot;
            }
            void* handle = create_all_reduce(
                    static_cast<half*>(buffer.data_ptr()),
                    buffer.numel(),
                    packet_size,
                    block_size,
                    nnodes,
                    routes,
                    t,
                    stream
                    );
            return reinterpret_cast<uintptr_t>(handle);
    });
    m.def("all_reduce_run", [](uintptr_t handle, torch::Tensor& output) {
            auto stream = at::cuda::getCurrentCUDAStream();
            all_reduce(reinterpret_cast<void*>(handle), static_cast<half*>(output.data_ptr()), stream);
            });
    m.def("all_reduce_destroy", [](uintptr_t handle) {
            destroy_all_reduce(reinterpret_cast<void*>(handle));
            });
}
