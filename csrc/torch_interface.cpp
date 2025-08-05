#include "bootstrap_device_host/nvshmem_uniqueid.h"
#include <pybind11/functional.h>
#include <torch/python.h>
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

void run_example();

void all_reduce(torch::Tensor& buffer, int world_size, int local_size);

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
    m.def("run_example", &run_example);
    m.def("all_reduce", &all_reduce);
    m.def("exchange", &exchange);
}
