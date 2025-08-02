#include <iostream>
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <pybind11/functional.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define PACKET_SIZE 16
#define BLOCK_SIZE 32

template <typename scalar_t>
__global__ void simple_shift(scalar_t *destination, scalar_t* buffer, int peer) 
{
    const uint32_t off = (blockIdx.x * blockDim.x + threadIdx.x) * PACKET_SIZE/sizeof(scalar_t);

    nvshmem_putmem(destination + off, buffer + off, PACKET_SIZE, peer);

    for (int i = 0; i < PACKET_SIZE/sizeof(scalar_t); i++)
    {
        buffer[i + off] += destination[i + off];
    }
}

void all_reduce(torch::Tensor& buffer, int world_size, int local_size) 
{
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    half *destination = (half *) nvshmem_malloc(buffer.numel() * sizeof(half));
    nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * sizeof(half));
    
    const uint32_t grid_size = buffer.numel()*sizeof(half) / (PACKET_SIZE*BLOCK_SIZE);
    std::cout<<(nvshmem_my_pe() + local_size) % world_size<< " dst, grid " << grid_size<<std::endl;

    simple_shift<<<grid_size, BLOCK_SIZE, 0, stream>>>(destination, static_cast<half*>(buffer.data_ptr()),
            (nvshmem_my_pe() + local_size) % world_size);
    nvshmemx_barrier_all_on_stream(stream);

    cudaStreamSynchronize(stream);

    nvshmem_free(destination);
    nvshmem_finalize();
}
