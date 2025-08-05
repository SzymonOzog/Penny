#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <pybind11/functional.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

template <typename scalar_t>
__global__ void exchange(scalar_t *destination, scalar_t* buffer, int peer, int packet_size) 
{
    const uint32_t off = (blockIdx.x * blockDim.x + threadIdx.x) * packet_size/sizeof(scalar_t);

    // nvshmem_putmem(destination + off, buffer + off, PACKET_SIZE, peer);
    nvshmemx_putmem_block(destination + off, buffer + off, packet_size*blockDim.x, peer);
    // nvshmemx_putmem_warp(destination + off, buffer + off, PACKET_SIZE*32, peer);
}

void exchange(torch::Tensor& buffer, int packet_size, int block_size, int peer) 
{
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    half *destination = (half *) nvshmem_malloc(buffer.numel() * sizeof(half));
    nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * sizeof(half));
    
    const uint32_t grid_size = buffer.numel()*sizeof(half) / (packet_size*block_size);

    exchange<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer.data_ptr()),
            peer,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(buffer.data_ptr(), (void*)destination, buffer.numel() * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer.data_ptr());
    nvshmem_free(destination);
}
