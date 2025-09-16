#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <pybind11/functional.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

template <typename scalar_t>
__global__ void exchange(scalar_t *destination, scalar_t* buffer, uint64_t* signal, int peer, int packet_size) 
{
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;

    constexpr uint64_t SIG_SYNC = 1;
    nvshmemx_putmem_signal_block(destination + off, buffer + off, block_size, signal + blockIdx.x, SIG_SYNC, NVSHMEM_SIGNAL_SET, peer);
    // nvshmemx_putmem_block(destination + off, buffer + off, block_size, peer);
    // nvshmem_fence();
    // __syncthreads();
    //
    // if (threadIdx.x == 0)
    // {
    //     nvshmemx_signal_op(signal + blockIdx.x, SIG_SYNC, NVSHMEM_SIGNAL_SET, peer);
    // }
    if (threadIdx.x == 0)
        nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_EQ, SIG_SYNC);
    __syncthreads();

    for (int i = threadIdx.x; i < block_size/(sizeof(float4)); i += blockDim.x)
        reinterpret_cast<float4*>(buffer +off)[i] = reinterpret_cast<float4*>(destination + off)[i];
}

void exchange(torch::Tensor& buffer, int packet_size, int block_size, int peer) 
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    half *destination = (half *) nvshmem_malloc(buffer.numel() * sizeof(half));

    nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * sizeof(half));
    
    const uint32_t grid_size = std::ceil(buffer.numel()*sizeof(half) / float(packet_size*block_size));

    uint64_t *signal = (uint64_t *) nvshmem_malloc(grid_size * sizeof(uint64_t));
    cudaMemset(signal, 0, grid_size*sizeof(uint64_t));

    //sync the memset before running kernel
    nvshmemx_barrier_all_on_stream(stream);

    exchange<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer.data_ptr()),
            signal,
            peer,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);
    // cudaMemcpyAsync(buffer.data_ptr(), (void*)destination, buffer.numel() * sizeof(half), cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer.data_ptr());
    nvshmem_free(destination);
}
