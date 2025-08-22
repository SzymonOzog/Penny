#include "device_host_transport/nvshmem_common_transport.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <cstdint>
#include <cstdio>
#include <pybind11/functional.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

template <typename scalar_t>
__global__ void all_reduce(scalar_t *destination, scalar_t* buffer, uint64_t* signal, int send_peer, int recv_peer, int packet_size) 
{
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;
    const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);

    for (int chunk = 0; chunk < nvshmem_n_pes() - 1; chunk++)
    {
        int send_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk) % nvshmem_n_pes();
        int recv_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk - 1) % nvshmem_n_pes();

        nvshmem_signal_wait_until(signal + gridDim.x + blockIdx.x, NVSHMEM_CMP_GE, chunk);

        nvshmemx_putmem_block(destination + off, buffer + send_chunk*chunk_off + off, block_size, send_peer);

        nvshmem_fence();
        __syncthreads();

        if (threadIdx.x == 0)
        {
            nvshmemx_signal_op(signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);
        }
        nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, chunk+1);

        for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
        {
            float res = float(buffer[recv_chunk*chunk_off + off + i]) + float(destination[off + i]);
            buffer[recv_chunk*chunk_off + off + i] = res;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            nvshmemx_signal_op(signal + gridDim.x + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, recv_peer);
        }
    }

    for (int chunk = 0; chunk < nvshmem_n_pes() - 1; chunk++)
    {
        int send_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk + 1) % nvshmem_n_pes();
        int recv_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk) % nvshmem_n_pes();

        nvshmem_signal_wait_until(signal + gridDim.x + blockIdx.x, NVSHMEM_CMP_GE, nvshmem_n_pes() - 1 + chunk);
        nvshmemx_putmem_block(destination + off, buffer + send_chunk*chunk_off + off, block_size, send_peer);

        nvshmem_fence();
        __syncthreads();

        if (threadIdx.x == 0)
        {
            nvshmemx_signal_op(signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);
        }
        nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, nvshmem_n_pes()+chunk);

        for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
        {
            buffer[recv_chunk*chunk_off + off + i] = destination[off + i];
        }
        __syncthreads();
        if (threadIdx.x == 0 && chunk < nvshmem_n_pes() - 1)
        {
            nvshmemx_signal_op(signal + gridDim.x + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, recv_peer);
        }
    }
}

void all_reduce(torch::Tensor& buffer, int packet_size, int block_size) 
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    half *destination = (half *) nvshmem_malloc(buffer.numel() * sizeof(half) / nvshmem_n_pes());

    nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * sizeof(half));
    
    const uint32_t grid_size = std::ceil(buffer.numel()*sizeof(half) / float(packet_size*block_size*nvshmem_n_pes()));

    uint64_t *signal = (uint64_t *) nvshmem_malloc(grid_size * 2 * sizeof(uint64_t));
    cudaMemset(signal, 0, grid_size * 2 * sizeof(uint64_t));
    //sync the memset before running kernel
    nvshmemx_barrier_all_on_stream(stream);

    int send_peer = (nvshmem_my_pe()+1) % nvshmem_n_pes();
    int recv_peer = (nvshmem_n_pes() + nvshmem_my_pe()-1) % nvshmem_n_pes();

    all_reduce<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer.data_ptr()),
            signal,
            send_peer,
            recv_peer,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);
    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer.data_ptr());
    nvshmem_free(destination);
}
