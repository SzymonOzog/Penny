#include "device_host_transport/nvshmem_common_transport.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

template <typename scalar_t>
__global__ void all_reduce_ring_kernel(scalar_t *destination, scalar_t* buffer, uint64_t* signal, int send_peer, int recv_peer, int packet_size) 
{
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;
    const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);

    int stage = 1;
    for (int chunk = 0; chunk < nvshmem_n_pes() - 1; chunk++)
    {
        int send_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk) % nvshmem_n_pes();
        int recv_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk - 1) % nvshmem_n_pes();

        nvshmemx_putmem_signal_block(destination + off + chunk*chunk_off, buffer + send_chunk*chunk_off + off, block_size,
                signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);

        nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, stage);

        for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
        {
            float res = float(buffer[recv_chunk*chunk_off + off + i]) + float(destination[off+ chunk*chunk_off + i]);
            buffer[recv_chunk*chunk_off + off + i] = res;
        }
        stage++;
    }

    destination += nvshmem_n_pes() * chunk_off;

    for (int chunk = 0; chunk < nvshmem_n_pes() - 1; chunk++)
    {
        int send_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk + 1) % nvshmem_n_pes();
        int recv_chunk = (nvshmem_n_pes() + nvshmem_my_pe() - chunk) % nvshmem_n_pes();

        nvshmemx_putmem_signal_block(destination + off + chunk*chunk_off, buffer + send_chunk*chunk_off + off, block_size,
                signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);

        nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, stage);

        for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
        {
            buffer[recv_chunk*chunk_off + off + i] = destination[off + chunk*chunk_off + i];
        }
        stage++;
    }
}

void all_reduce_ring(half* buffer, int numel, int packet_size, int block_size, cudaStream_t stream) 
{
    // Can we reduce te size of this buffer?
    half *destination = (half *) nvshmem_malloc(2 * numel * sizeof(half));

    nvshmemx_buffer_register(buffer, numel * sizeof(half));
    
    const uint32_t grid_size = std::ceil(numel*sizeof(half) / float(packet_size*block_size*nvshmem_n_pes()));

    uint64_t *signal = (uint64_t *) nvshmem_malloc(grid_size * 2 * sizeof(uint64_t));
    cudaMemset(signal, 0, grid_size * 2 * sizeof(uint64_t));
    
    //sync the memset before running kernel
    nvshmemx_barrier_all_on_stream(stream);

    int send_peer = (nvshmem_my_pe()+1) % nvshmem_n_pes();
    int recv_peer = (nvshmem_n_pes() + nvshmem_my_pe()-1) % nvshmem_n_pes();

    all_reduce_ring_kernel<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer),
            signal,
            send_peer,
            recv_peer,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);
    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer);
    nvshmem_free(destination);
}
