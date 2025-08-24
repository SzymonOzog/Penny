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
__global__ void all_reduce_double_ring_kernel(scalar_t *destination, scalar_t* buffer, uint64_t* signal,
        const int nnodes, const int gpus_per_node, const int packet_size) 
{

    const int my_node = nvshmem_my_pe()/gpus_per_node;
    const int local_rank = nvshmem_my_pe()%gpus_per_node;
    int stage = 0;
    {
        const int scale = gpus_per_node/nnodes;
        const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
        const uint64_t block_size = blockDim.x * packet_size;
        const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);
        for (int part = 0; part < scale; part++)
        {
            const uint64_t part_off = chunk_off * nnodes * part;

            int send_peer = (nvshmem_my_pe()+gpus_per_node) % nvshmem_n_pes();
            int recv_peer = (nvshmem_n_pes() + nvshmem_my_pe()-gpus_per_node) % nvshmem_n_pes();

            for (int chunk = 0; chunk < nnodes - 1; chunk++)
            {
                int send_chunk = (nnodes + my_node - chunk) % nnodes;
                int recv_chunk = (nnodes + my_node - chunk - 1) % nnodes;

                nvshmem_signal_wait_until(signal + gridDim.x + blockIdx.x, NVSHMEM_CMP_GE, stage);

                nvshmemx_putmem_block(destination + off, buffer + part_off  + send_chunk*chunk_off + off, block_size, send_peer);

                nvshmem_fence();
                __syncthreads();
                stage++;

                if (threadIdx.x == 0)
                {
                    nvshmemx_signal_op(signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);
                }
                nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, stage);

                for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
                {
                    float res = float(buffer[recv_chunk*chunk_off + off + part_off  + i]) + float(destination[off + i]);
                    buffer[recv_chunk*chunk_off + off + part_off  + i] = res;
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    nvshmemx_signal_op(signal + gridDim.x + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, recv_peer);
                }
            }

            for (int chunk = 0; chunk < nnodes - 1; chunk++)
            {
                int send_chunk = (nnodes + my_node - chunk + 1) % nnodes;
                int recv_chunk = (nnodes + my_node - chunk) % nnodes;

                nvshmem_signal_wait_until(signal + gridDim.x + blockIdx.x, NVSHMEM_CMP_GE, stage);
                nvshmemx_putmem_block(destination + off, buffer  + part_off + send_chunk*chunk_off + off, block_size, send_peer);

                nvshmem_fence();
                __syncthreads();
                stage++;

                if (threadIdx.x == 0)
                {
                    nvshmemx_signal_op(signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);
                }
                nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, stage);

                for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
                {
                    buffer[recv_chunk*chunk_off + part_off + off + i] = destination[off + i];
                }
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    nvshmemx_signal_op(signal + gridDim.x + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, recv_peer);
                }
            }
        }
        signal += gridDim.x*2;
        destination += chunk_off * nnodes;
    }

    stage = 0;

    {
        const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
        const uint64_t block_size = blockDim.x * packet_size;
        const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);

        int send_peer = my_node*gpus_per_node + (local_rank+1) % gpus_per_node;
        int recv_peer = my_node*gpus_per_node + (gpus_per_node + local_rank-1) % gpus_per_node;

        for (int chunk = 0; chunk < gpus_per_node - 1; chunk++)
        {
            int send_chunk = (gpus_per_node + local_rank - chunk) % gpus_per_node;
            int recv_chunk = (gpus_per_node + local_rank - chunk - 1) % gpus_per_node;

            nvshmem_signal_wait_until(signal + gridDim.x + blockIdx.x, NVSHMEM_CMP_GE, stage);

            nvshmemx_putmem_block(destination + off, buffer + send_chunk*chunk_off + off, block_size, send_peer);

            nvshmem_fence();
            __syncthreads();
            stage++;

            if (threadIdx.x == 0)
            {
                nvshmemx_signal_op(signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);
            }
            nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, stage);

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

        for (int chunk = 0; chunk < gpus_per_node - 1; chunk++)
        {
            int send_chunk = (gpus_per_node + local_rank - chunk + 1) % gpus_per_node;
            int recv_chunk = (gpus_per_node + local_rank - chunk) % gpus_per_node;

            nvshmem_signal_wait_until(signal + gridDim.x + blockIdx.x, NVSHMEM_CMP_GE, stage);
            nvshmemx_putmem_block(destination + off, buffer + send_chunk*chunk_off + off, block_size, send_peer);

            nvshmem_fence();
            __syncthreads();
            stage++;

            if (threadIdx.x == 0)
            {
                nvshmemx_signal_op(signal + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, send_peer);
            }
            nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_GE, stage);

            for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
            {
                buffer[recv_chunk*chunk_off + off + i] = destination[off + i];
            }
            __syncthreads();
            if (threadIdx.x == 0 && chunk < gpus_per_node - 1)
            {
                nvshmemx_signal_op(signal + gridDim.x + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, recv_peer);
            }
        }
    }
}

void all_reduce_double_ring(half* buffer, int numel, int packet_size, int block_size, int nnodes, cudaStream_t stream) 
{
    int gpus_per_node = nvshmem_n_pes() / nnodes;
    half *destination = (half *) nvshmem_malloc((numel/nnodes + numel/gpus_per_node) * sizeof(half));

    nvshmemx_buffer_register(buffer, numel * sizeof(half));
    
    const uint32_t grid_size = std::ceil(numel*sizeof(half) / float(packet_size*block_size*std::max(nnodes, gpus_per_node)));

    const int signal_size = grid_size * 4 * sizeof(uint64_t);
    uint64_t *signal = (uint64_t *) nvshmem_malloc(signal_size);
    cudaMemset(signal, 0, signal_size);
    
    //sync the memset before running kernel
    nvshmemx_barrier_all_on_stream(stream);

    all_reduce_double_ring_kernel<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer),
            signal,
            nnodes,
            gpus_per_node,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);
    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer);
    nvshmem_free(destination);
}
