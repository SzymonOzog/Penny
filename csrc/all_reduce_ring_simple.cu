#include "device_host_transport/nvshmem_common_transport.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "common.h"


template <typename scalar_t>
__global__ void all_reduce_simple_ring_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    int send_peer = (pe+1) % n_pes;
    int ring_pos = pe;

    uint64_t* local_signal = signal + blockIdx.x;
    int send_stage = stage;
    int recv_stage = stage;

    if (ring_pos == 0)
    {
        nvshmemx_putmem_signal_nbi_block(destination + off,
                buffer + off,
                block_size, local_signal, send_stage, NVSHMEM_SIGNAL_SET, send_peer);
        send_stage++;
    }
    else 
    {
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_EQ, recv_stage);
        __syncthreads();
        recv_stage++;

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + off)[i];
            P dst = reinterpret_cast<P*>(destination + off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + off)[i] = res;
        }
        nvshmemx_putmem_signal_nbi_block(destination + off,
                buffer + off,
                block_size, local_signal, send_stage, NVSHMEM_SIGNAL_SET, send_peer);
        send_stage++;
    }


    if (ring_pos != n_pes - 1)
    {
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_EQ, recv_stage);
        __syncthreads();

       if (ring_pos < n_pes - 2)
            nvshmemx_putmem_signal_nbi_block(destination + off,
                    destination + off,
                    block_size, local_signal, send_stage, NVSHMEM_SIGNAL_SET, send_peer);

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + off)[i] =
                reinterpret_cast<P*>(destination + off)[i];
        }
    }
}

AllReduceRingSimple::AllReduceRingSimple(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel, numel, packet_size, block_size, nnodes,
            std::ceil(numel*sizeof(half) / float(packet_size*block_size*routes)) * routes, stream)
{
    grid_dim.x = std::ceil(numel*sizeof(half) / float(packet_size*block_size*routes));
    grid_dim.y = routes;
}
void AllReduceRingSimple::run(cudaStream_t stream)
{
    all_reduce_simple_ring_kernel<half><<<grid_dim, block_dim, 0, stream>>>(
            destination,
            static_cast<half*>(buffer),
            signal,
            packet_size,
            gpus_per_node,
            stage
            );
    stage+=2;
}
