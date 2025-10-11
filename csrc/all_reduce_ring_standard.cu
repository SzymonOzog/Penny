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

template <typename scalar_t, bool INTERNODE>
__global__ void all_reduce_ring_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint64_t base_off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint32_t block_size = blockDim.x * packet_size;
    const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint32_t ring_id = blockIdx.y;
    const uint64_t ring_off = ring_id * chunk_off * nvshmem_n_pes();
    const uint64_t off = base_off + ring_off;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();


    int send_peer;
    int recv_peer;
    int ring_pos;

    if constexpr (INTERNODE)
    {
    // TODO this is currently a hack to get the ring position, since it changes a lot
    // it's easier to find it than to derive an expression for it
        int curr_pe = -1;
        send_peer = 0;
        ring_pos = -1;
        while (curr_pe != pe)
        {
            curr_pe = send_peer;
            int curr_node = curr_pe/gpus_per_node;
            int curr_rank = curr_pe%gpus_per_node;
            if (curr_rank == (ring_id/2)*2)
            {
                if (curr_node%2 == 1)
                {
                    send_peer = curr_node * gpus_per_node + (gpus_per_node + curr_rank - 1) % gpus_per_node;
                    recv_peer = (n_pes + curr_pe - gpus_per_node) % n_pes;
                }
                else
                {
                    send_peer = (n_pes + curr_pe + gpus_per_node) % n_pes;
                    recv_peer = curr_node * gpus_per_node + (gpus_per_node + curr_rank - 1) % gpus_per_node;
                }
            }
            else if (curr_rank == (ring_id/2)*2 + 1)
            {
                if (curr_node%2 == 1)
                {
                    send_peer = (n_pes + curr_pe + gpus_per_node) % n_pes;
                    recv_peer = curr_node * gpus_per_node + (curr_rank + 1) % gpus_per_node;
                }
                else
                {
                    send_peer = curr_node * gpus_per_node + (curr_rank + 1) % gpus_per_node;
                    recv_peer = (n_pes + curr_pe - gpus_per_node) % n_pes;
                }
            }
            else
            {
                send_peer = curr_node*gpus_per_node + (curr_rank+1) % gpus_per_node;
                recv_peer = curr_node*gpus_per_node + (gpus_per_node + curr_rank-1) % gpus_per_node;
                if (curr_node%2 == 1)
                    swap_cu(send_peer, recv_peer);
            }
            ring_pos++;
        }
    }
    else 
    {
        send_peer = (pe+1) % n_pes;
        recv_peer = (n_pes + pe-1) % n_pes;
        ring_pos = pe;
    }

    int send_chunk = ring_pos % n_pes;
    int recv_chunk = (n_pes + ring_pos-1) % n_pes;
    if(ring_id%2 == 1 && INTERNODE)
    {
        swap_cu(send_chunk, recv_chunk);
        swap_cu(send_peer, recv_peer);
    }

    uint64_t* local_signal = signal + blockIdx.x + blockIdx.y * gridDim.x;
    for (int chunk = 0; chunk < n_pes - 1; chunk++)
    {
        nvshmemx_putmem_signal_nbi_block(destination + off + send_chunk*chunk_off,
                buffer + send_chunk*chunk_off + off,
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i];
            P dst = reinterpret_cast<P*>(destination + off+ recv_chunk*chunk_off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] = res;
        }
        stage++;
        send_chunk = recv_chunk;
        if(ring_id%2 == 1 && INTERNODE)
            recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
        else
            recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }

    for (int chunk = 0; chunk < n_pes - 1; chunk++) 
    {
        nvshmemx_putmem_signal_nbi_block(destination + off + send_chunk*chunk_off,
                buffer + send_chunk*chunk_off + off,
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] =
                reinterpret_cast<P*>(destination + off+ recv_chunk*chunk_off)[i];
        }
        stage++;
        send_chunk = recv_chunk;
        if(ring_id%2 == 1 && INTERNODE)
            recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
        else
            recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }
}

AllReduceRingStandard::AllReduceRingStandard(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel, numel, packet_size, block_size, nnodes,
            std::ceil(numel*sizeof(half) / float(packet_size*block_size*nvshmem_n_pes()*routes)) * routes, stream),
    internode(nnodes > 1)
{
    grid_dim.x = std::ceil(numel*sizeof(half) / float(packet_size*block_size*nvshmem_n_pes()*routes));
    grid_dim.y = routes;
}
void AllReduceRingStandard::run(cudaStream_t stream)
{
    if(internode)
    {
        all_reduce_ring_kernel<half, true><<<grid_dim, block_dim, 0, stream>>>(
                destination,
                static_cast<half*>(buffer),
                signal,
                packet_size,
                gpus_per_node,
                stage
                );
    }
    else 
    {
        all_reduce_ring_kernel<half, false><<<grid_dim, block_dim, 0, stream>>>(
                destination,
                static_cast<half*>(buffer),
                signal,
                packet_size,
                gpus_per_node,
                stage
                );
    }
    stage += 2*(nvshmem_n_pes()-1);
}
