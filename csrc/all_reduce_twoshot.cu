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
#include <cuda/atomic>
#include <cooperative_groups.h>
#include "common.h"


__device__ int buffer_lock_[4] = {0};

template <typename scalar_t, int N_PES=8>
__global__ void all_reduce_twoshot_kernel(scalar_t* __restrict__ destination, scalar_t*  buffer, scalar_t* __restrict__ output, uint64_t* signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    const uint32_t block_size = blockDim.x * packet_size;

    const uint32_t pe_off = block_size/sizeof(scalar_t);

    uint32_t write_chunk = blockIdx.x;

    const bool pr = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && pe == 0;
    if (write_chunk != pe && blockIdx.y == 0)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                    buffer + write_chunk*pe_off,
                    block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }

    for(int tid = 0; tid<N_PES; tid++)
    {
        if (threadIdx.x == tid && tid != pe)
        {
            nvshmem_signal_wait_until(signal+tid, NVSHMEM_CMP_EQ, stage);
        }
    }

    __syncthreads();
    const uint32_t reduce_size = block_size/(N_PES*gridDim.y);
    const uint32_t reduce_off = (blockIdx.y*gridDim.x + blockIdx.x)*reduce_size/sizeof(scalar_t);

    for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(buffer + pe*pe_off + reduce_off)[i];
        for (int recv_pe = 0; recv_pe < N_PES; recv_pe++)
        {
            if(recv_pe == pe)
                continue;
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off + reduce_off)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(output + pe * pe_off + reduce_off)[i] = res;
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        nvshmemx_signal_op(signal+n_pes+pe, 1, NVSHMEM_SIGNAL_ADD, pe);
    }
    if (write_chunk == pe)
    {
        return;
    }
    if (threadIdx.x == 0)
    {
         
        nvshmem_signal_wait_until(signal+n_pes+pe, NVSHMEM_CMP_EQ, stage*N_PES*gridDim.y);
    }

    __syncthreads();
    __threadfence_system();

    if(blockIdx.y == 0)
    {
        nvshmemx_putmem_signal_nbi_block(destination + (n_pes+pe)*pe_off,
                output + pe*pe_off,
                block_size, signal+n_pes+pe, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }

    if (threadIdx.x == 0)
        nvshmem_signal_wait_until(signal+n_pes+write_chunk, NVSHMEM_CMP_EQ, stage);
    __syncthreads();

    const uint32_t write_size = block_size/gridDim.y;
    const uint32_t write_off = (blockIdx.y*write_size)/sizeof(scalar_t);

    for (int i = threadIdx.x; i < write_size/(sizeof(P)); i += blockDim.x)
    {
        reinterpret_cast<P*>(output + write_chunk*pe_off + write_off)[i] =
            reinterpret_cast<P*>(destination + (n_pes+write_chunk)*pe_off + write_off)[i];
    }
}

AllReduceTwoShot::AllReduceTwoShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel, numel*2, packet_size, block_size, nnodes,
            nvshmem_n_pes()*2, stream)
{
    assert(packet_size*block_size  == numel * sizeof(half) / nvshmem_n_pes());
    assert(block_size*packet_size/(nvshmem_n_pes()*routes) > 0);
    assert(((block_size*packet_size)/(nvshmem_n_pes()*routes))%16 == 0);
    grid_dim.x = nvshmem_n_pes();
    grid_dim.y = routes;
}
void AllReduceTwoShot::run(half* output, cudaStream_t stream)
{
    all_reduce_twoshot_kernel<half><<<grid_dim, block_dim, 0, stream>>>(
            destination,
            static_cast<half*>(buffer),
            output,
            signal,
            packet_size,
            gpus_per_node,
            stage
            );
    stage+=1;
}
