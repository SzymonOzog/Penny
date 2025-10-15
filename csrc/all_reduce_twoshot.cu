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

template <typename scalar_t>
__global__ void all_reduce_twoshot_kernel(scalar_t* __restrict__ destination, scalar_t*  buffer, uint64_t* signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    const uint32_t block_size = blockDim.x * packet_size;

    const uint32_t pe_off = block_size/sizeof(scalar_t);

    auto reduce = [&](scalar_t* s1_p, scalar_t* s2_p, scalar_t* dst)
    {
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P src1 = reinterpret_cast<P*>(s1_p)[i];
            P src2 = reinterpret_cast<P*>(s2_p)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(src1.data[j]) + float(src2.data[j]);
            reinterpret_cast<P*>(dst)[i] = res;
        }
    };
    uint32_t write_chunk = blockIdx.x;
    //Makes reduciton easier
    if (blockIdx.x == 0)
        write_chunk = pe;
    if( blockIdx.x == pe)
        write_chunk = 0;

    auto wait_send_write = [&]()
    {
        if (threadIdx.x == 0)
        {
            nvshmem_signal_wait_until(signal+n_pes+pe, NVSHMEM_CMP_EQ, stage);
        }
        __syncthreads();
        __threadfence();
        nvshmemx_putmem_signal_nbi_block(destination + (n_pes+pe)*pe_off,
                buffer + pe*pe_off,
                block_size, signal+n_pes+pe, stage, NVSHMEM_SIGNAL_SET, write_chunk);
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(signal+n_pes+write_chunk, NVSHMEM_CMP_EQ, stage);
        __syncthreads();
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + write_chunk*pe_off)[i] =
                reinterpret_cast<P*>(destination + (n_pes+write_chunk)*pe_off)[i];
        }
    };

    if (write_chunk != pe)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                    buffer + write_chunk*pe_off,
                    block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }
    int off = 4;
    if (blockIdx.x >= off)
    {
        wait_send_write();
        return;
    }

    int recv_pe0 = blockIdx.x;
    int recv_pe1 = blockIdx.x + off;

    if (threadIdx.x == 0 && recv_pe0 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe0, NVSHMEM_CMP_EQ, stage);
    }
    if (threadIdx.x == 1 && recv_pe1 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe1, NVSHMEM_CMP_EQ, stage);
    }

    __syncthreads();
    reduce(recv_pe0 == pe ? buffer + pe*pe_off : destination + recv_pe0*pe_off,
            recv_pe1 == pe ? buffer + pe*pe_off : destination + recv_pe1*pe_off,
            destination + recv_pe0*pe_off);
    __syncthreads();

    off /= 4;
    if (blockIdx.x >= off)
    {
        if (threadIdx.x == 0) { atomicAdd(&buffer_lock_[blockIdx.x], 1); }
        wait_send_write();
        return;
    }
    if(threadIdx.x < 3)
    {
        while(atomicCAS(&buffer_lock_[1+threadIdx.x], 1, 0) != 1) { }
        nvshmem_fence();
    }

    __syncthreads();
    for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(destination + recv_pe0*pe_off)[i];
        for (int recv_pe = 1; recv_pe < 4; recv_pe++)
        {
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(buffer + pe * pe_off)[i] = res;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        nvshmemx_signal_op(signal+n_pes+write_chunk, stage, NVSHMEM_SIGNAL_SET, pe);
    }
}

AllReduceTwoShot::AllReduceTwoShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel, numel*2, packet_size, block_size, nnodes,
            nvshmem_n_pes()*2, stream)
{
    assert(packet_size*block_size  == numel * sizeof(half) / nvshmem_n_pes());
    grid_dim.x = nvshmem_n_pes();
    grid_dim.y = 1;
}
void AllReduceTwoShot::run(cudaStream_t stream)
{
    all_reduce_twoshot_kernel<half><<<grid_dim, block_dim, 0, stream>>>(
            destination,
            static_cast<half*>(buffer),
            signal,
            packet_size,
            gpus_per_node,
            stage
            );
    stage+=2;
}
