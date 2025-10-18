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


__device__ int buffer_lock_[2] = {0};

template <typename scalar_t>
__global__ void all_reduce_twoshot_kernel(scalar_t* __restrict__ destination, scalar_t*  buffer, uint64_t* signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    const uint32_t block_size = blockDim.x * packet_size;

    const uint32_t pe_off = block_size/sizeof(scalar_t);

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

    if (write_chunk != pe && blockIdx.y == 0)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                    buffer + write_chunk*pe_off,
                    block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }
    int off = 2;

    int recv_pe0 = blockIdx.x%off;
    int recv_pe1 = recv_pe0 + off;
    int recv_pe2 = recv_pe1 + off;
    int recv_pe3 = recv_pe2 + off;

    if (threadIdx.x == 0 && recv_pe0 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe0, NVSHMEM_CMP_EQ, stage);
    }
    if (threadIdx.x == 1 && recv_pe1 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe1, NVSHMEM_CMP_EQ, stage);
    }
    if (threadIdx.x == 2 && recv_pe2 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe2, NVSHMEM_CMP_EQ, stage);
    }
    if (threadIdx.x == 3 && recv_pe3 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe3, NVSHMEM_CMP_EQ, stage);
    }

    __syncthreads();
    const int blocks = gridDim.y * gridDim.x/off;
    {
        const uint32_t reduce_size = block_size/blocks;
        const uint32_t reduce_off = (blockIdx.y * blocks/gridDim.y + blockIdx.x/off) * reduce_size/sizeof(scalar_t);

        scalar_t* s1_p = recv_pe0 == pe ? buffer + pe*pe_off : destination + recv_pe0*pe_off;
        scalar_t* s2_p = recv_pe1 == pe ? buffer + pe*pe_off : destination + recv_pe1*pe_off;
        scalar_t* s3_p = recv_pe2 == pe ? buffer + pe*pe_off : destination + recv_pe2*pe_off;
        scalar_t* s4_p = recv_pe3 == pe ? buffer + pe*pe_off : destination + recv_pe3*pe_off;
        scalar_t* dst = destination + recv_pe0*pe_off;

        // if(pe == 0 && blockIdx.x%4 == 1 && threadIdx.x == 0)
        //     printf("reducing %d, %d, %d, %d,block  %d, %d \n",
        //             reduce_size, reduce_off, int(s1_p[reduce_off]), int(s2_p[reduce_off]), blockIdx.x, blockIdx.y);

        for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
        {
            P src1 = reinterpret_cast<P*>(s1_p + reduce_off)[i];
            P src2 = reinterpret_cast<P*>(s2_p + reduce_off)[i];
            P src3 = reinterpret_cast<P*>(s3_p + reduce_off)[i];
            P src4 = reinterpret_cast<P*>(s4_p + reduce_off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(src1.data[j]) + float(src2.data[j]) + float(src3.data[j]) + float(src4.data[j]);
            reinterpret_cast<P*>(dst + reduce_off)[i] = res;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) { atomicAdd(&buffer_lock_[blockIdx.x%off], 1); }

    if(threadIdx.x < 2)
    {
        while(atomicAdd(&buffer_lock_[threadIdx.x], 0) != blocks) { }
        nvshmem_fence();
    }

    __syncthreads();
    const uint32_t reduce_size = block_size/(n_pes*gridDim.y);
    const uint32_t reduce_off = (blockIdx.y*gridDim.x + blockIdx.x)*reduce_size/sizeof(scalar_t);
    for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(destination + recv_pe0*pe_off + reduce_off)[i];
        for (int recv_pe = 1; recv_pe < 2; recv_pe++)
        {
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off + reduce_off)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(buffer + pe * pe_off + reduce_off)[i] = res;
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        nvshmemx_signal_op(signal+n_pes+pe, 1, NVSHMEM_SIGNAL_ADD, pe);
    }
    if (threadIdx.x == 0)
    {
         
        nvshmem_signal_wait_until(signal+n_pes+pe, NVSHMEM_CMP_EQ, stage*nvshmem_n_pes()*gridDim.y);
    }
    __syncthreads();
    __threadfence_system();
    if (write_chunk == pe)
    {
        reinterpret_cast<int2*>(buffer_lock_)[0] = make_int2(0,0);
        return;
    }


    if(blockIdx.y == 0)
    {
        nvshmemx_putmem_signal_nbi_block(destination + (n_pes+pe)*pe_off,
                buffer + pe*pe_off,
                block_size, signal+n_pes+pe, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }

    if (threadIdx.x == 0)
        nvshmem_signal_wait_until(signal+n_pes+write_chunk, NVSHMEM_CMP_EQ, stage);
    __syncthreads();

    const uint32_t write_size = block_size/gridDim.y;
    const uint32_t write_off = (blockIdx.y*write_size)/sizeof(scalar_t);

    for (int i = threadIdx.x; i < write_size/(sizeof(P)); i += blockDim.x)
    {
        reinterpret_cast<P*>(buffer + write_chunk*pe_off + write_off)[i] =
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
    stage+=1;
}
