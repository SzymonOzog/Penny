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

__device__ int buffer_lock[4] = {0};

template <typename scalar_t>
__global__ void all_reduce_oneshot_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint32_t pe_off = block_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

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

    if (blockIdx.x != pe)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                    buffer,
                    block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, blockIdx.x);
    }
    int off = 4;
    if (blockIdx.x >= off)
    {
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
    reduce(recv_pe0 == pe ? buffer : destination + recv_pe0*pe_off,
            recv_pe1 == pe ? buffer : destination + recv_pe1*pe_off,
            destination + recv_pe0*pe_off);
    __syncthreads();

    off /= 2;
    if (blockIdx.x >= off)
    {
        if (threadIdx.x == 0) { atomicAdd(&buffer_lock[blockIdx.x], 1); }
        return;
    }
    if(threadIdx.x == 0)
    {
        while(atomicCAS(&buffer_lock[blockIdx.x+off], 1, 0) != 1) { }
    }
    __syncthreads();
    recv_pe1 = blockIdx.x + off;
    reduce(destination + recv_pe0*pe_off,
            destination + recv_pe1*pe_off,
            destination + recv_pe0*pe_off);
    __syncthreads();

    off /= 2;
    if (blockIdx.x >= off)
    {
        if (threadIdx.x == 0) { atomicAdd(&buffer_lock[blockIdx.x], 1); }
        return;
    }
    if(threadIdx.x == 0)
    {
        while(atomicCAS(&buffer_lock[blockIdx.x+off], 1, 0) != 1) {}
        nvshmem_quiet();
    }
    __syncthreads();
    recv_pe1 = blockIdx.x + off;
    reduce(destination + recv_pe0*pe_off,
            destination + recv_pe1*pe_off,
            buffer);
}

AllReduceOneShot::AllReduceOneShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel, numel*nvshmem_n_pes(), packet_size, block_size, nnodes,
            nvshmem_n_pes(), stream)
{
    assert(packet_size*block_size  == numel * sizeof(half));
    grid_dim.x = nvshmem_n_pes();
    grid_dim.y = 1;
}
void AllReduceOneShot::run(cudaStream_t stream)
{
    all_reduce_oneshot_kernel<half><<<grid_dim, block_dim, 0, stream>>>(
            destination,
            static_cast<half*>(buffer),
            signal,
            packet_size,
            gpus_per_node,
            stage
            );
    stage+=2;
}
