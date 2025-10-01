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
__global__ void all_reduce_oneshot_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint32_t pe_off = block_size/sizeof(scalar_t);
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    for (int send_pe = 0; send_pe<n_pes; send_pe++)
    {
        if (send_pe == pe)
            continue;
        nvshmemx_putmem_signal_nbi_block(destination + off + pe*pe_off,
                buffer + off,
                block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, send_pe);
    }
    for (int recv_pe = 0; recv_pe<n_pes; recv_pe++)
    {
        if (recv_pe == pe)
            continue;
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(signal+recv_pe, NVSHMEM_CMP_EQ, stage);
        __syncthreads();
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + off)[i];
            P dst = reinterpret_cast<P*>(destination + off + recv_pe*pe_off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + off)[i] = res;
        }
    }
}

AllReduceOneShot::AllReduceOneShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel*nvshmem_n_pes(), packet_size, block_size, nnodes,
            nvshmem_n_pes(), stream)
{
    assert(packet_size*block_size  == numel * sizeof(half));
    grid_dim.x = 1;
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
