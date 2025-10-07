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
        const int packet_size, const int gpus_per_node, int stage, const int block_size)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;
    __shared__ int buffer_lock;
    if (threadIdx.x == 0)
        buffer_lock = 0;

    const uint32_t pe_off = block_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    for (int send_pe = 0; send_pe<n_pes; send_pe++)
    {
        if (send_pe == pe) continue;

        nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                buffer,
                block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, send_pe);
    }
    nvshmem_quiet();
    __syncthreads();
    int warp_id = threadIdx.x/32;
    if(warp_id == 0)
        return;
    int lane_id = threadIdx.x%32;
    int recv_pe = warp_id;

    if (lane_id == 0)
    {
        nvshmem_signal_wait_until(signal+recv_pe, NVSHMEM_CMP_EQ, stage);
        while (atomicCAS(&buffer_lock, 0, 1) != 0) {/*wait*/}
    }

    __syncwarp();
    for (int i = lane_id; i < block_size/(sizeof(P)); i += 32)
    {
        P buf = reinterpret_cast<P*>(buffer)[i];
        P dst = reinterpret_cast<P*>(destination + recv_pe*pe_off)[i];
        P res;
        for (int j = 0; j < P::size; j++)
            res.data[j] = float(buf.data[j]) + float(dst.data[j]);
        reinterpret_cast<P*>(buffer)[i] = res;
    }
    __syncwarp();
    if (lane_id == 0) 
    {
        atomicExch(&buffer_lock, 0);
    }
}

AllReduceOneShot::AllReduceOneShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel*nvshmem_n_pes(), packet_size, 32*nvshmem_n_pes(), nnodes,
            nvshmem_n_pes(), stream), block_size(numel*sizeof(half))
{
    // assert(packet_size*block_size  == numel * sizeof(half));
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
            stage,
            block_size
            );
    stage+=2;
}
