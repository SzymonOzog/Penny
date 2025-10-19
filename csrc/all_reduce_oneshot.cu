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

template <typename scalar_t, int N_PES = 8>
__global__ void all_reduce_oneshot_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, scalar_t* __restrict__ output, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint32_t pe_off = block_size/sizeof(scalar_t);
    const uint32_t off = blockIdx.z * pe_off;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    if (blockIdx.x != pe && blockIdx.y == 0)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off + off*N_PES,
                    buffer + off,
                    block_size, signal+pe + blockIdx.z*N_PES, stage, NVSHMEM_SIGNAL_SET, blockIdx.x);
    }

    for(int tid = 0; tid<N_PES; tid++)
    {
        if (threadIdx.x == tid && tid != pe)
        {
            nvshmem_signal_wait_until(signal+tid + blockIdx.z*N_PES, NVSHMEM_CMP_EQ, stage);
        }
    }

    __syncthreads();
    const uint32_t reduce_size = block_size/(N_PES*gridDim.y);
    const uint32_t reduce_off = (blockIdx.y*gridDim.x + blockIdx.x)*reduce_size/sizeof(scalar_t);
    // if(pe == 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    //     printf("reducing z %d sz %d, off %d, buff %d, pe_of %d, block_size %d, off %d\n",
    //             blockIdx.z, reduce_size, reduce_off, int(buffer[reduce_off + off]), pe_off, block_size, off);

    for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(buffer + reduce_off + off)[i];
        for (int recv_pe = 0; recv_pe < N_PES; recv_pe++)
        {
            if(recv_pe == pe)
                continue;
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off + reduce_off + off*N_PES)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(output + reduce_off + off)[i] = res;
    }
}

AllReduceOneShot::AllReduceOneShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream)
    : AllReduce(_buffer, numel, numel*nvshmem_n_pes(), packet_size, block_size, nnodes,
            nvshmem_n_pes() * numel*sizeof(half)/(block_size*packet_size), stream)
{
    grid_dim.x = nvshmem_n_pes();
    grid_dim.y = routes;
    grid_dim.z = (numel*sizeof(half))/(block_size*packet_size);
}
void AllReduceOneShot::run(half* output, cudaStream_t stream)
{
    all_reduce_oneshot_kernel<half><<<grid_dim, block_dim, 0, stream>>>(
            destination,
            static_cast<half*>(buffer),
            output,
            signal,
            packet_size,
            gpus_per_node,
            stage
            );
    stage+=2;
}
