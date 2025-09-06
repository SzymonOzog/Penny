#include "device_host_transport/nvshmem_common_transport.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <nvshmem.h>
#include <nvshmemx.h>

template <typename T> __device__ __forceinline__ void swap_cu(T& a, T& b)
{
    T c(a); a=b; b=c;
}

// like std::array, but aligned
// goal: generate ld.128 and st.128 instructions
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};


template <typename scalar_t>
__global__ void all_reduce_tree_kernel(scalar_t *destination, scalar_t* buffer, uint64_t* signal, int packet_size, int gpus_per_node) 
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint64_t base_off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;
    const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint32_t tree_id = blockIdx.y;
    const uint64_t tree_off = tree_id * chunk_off;
    const uint64_t off = base_off + tree_off;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    const uint32_t local_rank = pe%gpus_per_node;
    const uint32_t my_node = pe/gpus_per_node;

    int send_peer = 0;
    int recv_peer;

    if (local_rank == tree_id)
    {
        //TODO this only works for 2 nodes
        send_peer = (n_pes + pe + gpus_per_node) % n_pes;
        if (tree_id % 2 == 1)
        {
            recv_peer = my_node * gpus_per_node + (gpus_per_node + local_rank + 1) % gpus_per_node;
        }
        else 
        {
            recv_peer = my_node * gpus_per_node + (gpus_per_node + local_rank - 1) % gpus_per_node;
        }
    }

    else
    {
        send_peer = my_node * gpus_per_node + (local_rank + 1) % gpus_per_node;
        recv_peer = my_node * gpus_per_node + (gpus_per_node + local_rank - 1) % gpus_per_node;
        if (tree_id % 2 == 1)
        {
            swap_cu(send_peer, recv_peer);
            if (tree_id - local_rank == 1)
            {
                recv_peer = - 1;
            }
        }
        else if (local_rank - tree_id == 1)
        {
            recv_peer = - 1;
        }
    }

    int curr_pe = -1;
    int ring_pos = -1;

    int stage = 1;
    uint64_t* local_signal = signal + blockIdx.x + blockIdx.y * gridDim.x;

    if (recv_peer != -1)
    {
        nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + off)[i];
            P dst = reinterpret_cast<P*>(destination + off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + off)[i] = res;
        }
    }

    if (local_rank != tree_id)
    {
        nvshmemx_putmem_signal_block(destination + off , buffer + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, send_peer);
    }

    destination += n_pes * chunk_off * gridDim.y;
    local_signal += gridDim.x * gridDim.y;

    if (local_rank == tree_id)
    {
        nvshmemx_putmem_signal_block(destination + off , buffer + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, send_peer);
    }

    nvshmem_signal_wait_until(local_signal , NVSHMEM_CMP_GE, stage);

    if (local_rank == tree_id)
    {
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + off)[i];
            P dst = reinterpret_cast<P*>(destination + off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + off)[i] = res;
        }
    }
    else
    {
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + off)[i] =
                reinterpret_cast<P*>(destination + off)[i];
        }
    }

    if (recv_peer != -1)
    {
        nvshmemx_putmem_signal_block(destination + off , buffer + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, recv_peer); 
    }
}

void all_reduce_tree(half* buffer, int numel, int packet_size, int block_size, int nnodes, cudaStream_t stream) 
{
    // Can we reduce te size of this buffer?
    half *destination = (half *) nvshmem_malloc(2 * numel * sizeof(half));

    nvshmemx_buffer_register(buffer, numel * sizeof(half));
    
    const uint32_t gpus_per_node = nvshmem_n_pes()/nnodes;
    const uint32_t trees = gpus_per_node;
    const uint32_t grid_size_x = std::ceil(numel*sizeof(half) / float(packet_size*block_size*trees));
    dim3 grid_size(grid_size_x, trees, 1);

    int signal_size = 2 * grid_size_x * trees * sizeof(uint64_t);
    uint64_t *signal = (uint64_t *) nvshmem_malloc(signal_size);
    cudaMemset(signal, 0, signal_size);
    
    //sync the memset before running kernel
    nvshmemx_barrier_all_on_stream(stream);

    all_reduce_tree_kernel<<<grid_size, block_size, 0, stream>>>(
            destination,
            static_cast<half*>(buffer),
            signal,
            packet_size,
            gpus_per_node
            );

    nvshmemx_barrier_all_on_stream(stream);
    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer);
    nvshmem_free(destination);
    nvshmem_free(signal);
}
