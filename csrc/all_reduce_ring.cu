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
__global__ void all_reduce_ring_kernel(scalar_t *destination, scalar_t* buffer, uint64_t* signal, int packet_size, int gpus_per_node) 
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint64_t base_off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;
    const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint32_t ring_id = blockIdx.y;
    const uint64_t ring_off = ring_id * chunk_off * nvshmem_n_pes();
    const uint64_t off = base_off + ring_off;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    const uint32_t local_rank = pe%gpus_per_node;
    const uint32_t my_node = pe/gpus_per_node;

    int send_peer = 0;
    int recv_peer;

    int curr_pe = -1;
    int ring_pos = -1;
    // TODO this is currently a hack to get the ring position, since it changes a lot  
    // it's easier to find it than to derive an expression for it
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

    int send_chunk = ring_pos % n_pes;
    int recv_chunk = (n_pes + ring_pos-1) % n_pes;
    if(ring_id%2 == 1)
    {
        swap_cu(send_chunk, recv_chunk);
        swap_cu(send_peer, recv_peer);
    }

    int stage = 1;
    uint64_t* local_signal = signal + blockIdx.x + blockIdx.y * gridDim.x;
    for (int chunk = 0; chunk < n_pes - 1; chunk++)
    {
        nvshmemx_putmem_signal_nbi_block(destination + off + chunk*chunk_off, buffer + send_chunk*chunk_off + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, send_peer);

        nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i];
            P dst = reinterpret_cast<P*>(destination + off+ chunk*chunk_off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] = res;
        }
        stage++;
        send_chunk = recv_chunk;
        if(ring_id%2 == 1)
            recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
        else
            recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }

    destination += n_pes * chunk_off * gridDim.y;
    for (int chunk = 0; chunk < n_pes - 1; chunk++)
    {
        nvshmemx_putmem_signal_nbi_block(destination + off + chunk*chunk_off, buffer + send_chunk*chunk_off + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, send_peer); 

        nvshmem_signal_wait_until(local_signal , NVSHMEM_CMP_GE, stage);

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] =
                reinterpret_cast<P*>(destination + off+ chunk*chunk_off)[i];
        }
        stage++;
        send_chunk = recv_chunk;
        if(ring_id%2 == 1)
            recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
        else
            recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }
}

void all_reduce_ring(half* buffer, int numel, int packet_size, int block_size, int nnodes, cudaStream_t stream) 
{
    // Can we reduce te size of this buffer?
    half *destination = (half *) nvshmem_malloc(2 * numel * sizeof(half));

    nvshmemx_buffer_register(buffer, numel * sizeof(half));
    
    const uint32_t gpus_per_node = nvshmem_n_pes()/nnodes;
    const uint32_t rings = gpus_per_node;
    const uint32_t grid_size_x = std::ceil(numel*sizeof(half) / float(packet_size*block_size*nvshmem_n_pes()*rings));
    dim3 grid_size(grid_size_x, rings, 1);

    uint64_t *signal = (uint64_t *) nvshmem_malloc(grid_size_x * rings * sizeof(uint64_t));
    cudaMemset(signal, 0, grid_size_x * 2 * sizeof(uint64_t));
    
    //sync the memset before running kernel
    nvshmemx_barrier_all_on_stream(stream);

    all_reduce_ring_kernel<<<grid_size, block_size, 0, stream>>>(
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
}
