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
__global__ void all_reduce_simple_ring_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    int send_peer = (pe+1) % n_pes;
    int recv_peer = (n_pes + pe-1) % n_pes;
    int ring_pos = pe;

    int send_chunk = ring_pos % n_pes;
    int recv_chunk = (n_pes + ring_pos-1) % n_pes;

    uint64_t* local_signal = signal + blockIdx.x + blockIdx.y * gridDim.x;
    if (ring_pos == 0)
    {
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                reinterpret_cast<float4*>(buffer + off),
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);
    }
    else 
    {
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + off)[i];
            P dst = reinterpret_cast<P*>(destination + off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + off)[i] = res;
        }
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                reinterpret_cast<float4*>(buffer + off),
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);
    }

    stage++;

    if (ring_pos == n_pes - 1)
    {
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                reinterpret_cast<float4*>(buffer  + off),
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);
    }
    else
    {

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + off)[i] =
                reinterpret_cast<P*>(destination + off)[i];
        }
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                reinterpret_cast<float4*>(buffer + off),
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);
    }
}

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
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off + chunk*chunk_off),
                reinterpret_cast<float4*>(buffer + send_chunk*chunk_off + off),
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

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
        if(ring_id%2 == 1 && INTERNODE)
            recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
        else
            recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }

    destination += n_pes * chunk_off * gridDim.y;
    for (int chunk = 0; chunk < n_pes - 1; chunk++) 
    {
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off + chunk*chunk_off),
                reinterpret_cast<float4*>(buffer + send_chunk*chunk_off + off),
                block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] =
                reinterpret_cast<P*>(destination + off+ chunk*chunk_off)[i];
        }
        stage++;
        send_chunk = recv_chunk;
        if(ring_id%2 == 1 && INTERNODE)
            recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
        else
            recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }
}

class AllReduce
{
public:
    AllReduce(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, RingType _ring_type, cudaStream_t stream) :
        packet_size(packet_size),
        internode(nnodes > 1),
        ring_type(_ring_type)

    {
        // Can we reduce te size of this buffer?
        destination = (half *) nvshmem_malloc(2 * numel * sizeof(half));

        nvshmemx_buffer_register(_buffer, numel * sizeof(half));
        buffer = _buffer;
        
        gpus_per_node = nvshmem_n_pes()/nnodes;
        const uint32_t rings = routes;

        uint32_t grid_size_x;
        if(ring_type == RingType::standard)
        {
            grid_size_x = std::ceil(numel*sizeof(half) / float(packet_size*block_size*nvshmem_n_pes()*rings));
        }
        else if (ring_type == RingType::simple)
        {
            grid_size_x = std::ceil(numel*sizeof(half) / float(packet_size*block_size*rings));
        }
        grid_size = dim3(grid_size_x, rings, 1);
        this->block_size = dim3(block_size, 1, 1);

        signal = (uint64_t *) nvshmem_malloc(grid_size_x * rings * sizeof(uint64_t));
        cudaMemset(signal, 0, grid_size_x * rings * sizeof(uint64_t));
        
        //sync the memset before running kernel
        nvshmemx_barrier_all_on_stream(stream);
    }
    ~AllReduce()
    {
        nvshmemx_buffer_unregister(buffer);
        nvshmem_free(destination);
        nvshmem_free(signal);
    }

    void run(cudaStream_t stream)
    {
        if(ring_type == RingType::standard)
        {
            if(internode)
            {
                all_reduce_ring_kernel<half, true><<<grid_size, block_size, 0, stream>>>(
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
                all_reduce_ring_kernel<half, false><<<grid_size, block_size, 0, stream>>>(
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
        else if (ring_type == RingType::simple)
        {
            all_reduce_simple_ring_kernel<half><<<grid_size, block_size, 0, stream>>>(
                    destination,
                    static_cast<half*>(buffer),
                    signal,
                    packet_size,
                    gpus_per_node,
                    stage
                    );
            stage+=2;
        }
    }

    half* destination;
    half* buffer;
    uint32_t gpus_per_node;
    dim3 grid_size;
    dim3 block_size;
    uint64_t *signal;
    const int packet_size;
    const bool internode;
    int stage = 1;
    RingType ring_type;
};

void* create_all_reduce_ring(half* buffer, int numel, int packet_size, int block_size, int nnodes, int routes, RingType ring_type, cudaStream_t stream)
{
    return reinterpret_cast<void*>(new AllReduce(buffer, numel, packet_size, block_size, nnodes, routes, ring_type, stream));
}

void destroy_all_reduce_ring(void* all_reduce_obj)
{
    delete reinterpret_cast<AllReduce*>(all_reduce_obj);
}

void all_reduce_ring(void* all_reduce_obj, cudaStream_t stream) 
{
    reinterpret_cast<AllReduce*>(all_reduce_obj)->run(stream);
}
