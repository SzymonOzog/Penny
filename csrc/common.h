#pragma once
#include <cuda_fp16.h>
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
enum class AlgoType { ring_simple, ring_standard, oneshot };

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

class AllReduce
{
public:
    AllReduce(half* _buffer, int sym_mem_size, int packet_size, int block_size, int nnodes, int signals, cudaStream_t stream) :
        packet_size(packet_size)
    {
        destination = (half *) nvshmem_malloc(sym_mem_size * sizeof(half));

        nvshmemx_buffer_register(_buffer, sym_mem_size * sizeof(half));
        buffer = _buffer;
        
        gpus_per_node = nvshmem_n_pes()/nnodes;
        this->block_dim = dim3(block_size, 1, 1);

        signal = (uint64_t *) nvshmem_malloc(signals * sizeof(uint64_t));
        cudaMemset(signal, 0, signals * sizeof(uint64_t));
        
        //sync the memset before running kernel
        nvshmemx_barrier_all_on_stream(stream);
    }
    virtual ~AllReduce()
    {
        nvshmemx_buffer_unregister(buffer);
        nvshmem_free(destination);
        nvshmem_free(signal);
    }

    virtual void run(cudaStream_t stream) = 0;

    half* destination;
    half* buffer;
    uint32_t gpus_per_node;
    dim3 grid_dim;
    dim3 block_dim;
    uint64_t *signal;
    const int packet_size;
    int stage = 1;
};

class AllReduceRingSimple : public AllReduce
{
public:
    AllReduceRingSimple(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream);
    virtual void run(cudaStream_t stream) override;
};

class AllReduceRingStandard : public AllReduce
{
public:
    AllReduceRingStandard(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream);
    virtual void run(cudaStream_t stream) override;
    const bool internode;
};

class AllReduceOneShot : public AllReduce
{
public:
    AllReduceOneShot(half* _buffer, int numel, int packet_size, int block_size, int nnodes, int routes, cudaStream_t stream);
    virtual void run(cudaStream_t stream) override;
    const int block_size;
};

inline void* create_all_reduce(half* buffer, int numel, int packet_size, int block_size, int nnodes, int routes, AlgoType algo_type, cudaStream_t stream)
{
    if (algo_type == AlgoType::ring_simple)
    {
        return reinterpret_cast<void*>(new AllReduceRingSimple(buffer, numel, packet_size, block_size, nnodes, routes, stream));
    }
    else if (algo_type == AlgoType::ring_standard)
    {
        return reinterpret_cast<void*>(new AllReduceRingStandard(buffer, numel, packet_size, block_size, nnodes, routes, stream));
    }
    else if (algo_type == AlgoType::oneshot)
    {
        return reinterpret_cast<void*>(new AllReduceOneShot(buffer, numel, packet_size, block_size, nnodes, routes, stream));
    }
    return nullptr;
}

inline void destroy_all_reduce(void* all_reduce_obj)
{
    delete reinterpret_cast<AllReduce*>(all_reduce_obj);
}

inline void all_reduce(void* all_reduce_obj, cudaStream_t stream) 
{
    reinterpret_cast<AllReduce*>(all_reduce_obj)->run(stream);
}
