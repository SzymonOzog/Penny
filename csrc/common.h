#pragma once
#include <cuda_fp16.h>
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
enum class RingType { simple, standard };

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

