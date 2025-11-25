“What I cannot create I do not understand” - This is why I started Penny, my own version of NCCL.

If you want to read about it, there is a worklog on my blogpost where I describe a step by step process of creating it:
- [Part 1](https://szymonozog.github.io/posts/2025-09-21-Penny-worklog-1.html)
- [Part 2](https://szymonozog.github.io/posts/2025-10-26-Penny-worklog-2.html)
- [Part 3](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html)

# Installation 

To install Penny you need to export NVSHMEM_LIB and NVSHMEM_INC environment variables that point to the `/lib` and `/include` directories of your NVSHMEM installation

Afterwards just

```
git clone https://github.com/SzymonOzog/Penny.git
cd Penny
pip install -e . --no-build-isolation
```

# Using Low Latency Intranode Allreduce

Penny provides a drop in replacement for the vLLM/SGLang custom all reduce class that allows it to run multinode. For SGLang there is a patch that you can apply to get it running:
```
cd YOUR_SGLANG_DIR
git apply YOUR_PENNY_DIR/extra/sglang.patch
```

You also need to export the number of nodes that you're running(Currently up to 4 nodes are templated and tested, for more edit `extra/custom_all_reduce.cuh` at your own risk)
```
export NNODES=2
```

Afterwards you can serve your favourite model with Low Latency allreduce

