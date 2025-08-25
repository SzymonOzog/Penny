import os
import subprocess
import setuptools
import importlib
import importlib.resources
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
torch.utils.cpp_extension.COMMON_NVCC_FLAGS = []


if __name__ == '__main__':
    nvshmem_inc = os.getenv('NVSHMEM_INC', "/usr/include/nvshmem_12")
    nvshmem_lib = os.getenv('NVSHMEM_LIB', "/usr/lib/x86_64-linux-gnu/nvshmem/12")
    nvshmem_host_lib = 'libnvshmem_host.so'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true']
    sources = ['csrc/torch_interface.cpp',
               "csrc/exchange.cu",
               "csrc/all_reduce_ring.cu",
               "csrc/all_reduce_double_ring.cu",
               ]
    include_dirs = ['csrc/', nvshmem_inc]
    library_dirs = [nvshmem_lib]
    nvcc_dlink = ['-dlink', f'-L{nvshmem_lib}', '-lnvshmem_device']
    extra_link_args = [f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_lib}']


    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
        'nvcc_dlink': nvcc_dlink,
    }
    # Summary
    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > NVSHMEM lib: {nvshmem_lib}')
    print(f' > NVSHMEM include: {nvshmem_inc}')
    print()

    setuptools.setup(
        name='penny',
        version='0.0.1',
        packages=setuptools.find_packages(
            include=['penny']
        ),
        ext_modules=[
            CUDAExtension(
                name='penny_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
