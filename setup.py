# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
from setuptools import setup
from pathlib import Path
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

def make_cuda_ext(name, module, sources, include_dirs, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
        print(f'Compiling {sources} with CUDA')
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,)


if __name__ == '__main__':
    setup(
        name='gspn_cuda_extension',
        description='Generalized Spatial Propagation Network c and cuda modules',
        author='Hongjun Wang and Sifei Liu',
        ext_modules=[
            make_cuda_ext(
                name='gaterecurrent2dnoind_cuda',
                module='ops.gaterecurrent',
                sources=['src/gaterecurrent2dnoind_cuda.cpp', 'src/gaterecurrent2dnoind_kernel.cu'],
                include_dirs=[Path(this_dir) / "ops" / "gaterecurrent"]),
            # make_cuda_ext(
            #     name='gaterecurrent2d_cuda',
            #     module='custom_ops.gaterecurrent',
            #     sources=['src/gaterecurrent2d_cuda.cpp', 'src/gaterecurrent2d_kernel.cu'],
            #     include_dirs=[Path(this_dir) / "custom_ops" / "gaterecurrent"]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
    )
