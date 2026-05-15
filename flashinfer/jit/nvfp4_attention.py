"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags


_NVFP4_ATTENTION_SM120_MODULE_NAME = "nvfp4_attention_sm120"

_NVFP4_ATTENTION_SM120_SOURCE_FILES = (
    "nvfp4_attention_sm120_binding.cu",
    "nvfp4_attention_quantization_sm120.cu",
)

_NVFP4_ATTENTION_SM120_CUDA_FLAGS = [
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "-U__CUDA_NO_NVFP4_OPERATORS__",
    "-U__CUDA_NO_NVFP4_CONVERSIONS__",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DQBLKSIZE=128",
    "-DKBLKSIZE=128",
    "-DCTA256",
    "-DDQINRMEM",
    "-DPINGPONG_MATH_ORDER",
    "-DPINGPONG_EARLY_RELEASE_K",
    "-DCAUSAL_DISABLE_QK_ORDER",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]


def gen_nvfp4_attention_sm120_module() -> JitSpec:
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / source_file
        for source_file in _NVFP4_ATTENTION_SM120_SOURCE_FILES
    ]
    return gen_jit_spec(
        _NVFP4_ATTENTION_SM120_MODULE_NAME,
        source_paths,
        extra_cuda_cflags=sm120a_nvcc_flags + _NVFP4_ATTENTION_SM120_CUDA_FLAGS,
    )
