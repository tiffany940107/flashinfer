// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Fused standard-MXFP8 quantization and paged-cache insertion for the
// DeepSeek-V4 sparse-MLA cache ABI. Each warp owns one latent-KV row.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120 {

namespace {

using Cache = KVCacheTraits<ModelType::DSV4_MXFP8>;
using bf16 = __nv_bfloat16;

constexpr int kDNope = Cache::D_NOPE;
constexpr int kDRope = Cache::D_ROPE;
constexpr int kDLatent = Cache::D_QK;
constexpr int kGroupSize = Cache::QUANT_TILE;
constexpr int kNumScales = Cache::NUM_SCALES;
constexpr int kDataBytesPerToken = Cache::KV_SCALE_GMEM_OFFSET;
constexpr int kScaleBytesPerToken = Cache::SCALE_BYTES_PER_TOKEN;
constexpr int kBytesPerToken = Cache::KV_GMEM_STRIDE;
constexpr int kThreadsPerToken = 32;
constexpr int kDuplicateScanMaxTokens = 256;
// The last two scales and the two ABI padding bytes form one aligned owner
// word during large append calls. The winning row rewrites all four bytes.
constexpr int kOwnerByteOffset = 12;
constexpr size_t kVectorAlignment = alignof(uint4);
constexpr float kFp8Max = 448.0f;

static_assert(kDNope == 448);
static_assert(kDRope == 64);
static_assert(kDLatent == 512);
static_assert(kGroupSize == 32);
static_assert(kNumScales == 14);
static_assert(kDataBytesPerToken == 576);
static_assert(kScaleBytesPerToken == 16);
static_assert(kBytesPerToken == 592);
static_assert(kOwnerByteOffset % alignof(uint32_t) == 0);
static_assert(kOwnerByteOffset + sizeof(uint32_t) == kScaleBytesPerToken);

struct PagedLayout {
  int num_pages;
  int page_size;
  size_t page_stride_bytes;
};

PagedLayout parse_mxfp8_paged_layout(const TensorView& cache) {
  TVM_FFI_ICHECK_EQ(cache.dtype(), dl_uint8) << "cache must have dtype uint8";
  TVM_FFI_ICHECK_GE(cache.ndim(), 2);
  TVM_FFI_ICHECK_LE(cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(cache.data_ptr()) % kVectorAlignment, 0)
      << "MXFP8 cache base pointer must be " << kVectorAlignment << "-byte aligned";
  TVM_FFI_ICHECK_EQ(static_cast<size_t>(cache.stride(0)) % kVectorAlignment, 0)
      << "MXFP8 cache page stride must be a multiple of " << kVectorAlignment << " bytes";
  TVM_FFI_ICHECK_EQ(cache.stride(-1), 1) << "cache last dimension must be contiguous";

  TVM_FFI_ICHECK_LE(cache.size(0), std::numeric_limits<int>::max());
  const int num_pages = static_cast<int>(cache.size(0));
  int page_size = 0;
  const size_t page_stride_bytes = static_cast<size_t>(cache.stride(0));
  if (cache.ndim() == 2) {
    const size_t page_bytes = static_cast<size_t>(cache.size(1));
    TVM_FFI_ICHECK_EQ(page_bytes % kBytesPerToken, 0)
        << "2D cache page width must be divisible by " << kBytesPerToken;
    TVM_FFI_ICHECK_LE(page_bytes / kBytesPerToken,
                      static_cast<size_t>(std::numeric_limits<int>::max()));
    page_size = static_cast<int>(page_bytes / kBytesPerToken);
    TVM_FFI_ICHECK_GE(page_stride_bytes, page_bytes)
        << "cache page stride is smaller than its MXFP8 payload";
  } else {
    TVM_FFI_ICHECK_EQ(cache.size(-1), kBytesPerToken)
        << "MXFP8 cache row width must be exactly " << kBytesPerToken;
    int page_dim = 1;
    if (cache.ndim() == 3) {
      page_dim = 1;
    } else if (cache.size(1) == 1) {
      page_dim = 2;
    } else if (cache.size(2) == 1) {
      page_dim = 1;
    } else {
      TVM_FFI_ICHECK(false)
          << "4D cache must have a singleton latent-head axis at dimension 1 or 2";
    }
    TVM_FFI_ICHECK_EQ(cache.stride(page_dim), kBytesPerToken)
        << "MXFP8 cache entries inside a page must have stride " << kBytesPerToken;
    TVM_FFI_ICHECK_LE(cache.size(page_dim), std::numeric_limits<int>::max());
    page_size = static_cast<int>(cache.size(page_dim));
    TVM_FFI_ICHECK_GE(page_stride_bytes, static_cast<size_t>(page_size) * kBytesPerToken)
        << "cache page stride is smaller than its MXFP8 payload";
  }
  return {num_pages, page_size, page_stride_bytes};
}

__device__ __forceinline__ uint8_t ue8m0_scale_from_amax(float amax, float& inv_scale) {
  if (amax == 0.0f) {
    inv_scale = 1.0f;
    return 0;
  }

  const float raw_scale = amax / kFp8Max;
  uint32_t bits = __float_as_uint(raw_scale);
  uint8_t exponent = static_cast<uint8_t>(bits >> 23);
  const uint32_t mantissa = bits & 0x007FFFFF;
  // Round toward +inf to a power of two, matching FlashInfer's linear-layout
  // mxfp8_quantize contract. Preserve the representable subnormal-zero case.
  if (mantissa != 0 && exponent != 0xFE && !(exponent == 0 && mantissa <= 0x00400000)) {
    ++exponent;
  }
  inv_scale = exponent == 0 ? 1.0f : exp2f(127.0f - static_cast<float>(exponent));
  return exponent;
}

__device__ __forceinline__ void quantize_token(const bf16* input, uint8_t* data_output,
                                               uint8_t* scale_output) {
  const int lane = threadIdx.x;
#pragma unroll
  for (int group = 0; group < kNumScales; ++group) {
    const float value = __bfloat162float(input[group * kGroupSize + lane]);
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
    }
    float inv_scale;
    const uint8_t scale = ue8m0_scale_from_amax(amax, inv_scale);
    if (lane == 0) scale_output[group] = scale;
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    data_output[group * kGroupSize + lane] = quantized.__x;
  }

  // Preserve the BF16 RoPE bits. Eight lanes copy one aligned uint4 each.
  if (lane < 8) {
    *reinterpret_cast<uint4*>(data_output + kDNope + lane * sizeof(uint4)) =
        *reinterpret_cast<const uint4*>(input + kDNope + lane * 8);
  }
  if (lane == 0) {
    *reinterpret_cast<uint16_t*>(scale_output + kNumScales) = 0;
  }
}

__global__ void Mxfp8QuantizePackKernel(const bf16* input, uint8_t* cache, int num_pages,
                                        int page_size, size_t page_stride_bytes) {
  const int page_idx = blockIdx.x;
  const int entry_idx = blockIdx.y;
  if (page_idx >= num_pages || entry_idx >= page_size) return;

  const size_t token_idx = static_cast<size_t>(page_idx) * page_size + entry_idx;
  const bf16* token_input = input + token_idx * kDLatent;
  uint8_t* page = cache + static_cast<size_t>(page_idx) * page_stride_bytes;
  uint8_t* data_output = page + static_cast<size_t>(entry_idx) * kDataBytesPerToken;
  uint8_t* scale_output = page + static_cast<size_t>(page_size) * kDataBytesPerToken +
                          static_cast<size_t>(entry_idx) * kScaleBytesPerToken;
  quantize_token(token_input, data_output, scale_output);
}

__device__ __forceinline__ uint8_t* append_data_output(uint8_t* cache, size_t slot, int page_size,
                                                       size_t page_stride_bytes) {
  const size_t page_idx = slot / page_size;
  const size_t entry_idx = slot % page_size;
  return cache + page_idx * page_stride_bytes + entry_idx * kDataBytesPerToken;
}

__device__ __forceinline__ uint8_t* append_scale_output(uint8_t* cache, size_t slot, int page_size,
                                                        size_t page_stride_bytes) {
  const size_t page_idx = slot / page_size;
  const size_t entry_idx = slot % page_size;
  uint8_t* page = cache + page_idx * page_stride_bytes;
  return page + static_cast<size_t>(page_size) * kDataBytesPerToken +
         entry_idx * kScaleBytesPerToken;
}

template <typename IdType>
__global__ void Mxfp8QuantizeAppendFirstKernel(const bf16* input, const IdType* slot_mapping,
                                               int num_tokens, uint8_t* cache, int num_pages,
                                               int page_size, size_t page_stride_bytes) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(num_pages) * page_size) return;

  // Decode-sized appends avoid extra launches. Only the first input row for
  // a valid slot writes, preventing two blocks from tearing one cache record.
  bool has_earlier_duplicate = false;
  for (int prior = threadIdx.x; prior < token_idx; prior += kThreadsPerToken) {
    has_earlier_duplicate |= slot_mapping[prior] == slot;
  }
  if (__any_sync(0xffffffffu, has_earlier_duplicate)) return;

  const bf16* token_input = input + static_cast<size_t>(token_idx) * kDLatent;
  uint8_t* data_output =
      append_data_output(cache, static_cast<size_t>(slot), page_size, page_stride_bytes);
  uint8_t* scale_output =
      append_scale_output(cache, static_cast<size_t>(slot), page_size, page_stride_bytes);
  quantize_token(token_input, data_output, scale_output);
}

template <typename IdType>
__global__ void ResetAppendOwnersKernel(const IdType* slot_mapping, int num_tokens, uint8_t* cache,
                                        int num_pages, int page_size, size_t page_stride_bytes) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) return;
  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(num_pages) * page_size) return;
  uint8_t* scale_output =
      append_scale_output(cache, static_cast<size_t>(slot), page_size, page_stride_bytes);
  atomicExch(reinterpret_cast<unsigned int*>(scale_output + kOwnerByteOffset), 0xffffffffu);
}

template <typename IdType>
__global__ void ClaimAppendOwnersKernel(const IdType* slot_mapping, int num_tokens, uint8_t* cache,
                                        int num_pages, int page_size, size_t page_stride_bytes) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) return;
  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(num_pages) * page_size) return;
  uint8_t* scale_output =
      append_scale_output(cache, static_cast<size_t>(slot), page_size, page_stride_bytes);
  atomicMin(reinterpret_cast<unsigned int*>(scale_output + kOwnerByteOffset),
            static_cast<unsigned int>(token_idx));
}

template <typename IdType>
__global__ void Mxfp8QuantizeAppendWinnerKernel(const bf16* input, const IdType* slot_mapping,
                                                int num_tokens, uint8_t* cache, int num_pages,
                                                int page_size, size_t page_stride_bytes) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(num_pages) * page_size) return;

  uint8_t* scale_output =
      append_scale_output(cache, static_cast<size_t>(slot), page_size, page_stride_bytes);
  const unsigned int owner =
      *reinterpret_cast<const unsigned int*>(scale_output + kOwnerByteOffset);
  if (owner != static_cast<unsigned int>(token_idx)) return;

  const bf16* token_input = input + static_cast<size_t>(token_idx) * kDLatent;
  uint8_t* data_output =
      append_data_output(cache, static_cast<size_t>(slot), page_size, page_stride_bytes);
  // quantize_token restores scales 12/13 and the two zero-padding bytes after
  // the owner word has served its temporary purpose.
  quantize_token(token_input, data_output, scale_output);
}

template <typename IdType>
void launch_quantize_append(const bf16* input, const IdType* slot_mapping, int num_tokens,
                            uint8_t* cache, int num_pages, int page_size, size_t page_stride_bytes,
                            cudaStream_t stream) {
  const dim3 token_grid(num_tokens);
  const dim3 token_block(kThreadsPerToken);
  if (num_tokens <= kDuplicateScanMaxTokens) {
    Mxfp8QuantizeAppendFirstKernel<IdType><<<token_grid, token_block, 0, stream>>>(
        input, slot_mapping, num_tokens, cache, num_pages, page_size, page_stride_bytes);
    return;
  }

  constexpr int kOwnerThreads = 256;
  const dim3 owner_grid((num_tokens + kOwnerThreads - 1) / kOwnerThreads);
  const dim3 owner_block(kOwnerThreads);
  ResetAppendOwnersKernel<IdType><<<owner_grid, owner_block, 0, stream>>>(
      slot_mapping, num_tokens, cache, num_pages, page_size, page_stride_bytes);
  ClaimAppendOwnersKernel<IdType><<<owner_grid, owner_block, 0, stream>>>(
      slot_mapping, num_tokens, cache, num_pages, page_size, page_stride_bytes);
  Mxfp8QuantizeAppendWinnerKernel<IdType><<<token_grid, token_block, 0, stream>>>(
      input, slot_mapping, num_tokens, cache, num_pages, page_size, page_stride_bytes);
}

int64_t num_latent_rows(const TensorView& latent_kv) {
  TVM_FFI_ICHECK(latent_kv.ndim() >= 2 && latent_kv.ndim() <= 4)
      << "latent_kv must be 2D, 3D, or 4D";
  TVM_FFI_ICHECK_EQ(latent_kv.size(-1), kDLatent)
      << "latent_kv last dimension must be " << kDLatent;
  TVM_FFI_ICHECK_EQ(latent_kv.dtype(), dl_bfloat16) << "latent_kv must have dtype bfloat16";
  TVM_FFI_ICHECK(latent_kv.IsContiguous()) << "latent_kv must be contiguous";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(latent_kv.data_ptr()) % kVectorAlignment, 0)
      << "latent_kv base pointer must be " << kVectorAlignment << "-byte aligned";
  int64_t rows = 1;
  for (int i = 0; i + 1 < latent_kv.ndim(); ++i) rows *= latent_kv.size(i);
  return rows;
}

}  // namespace

void SparseMlaSm120Mxfp8QuantizePack(TensorView latent_kv, TensorView cache) {
  CHECK_CUDA(latent_kv);
  CHECK_CUDA(cache);
  TVM_FFI_ICHECK_EQ(latent_kv.device().device_id, cache.device().device_id)
      << "latent_kv and cache must be on the same CUDA device";
  const PagedLayout layout = parse_mxfp8_paged_layout(cache);
  TVM_FFI_ICHECK_EQ(num_latent_rows(latent_kv),
                    static_cast<int64_t>(layout.num_pages) * layout.page_size)
      << "latent_kv must contain exactly one row per cache slot";
  if (layout.num_pages == 0 || layout.page_size == 0) return;

  ffi::CUDADeviceGuard device_guard(latent_kv.device().device_id);
  cudaStream_t stream = get_stream(latent_kv.device());
  Mxfp8QuantizePackKernel<<<dim3(layout.num_pages, layout.page_size), kThreadsPerToken, 0,
                            stream>>>(static_cast<const bf16*>(latent_kv.data_ptr()),
                                      static_cast<uint8_t*>(cache.data_ptr()), layout.num_pages,
                                      layout.page_size, layout.page_stride_bytes);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "MXFP8 sparse-MLA full-page pack launch failed: " << cudaGetErrorString(status);
}

void SparseMlaSm120Mxfp8QuantizeAppend(TensorView latent_kv, TensorView slot_mapping,
                                       TensorView cache) {
  CHECK_CUDA(latent_kv);
  CHECK_CUDA(slot_mapping);
  CHECK_CUDA(cache);
  TVM_FFI_ICHECK_EQ(latent_kv.device().device_id, cache.device().device_id)
      << "latent_kv and cache must be on the same CUDA device";
  TVM_FFI_ICHECK_EQ(slot_mapping.device().device_id, cache.device().device_id)
      << "slot_mapping and cache must be on the same CUDA device";
  TVM_FFI_ICHECK_EQ(slot_mapping.ndim(), 1) << "slot_mapping must be 1D";
  TVM_FFI_ICHECK(slot_mapping.dtype() == dl_int32 || slot_mapping.dtype() == dl_int64)
      << "slot_mapping must have dtype int32 or int64";
  TVM_FFI_ICHECK(slot_mapping.IsContiguous()) << "slot_mapping must be contiguous";

  const PagedLayout layout = parse_mxfp8_paged_layout(cache);
  TVM_FFI_ICHECK_LE(slot_mapping.size(0), std::numeric_limits<int>::max());
  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  TVM_FFI_ICHECK_EQ(num_latent_rows(latent_kv), num_tokens)
      << "latent_kv must contain exactly one row per slot_mapping entry";
  if (num_tokens == 0) return;

  ffi::CUDADeviceGuard device_guard(latent_kv.device().device_id);
  cudaStream_t stream = get_stream(latent_kv.device());
  if (slot_mapping.dtype() == dl_int32) {
    launch_quantize_append(static_cast<const bf16*>(latent_kv.data_ptr()),
                           static_cast<const int32_t*>(slot_mapping.data_ptr()), num_tokens,
                           static_cast<uint8_t*>(cache.data_ptr()), layout.num_pages,
                           layout.page_size, layout.page_stride_bytes, stream);
  } else {
    launch_quantize_append(static_cast<const bf16*>(latent_kv.data_ptr()),
                           static_cast<const int64_t*>(slot_mapping.data_ptr()), num_tokens,
                           static_cast<uint8_t*>(cache.data_ptr()), layout.num_pages,
                           layout.page_size, layout.page_stride_bytes, stream);
  }
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "MXFP8 sparse-MLA append launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_mxfp8_quantize_pack,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120Mxfp8QuantizePack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_mxfp8_quantize_append,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120Mxfp8QuantizeAppend);
