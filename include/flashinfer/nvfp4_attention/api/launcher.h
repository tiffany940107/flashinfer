/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"
#include "flashinfer/utils.cuh"

#include "../common/static_switch.h"
#include "../common/params.h"
#include "../kernel/scheduler.h"
#include "../kernel/attention_kernel.h"
#if defined(NON_WS)
#include "../kernel/attention_kernel_nonws.h"
#endif
#include "../kernel/traits.h"
#include "../compute/producer/load_q.cuh"
#include "../compute/producer/load_k.cuh"
#include "../compute/producer/load_v.cuh"
#include "../compute/epilogue/output_writer.cuh"
#include "../compute/epilogue/lse_writer.cuh"

namespace sage {

/**
 * Kernel Launcher
 *
 * 职责：
 * 1. 配置 kernel 参数
 * 2. 设置 shared memory 大小
 * 3. 计算 grid/block 维度
 * 4. 启动 kernel
 *
 * 模板参数：
 * @tparam Kernel_traits: Kernel 配置 traits
 * @tparam Is_causal: 是否使用 causal mask
 */
template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;
    using ElementSF = typename Kernel_traits::ElementSF;
    using ElementOut = typename Kernel_traits::ElementOut;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    // 定义 Collective 类型
    using CollectiveMainloop = sage::CollectiveMainloopFwd<Kernel_traits, Is_causal>;
    using CollectiveEpilogue = sage::CollectiveEpilogueFwd<Kernel_traits>;

    // 定义 Scheduler (使用 StaticPersistent 以获得更好的性能)
    using Scheduler = sage::StaticPersistentTileScheduler;

    // ============ 构建 Mainloop 参数 ============
    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            // Q tensor
            static_cast<Element const*>(params.q_ptr),
            {params.seqlen_q, params.d, params.h, params.b},  // shape_Q
            {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q

            // K tensor
            static_cast<Element const*>(params.k_ptr),
            {params.seqlen_k, params.d, params.h_k, params.b},  // shape_K
            {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
            {params.unpadded_seqlen_k, params.d, params.h_k, params.b},  // shape_K (unpadded)

            // V tensor (transposed)
            static_cast<Element const*>(params.v_ptr),
            {params.d, params.seqlen_k, params.h_k, params.b},  // shape_Vt
            {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_Vt

            // Scale factors
            static_cast<ElementSF const*>(params.sfq_ptr),
            {params.seqlen_q, params.d, params.h, params.b},  // shape_SFQ
            static_cast<ElementSF const*>(params.sfk_ptr),
            {params.seqlen_k, params.d, params.h_k, params.b},  // shape_SFK
            static_cast<ElementSF const*>(params.sfv_ptr),
            {params.d, params.seqlen_k, params.h_k, params.b},  // shape_SFVt

            // Delta_s correction
            static_cast<float const*>(params.delta_s_ptr),
            {params.seqlen_s, params.seqlen_k, params.h_k, params.b},
            {params.ds_row_stride, _1{}, params.ds_head_stride, params.ds_batch_stride},

            // Softmax scale
            params.scale_softmax_log2
        });

    // ============ 构建 Epilogue 参数 ============
    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({
            // O tensor
            static_cast<ElementOut*>(params.o_ptr),
            {params.seqlen_q, params.d, params.h, params.b},  // shape_O
            {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},  // stride_O

            // LSE (LogSumExp) tensor
            static_cast<float*>(params.softmax_lse_ptr),
            {_1{}, params.seqlen_q, params.h * params.seqlen_q},  // stride_LSE
        });

    // ============ 构建 Scheduler 参数 ============
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});

    typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // ============ 获取 Kernel 函数指针 ============
#if defined(NON_WS)
    void *kernel = (void *)sage::attention_kernel_nonws<Kernel_traits, Is_causal, Scheduler>;
#else
    void *kernel = (void *)sage::attention_kernel_ws<Kernel_traits, Is_causal, Scheduler>;
#endif

    // ============ 设置 Shared Memory 大小 ============
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    if (smem_size >= 48 * 1024) {
        // 如果 smem 超过 48KB，需要动态配置
        FLASHINFER_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        ));
    }

    // ============ 计算 Grid/Block 维度 ============
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;

    // 更新 params 中的一些参数
    params.m_block_divmod = cutlass::FastDivmod(num_blocks_m);
    params.total_blocks = num_blocks_m * params.h * params.b;

    // Grid 维度 (动态检测 SM 数量，适配不同 GPU)
    int device_id = 0;
    FLASHINFER_CUDA_CHECK(cudaGetDevice(&device_id));
    int num_sms = 0;
    FLASHINFER_CUDA_CHECK(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, num_sms);
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));

    // ============ 启动 Kernel ============
    cutlass::ClusterLaunchParams launch_params{
        grid_dims, block_dims, cluster_dims, smem_size, stream
    };

    cutlass::launch_kernel_on_cluster(
        launch_params, kernel,
        params, mainloop_params, epilogue_params, scheduler_params
    );

    // 检查 kernel 启动错误
    FLASHINFER_CUDA_CHECK(cudaGetLastError());
}

/**
 * MHA Forward Dispatcher
 *
 * 根据参数分发到不同的 kernel 配置
 *
 * 模板参数：
 * @tparam T: FP4 pair type
 * @tparam Headdim: Head dimension (64 or 128)
 * @tparam O: Output type (bfloat16 or float16)
 */
template<typename T, int Headdim, typename O = cutlass::bfloat16_t>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    // 根据 causal 分支
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // 根据 per_block_mean 分支
        BOOL_SWITCH(params.per_block_mean, per_block, [&] {
            if constexpr (Headdim == 64 || Headdim == 128) {
                // 选择 Kernel traits
                // kBlockM=128, kBlockN=128, kStages=3
                run_flash_fwd<
                    Flash_fwd_kernel_traits<Headdim, 128, 128, 3, 1, per_block, T, O>,
                    Is_causal
                >(params, stream);
            } else {
                static_assert(Headdim == 64 || Headdim == 128, "Unsupported Headdim");
            }
        });
    });
}

/**
 * 配置说明：
 *
 * Kernel Traits 参数：
 * - Headdim: Head dimension (64 or 128)
 * - kBlockM: 128 (M 维度 tile 大小)
 * - kBlockN: 128 (N 维度 tile 大小)
 * - kStages: 3 (Pipeline stages for K/V)
 * - kClusterM: 1 (Cluster size, 通常为 1)
 * - per_block: 是否使用 per-block mean
 * - T: FP4 pair type
 * - O: Output type (BF16/FP16)
 *
 * Scheduler:
 * - StaticPersistentTileScheduler: 每个 SM 处理多个 tiles
 * - Grid size: 170 (H100 的 SM 数量)
 *
 * Shared Memory:
 * - 自动计算所需的 smem 大小
 * - 如果超过 48KB，动态配置
 *
 * 性能优化：
 * - 使用 persistent kernel 减少启动开销
 * - TMA pipeline 隐藏内存延迟
 * - Warp specialization 提高并行度
 */

} // namespace sage
