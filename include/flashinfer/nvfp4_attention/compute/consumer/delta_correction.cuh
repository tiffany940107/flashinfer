/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace sage {

using namespace cute;

/**
 * Delta_s Correction
 *
 * 职责:
 * 1. 从 shared memory 读取 Delta_s 值
 * 2. 将 Delta_s 添加到 attention scores (Q@K 的结果)
 *
 * 背景:
 * Delta_s 是用于修正量化误差的校正项。在计算 Q@K 时,由于使用了
 * FP4 量化,会引入量化误差。Delta_s 包含了预先计算好的校正值,
 * 用于补偿这个误差,提高最终的精度。
 *
 * 特点:
 * - Per-block 或 global 的 delta_s (由 BlockMean 控制)
 * - 使用 float4 向量化读取以提高效率
 * - 直接在 accumulator 上就地修改
 *
 * 模板参数:
 * @tparam Traits - Kernel 配置 traits
 */
template<typename Traits>
struct DeltaSCorrection {

    using TileShape_MNK = typename Traits::TileShape_MNK;
    using SmemLayoutDS = typename Traits::SmemLayoutDS;
    static constexpr bool BlockMean = Traits::BlockMean;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    /**
     * 添加 Delta_s 校正到 accumulator
     *
     * 此函数从 mainloop_tma_ws.h:691-704 提取
     *
     * Delta_s 的 layout:
     * - 形状: (kBlockM, kBlockN)
     * - 存储在 shared memory 中
     * - 使用 float4 向量化读取
     *
     * 算法:
     * 1. 将 accumulator 和 sDS 重新解释为 float4
     * 2. 根据 thread ID 计算起始位置 (quad_id)
     * 3. 循环读取 Delta_s 并添加到 accumulator
     * 4. 每个线程处理多个元素 (使用 quad 结构)
     *
     * 注意: 此函数中的 layout 和索引计算与 WGMMA 的线程布局紧密相关
     *
     * @param acc - Attention scores accumulator (Q@K 的结果),将被就地修改
     * @param sDS - Delta_s 的 shared memory tensor
     * @param smem_pipe_read_k - K 的 pipeline 读取状态 (用于选择 stage)
     */
    template<typename TensorAcc, typename TensorSDS, typename PipelineStateK>
    __device__ __forceinline__ static void add_delta_s(
        TensorAcc& acc,
        TensorSDS const& sDS,
        PipelineStateK const& smem_pipe_read_k
    ) {
        // 1. 重新解释为 float4 以进行向量化访问
        // 注意: recast 会改变 tensor 的元素数量
        auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
        auto acc_float4 = recast<float4>(acc);

        // 2. 计算当前线程在 quad 中的位置
        // threadIdx.x % 4 给出 quad 内的线程 ID (0-3)
        // 乘以 2 是因为每个线程处理 2 个 float4
        int quad_id = (threadIdx.x % 4) * 2;

        // 3. 循环处理 4 个 chunks (每个 chunk 包含多个元素)
        // 这个循环结构与 WGMMA 的 accumulator layout 相匹配
        for (int i = 0; i < 4; i++) {
            // 计算在 Delta_s tensor 中的索引
            auto num = quad_id + i * 8;

            // 读取两个 float4 (8 个 float)
            float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
            float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));

            // 将 delta_s 添加到 accumulator 的对应位置
            // 注意: 这里直接赋值而不是累加,因为在调用此函数前 acc 已经被初始化
            // 实际上 delta_s 是作为 bias 项添加的
            acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
            acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
            acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
            acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
        }
    }

    /**
     * Lambda wrapper: 创建可以被传递的 lambda
     *
     * 在主循环中,add_delta_s 需要作为 lambda 传递给 QK GEMM
     * 这个函数创建一个捕获必要参数的 lambda
     *
     * 用法:
     * ```cpp
     * auto add_delta_s_func = DeltaSCorrection::make_lambda(sDS, smem_pipe_read_k);
     * // 在 QK GEMM 中调用
     * add_delta_s_func(acc);
     * ```
     *
     * @param sDS - Delta_s 的 shared memory tensor
     * @param smem_pipe_read_k - K 的 pipeline 读取状态
     * @return Lambda 函数,可以用 accumulator 调用
     */
    template<typename TensorSDS, typename PipelineStateK>
    __device__ __forceinline__ static auto make_lambda(
        TensorSDS const& sDS,
        PipelineStateK const& smem_pipe_read_k
    ) {
        return [&sDS, &smem_pipe_read_k](auto& acc) {
            add_delta_s(acc, sDS, smem_pipe_read_k);
        };
    }

    /**
     * No-op lambda: 当不需要 delta_s 校正时使用
     *
     * 在某些配置下可能不需要 delta_s 校正,
     * 这个函数提供一个空操作的 lambda
     *
     * @return 空 lambda
     */
    __device__ __forceinline__ static auto make_noop_lambda() {
        return [](auto& acc) {
            // Do nothing
        };
    }
};

/**
 * 辅助函数: 根据配置选择 delta_s lambda
 *
 * 这个函数根据是否启用 delta_s 来选择合适的 lambda
 *
 * @tparam UseDeltaS - 是否使用 delta_s 校正
 * @param sDS - Delta_s 的 shared memory tensor
 * @param smem_pipe_read_k - K 的 pipeline 读取状态
 * @return 适当的 lambda 函数
 */
template<bool UseDeltaS, typename Traits, typename TensorSDS, typename PipelineStateK>
__device__ __forceinline__ auto make_delta_s_lambda(
    TensorSDS const& sDS,
    PipelineStateK const& smem_pipe_read_k
) {
    if constexpr (UseDeltaS) {
        return DeltaSCorrection<Traits>::make_lambda(sDS, smem_pipe_read_k);
    } else {
        return DeltaSCorrection<Traits>::make_noop_lambda();
    }
}

}  // namespace sage
