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
 * LSE Writer (Log-Sum-Exp)
 *
 * 职责:
 * 1. 将 LSE (log-sum-exp) 值写入到 global memory
 * 2. 用于后续的 backward pass
 *
 * 背景:
 * LSE = log(sum(exp(x_i))) 是 softmax 计算中的中间结果
 * 在 forward pass 中保存 LSE 可以简化 backward pass 的计算
 *
 * LSE 的计算公式:
 * LSE = max + log(sum)
 * 其中 max 是 row_max, sum 是 row_sum
 *
 * 特点:
 * - 每行一个 LSE 值
 * - 使用普通的 global memory 写入 (不使用 TMA)
 * - 需要处理边界情况
 * - 对于空行,LSE = +INFINITY
 *
 * 模板参数:
 * @tparam Traits - Kernel 配置 traits
 */
template<typename Traits>
struct LSEWriter {

    using TileShape_MNK = typename Traits::TileShape_MNK;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kNWarps = Traits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

    // LSE tensor 的形状和 stride
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;       // (seqlen_q, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;          // (seqlen_q, head, batch)

    /**
     * 写入 LSE 到 global memory
     *
     * 此函数基于 epilogue_tma_ws.h:148-166 (注释掉的代码)
     *
     * 算法:
     * 1. 从 softmax state 中提取 row_max 和 row_sum
     * 2. 计算 LSE = row_max + log(row_sum)
     * 3. 只有特定的线程 (get<1>(coord) == 0) 写入 LSE
     * 4. 处理边界情况
     *
     * 注意: LSE 在 log2 域中计算,需要转换为自然对数
     *
     * @param ptr_LSE - LSE 的 global memory 指针
     * @param shape_LSE - LSE 的形状
     * @param stride_LSE - LSE 的 stride
     * @param softmax_fused - Softmax state (包含 row_max 和 row_sum)
     * @param softmax_scale_log2 - Softmax scale (log2 域)
     * @param tiled_mma - Tiled MMA 对象 (用于确定线程 layout)
     * @param thread_idx - 线程索引
     * @param m_block - M block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     */
    template<typename SoftmaxFused, typename TiledMma, typename Shape, typename Stride>
    __device__ __forceinline__ static void write_lse(
        float* ptr_LSE,
        Shape const& shape_LSE,
        Stride const& stride_LSE,
        SoftmaxFused const& softmax_fused,
        float softmax_scale_log2,
        TiledMma const& tiled_mma,
        int thread_idx,
        int m_block,
        int bidh,
        int bidb
    ) {
        // 1. 创建 global memory LSE tensor
        Tensor mLSE = make_tensor(make_gmem_ptr(ptr_LSE), shape_LSE, stride_LSE);
        Tensor gLSE = local_tile(mLSE(_, bidh, bidb), cute::Shape<cute::Int<kBlockM>>{}, make_coord(m_block));

        // 2. 从 softmax state 中获取 row_max 和 row_sum
        auto const& row_max = softmax_fused.row_max;
        auto const& row_sum = softmax_fused.row_sum;

        // 3. 创建坐标 tensor 用于确定哪些线程写入
        Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);  // (MMA, MMA_M, MMA_K)

        // 验证 layout 假设
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);

        // taccOcO 的形状是 ((2, 2, V), MMA_M, MMA_K)
        // 我们只取行索引
        Tensor taccOcO_row = taccOcO(make_coord(_0{}, _), _, _0{});
        CUTE_STATIC_ASSERT_V(size(row_max) == size(taccOcO_row));  // MMA_M

        // 4. 计算并写入 LSE
        // 只有列索引为 0 的线程写入 (避免重复写入)
        if (get<1>(taccOcO_row(_0{})) == 0) {
            constexpr float log2_e = 1.44269504088896340736f;  // log2(e) = 1/ln(2)
            constexpr float ln_2 = 0.69314718055994530942f;    // ln(2)

            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                const int row = get<0>(taccOcO_row(mi));

                // 边界检查
                if (row < get<0>(shape_LSE) - m_block * kBlockM) {
                    // 计算 LSE
                    // LSE = max * softmax_scale + log(sum)
                    // 注意: softmax_scale_log2 是在 log2 域,需要转换
                    float max_scaled = row_max(mi) * softmax_scale_log2 / log2_e;  // 转换为自然对数域
                    float sum = row_sum(mi);

                    // 处理特殊情况: sum = 0 或 NaN (空行)
                    float lse = (sum == 0.f || sum != sum) ?
                        INFINITY :
                        (max_scaled + logf(sum));

                    gLSE(row) = lse;
                }
            }
        }
    }

    /**
     * 将 LSE 置为 +INFINITY (用于 padding 的 blocks)
     *
     * 此函数从 epilogue_tma_ws.h:217 提取
     *
     * 对于超出序列长度的 blocks,LSE 应该是 +INFINITY
     * 这表示这些位置不包含有效数据
     *
     * @param ptr_LSE - LSE 指针
     * @param shape_O - 输出形状 (用于提取 shape_LSE)
     * @param stride_LSE - LSE stride
     * @param thread_idx - 线程索引
     * @param m_block - M block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     */
    template<typename ShapeO, typename Stride>
    __device__ __forceinline__ static void write_lse_infinity(
        float* ptr_LSE,
        ShapeO const& shape_O,
        Stride const& stride_LSE,
        int thread_idx,
        int m_block,
        int bidh,
        int bidb
    ) {
        // 1. 从 shape_O 中提取 shape_LSE
        auto shape_LSE = select<0, 2, 3>(shape_O);  // (seqlen_q, head, batch)

        // 2. 创建 global memory LSE tensor
        Tensor mLSE = make_tensor(make_gmem_ptr(ptr_LSE), shape_LSE, stride_LSE);
        Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

        // 3. 每个线程写入一个 LSE (简化的 layout)
        // 假设: kBlockM <= NumMmaThreads,所以每个线程最多写入一个
        static_assert(kBlockM <= NumMmaThreads);

        // 边界检查并写入
        if (thread_idx < get<0>(shape_LSE) - m_block * kBlockM) {
            gLSE(thread_idx) = INFINITY;
        }
    }

    /**
     * 完整的 LSE 写入流程
     *
     * 根据是否是 padding block,选择写入正常值或 INFINITY
     *
     * @param ptr_LSE - LSE 指针
     * @param shape_O - 输出形状
     * @param shape_LSE - LSE 形状
     * @param stride_LSE - LSE stride
     * @param softmax_fused - Softmax state
     * @param softmax_scale_log2 - Softmax scale
     * @param tiled_mma - Tiled MMA
     * @param thread_idx - 线程索引
     * @param m_block - M block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     * @param is_valid_block - 是否是有效的 block (不是 padding)
     */
    template<
        typename ShapeO,
        typename ShapeLSE,
        typename Stride,
        typename SoftmaxFused,
        typename TiledMma
    >
    __device__ __forceinline__ static void run(
        float* ptr_LSE,
        ShapeO const& shape_O,
        ShapeLSE const& shape_LSE,
        Stride const& stride_LSE,
        SoftmaxFused const& softmax_fused,
        float softmax_scale_log2,
        TiledMma const& tiled_mma,
        int thread_idx,
        int m_block,
        int bidh,
        int bidb,
        bool is_valid_block
    ) {
        if (is_valid_block) {
            // 写入正常的 LSE 值
            write_lse(
                ptr_LSE, shape_LSE, stride_LSE,
                softmax_fused, softmax_scale_log2,
                tiled_mma, thread_idx,
                m_block, bidh, bidb
            );
        } else {
            // 写入 INFINITY (padding block)
            write_lse_infinity(
                ptr_LSE, shape_O, stride_LSE,
                thread_idx, m_block, bidh, bidb
            );
        }
    }
};

/**
 * 辅助函数: 计算 log2 域的 LSE
 *
 * 有时我们需要在 log2 域中保持 LSE (例如用于某些优化)
 * 这个函数计算 LSE 但保持在 log2 域
 *
 * @param row_max - 行最大值 (log2 域)
 * @param row_sum - 行和
 * @param softmax_scale_log2 - Softmax scale (log2 域)
 * @return LSE in log2 domain
 */
__device__ __forceinline__ float compute_lse_log2(
    float row_max,
    float row_sum,
    float softmax_scale_log2
) {
    constexpr float log2_e = 1.44269504088896340736f;  // log2(e)

    // 处理特殊情况
    if (row_sum == 0.f || row_sum != row_sum) {
        return INFINITY;
    }

    // LSE (log2 domain) = max * scale + log2(sum)
    float max_scaled = row_max * softmax_scale_log2;
    float lse_log2 = max_scaled + log2f(row_sum);

    return lse_log2;
}

/**
 * 辅助函数: 从 log2 域转换到自然对数域
 */
__device__ __forceinline__ float log2_to_ln(float x_log2) {
    constexpr float ln_2 = 0.69314718055994530942f;  // ln(2)
    return x_log2 * ln_2;
}

/**
 * 辅助函数: 从自然对数域转换到 log2 域
 */
__device__ __forceinline__ float ln_to_log2(float x_ln) {
    constexpr float log2_e = 1.44269504088896340736f;  // log2(e) = 1/ln(2)
    return x_ln * log2_e;
}

}  // namespace sage
