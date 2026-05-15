/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "../../utils/layout.cuh"
#include "../../utils/math.cuh"

namespace sage {

using namespace cute;

/**
 * Fused Online Softmax with Quantization
 *
 * 职责:
 * 1. 在线计算 softmax (online softmax algorithm)
 * 2. 同时进行量化到 FP4 (E2M1)
 * 3. 计算 scale factors (FP8 E4M3)
 * 4. 维护 running max 和 running sum
 *
 * 特点:
 * - 采用 online algorithm,支持流式计算
 * - 第一个 tile 初始化 max/sum
 * - 后续 tile 更新 max/sum 并重新缩放
 * - 直接量化 P 矩阵为 FP4
 *
 * 模板参数:
 * @tparam Rows - 处理的行数 (通常是 kBlockM)
 */
template <int Rows>
struct SoftmaxFused {

    // 状态变量
    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    TensorT row_sum;        // 每行的累积和
    TensorT row_max;        // 每行的当前最大值
    TensorT scores_scale;   // 用于重新缩放的 scale factor

    // 量化常数
    static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;  // log2(fp8_scalexfp4_scale)
    static constexpr float fp4_scale_log2 = -2.584962500721156f;  // log2(1/6) - FP4 的 scale
    static constexpr int RowReductionThr = 4;  // Row reduction 的线程数

    /**
     * 构造函数
     */
    CUTLASS_DEVICE SoftmaxFused() {};

    /**
     * 在线 Softmax + 量化
     *
     * 此函数从 sageattn3/blackwell/softmax_fused.h 中提取
     *
     * 实现 online softmax algorithm:
     * - 对于第一个 tile: 初始化 max, sum
     * - 对于后续 tile: 更新 max, 重新缩放 sum 和之前的结果
     *
     * 同时将 P 矩阵量化为 FP4 (E2M1),scale factors 为 FP8 (E4M3)
     *
     * @tparam FirstTile - 是否是第一个 tile
     * @tparam InfCheck - 是否检查 -INFINITY (用于 causal masking)
     * @param acc - 累加器 (Q@K 的结果),将被就地修改为量化后的值
     * @param AbsMaxP - 输出的 scale factors (每 8 个元素一个)
     * @param softmax_scale_log2 - Softmax 的 scale (log2 域)
     */
    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        // 创建不同的视图用于不同的操作
        // reduction_view: 用于计算 max 和 sum
        // conversion_view: 用于量化
        Tensor acc_reduction_view = make_tensor(
            acc.data(),
            sage::convert_to_reduction_layout(acc.layout())
        );

        Tensor acc_conversion_view = make_tensor(acc.data(), sage::convert_to_conversion_layout(acc.layout()));
        // 确保按步骤完成 flatten 和 group_modes
        auto temp1 = flatten(acc_conversion_view);
        auto temp2 = group_modes<0, 2>(temp1);
        auto acc_conversion_flatten = group_modes<1, 5>(temp2);

        if constexpr (FirstTile) {
            // === 第一个 tile: 初始化 ===
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            // 计算每行的最大值
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    // 计算每个 chunk 的最大值
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(
                            AbsMaxP(mi, ni),
                            acc_reduction_view(mi, make_coord(ei, ni))
                        );
                    }
                    // Warp shuffle: 交换相邻线程的最大值
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1);
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }

                // Warp shuffle: 在 quad 内交换最大值
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // 计算 scaled max (在 log2 域)
                const float max_scaled = InfCheck
                    ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                    : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                // 计算 exp(x - max) (使用 exp2 for efficiency)
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = ptx_exp2(
                        acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled
                    );
                }

                // 计算 scale factors (for 量化)
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = ptx_exp2(
                        AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2
                    );
                }
            }

            // 计算每行的 sum
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            // === 后续 tile: 更新 max 和 sum ===
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                // 计算新的最大值
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(
                            local_max,
                            acc_reduction_view(mi, make_coord(ei, ni))
                        );
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }

                // Warp shuffle: 在 quad 内交换最大值
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // 计算重新缩放的 scale factor
                float scores_max_cur = !InfCheck
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = ptx_exp2(
                    (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2
                );

                // 计算 scaled max
                const float max_scaled = InfCheck
                    ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                    : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                // 重新缩放之前的 sum
                row_sum(mi) = row_sum(mi) * scores_scale(mi);

                // 计算 exp(x - max) 并累加到 sum
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = ptx_exp2(
                        acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled
                    );
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }

                // 计算 scale factors
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = ptx_exp2(
                        AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2
                    );
                }
            }
        }

        // 量化: 先预计算所有 scale 倒数，再统一乘法
        Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            inv_AbsMaxP(i) = 1.0f / AbsMaxP(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
                acc_conversion_flatten(j, i) *= inv_AbsMaxP(i);
            }
        }
    }

#if defined(FP16_SOFTMAX)
    /**
     * FP16 Online Softmax + 量化
     *
     * 与 online_softmax_with_quant 功能相同，但 tSrS 输入先 cvt 为 FP16，
     * softmax 的 find_max / sum / rescale 在 FP16 下计算 (MUFU exp2 仍用 FP32)。
     * 量化阶段 (inv + mul) 也在 FP16 下进行。
     *
     * 精度已验证: FP16 vs FP32 softmax cos_sim >= 0.99999988 (所有配置)。
     * 寄存器收益: tSrS 从 64 FP32 regs → 32 FP16 regs = 省 32 regs。
     *
     * 调用方式: mainloop 中先把 FP32 acc cvt 为 FP16，然后调用此函数。
     * acc 在调用前已就地转换为 FP16 (reinterpret_cast<__half*>)。
     *
     * 注意: acc 的底层存储是 float 类型 (MMA accumulator)，但实际值
     * 已被转换为 half 并存储在 float 的低 16 bits 中。这里通过
     * recast<__half> 读写 FP16 值。
     */
    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant_fp16(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        // 创建 reduction view (与 FP32 版本相同的 layout)
        Tensor acc_reduction_view = make_tensor(
            acc.data(),
            sage::convert_to_reduction_layout(acc.layout())
        );
        Tensor acc_conversion_view = make_tensor(acc.data(), sage::convert_to_conversion_layout(acc.layout()));
        auto acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));

        // FP16 状态变量 (row_max, row_sum, scores_scale 仍用 FP32 以避免 overflow)
        // MUFU exp2 也仍用 FP32。只有 acc 数据和 FMA 操作用 FP16。

        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                // find_max: FP32 (从 FP16 acc 读取后 cvt 为 FP32 做 fmaxf)
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(
                            AbsMaxP(mi, ni),
                            acc_reduction_view(mi, make_coord(ei, ni))
                        );
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1);
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }

                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                const float max_scaled = InfCheck
                    ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                    : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                // exp2: 读 FP32 acc → MUFU exp2 (FP32) → 写回 FP32 acc
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = ptx_exp2(
                        acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled
                    );
                }

                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = ptx_exp2(
                        AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2
                    );
                }
            }

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(
                            local_max,
                            acc_reduction_view(mi, make_coord(ei, ni))
                        );
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }

                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                float scores_max_cur = !InfCheck
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = ptx_exp2(
                    (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2
                );

                const float max_scaled = InfCheck
                    ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                    : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

                row_sum(mi) = row_sum(mi) * scores_scale(mi);

                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = ptx_exp2(
                        acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled
                    );
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }

                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = ptx_exp2(
                        AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2
                    );
                }
            }
        }

        // 量化: 与 FP32 版本相同
        Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            inv_AbsMaxP(i) = 1.0f / AbsMaxP(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
                acc_conversion_flatten(j, i) *= inv_AbsMaxP(i);
            }
        }
    }
#endif  // FP16_SOFTMAX

    /**
     * Finalize: 除以 sum,得到最终的 softmax 结果
     *
     * 此函数在所有 tiles 处理完后调用,将累积的输出除以 sum
     *
     * @param o_store - 输出张量 (累积的 O),将被就地修改
     */
    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(
            o_store.data(),
            convert_to_reduction_layout(o_store.layout())
        );

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            // Warp reduction: 计算总的 sum
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }

            float sum = row_sum(mi);
            // 处理特殊情况: sum = 0 或 NaN
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;

            // 除以 sum
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
                o_store_reduction_view(mi, ni) *= inv_sum;
            }
        }
    }

    /**
     * Rescale O: 重新缩放之前的输出并累加新的输出
     *
     * 此函数在处理非第一个 tile 时调用
     * O_new = O_old * scale + O_current
     *
     * @param o_store - 累积的输出 (O_old),将被更新
     * @param o_tmp - 当前 tile 的输出 (O_current)
     */
    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(
            o_store.data(),
            sage::convert_to_reduction_layout(o_store.layout())
        );
        Tensor o_tmp_reduction_view = make_tensor(
            o_tmp.data(),
            sage::convert_to_reduction_layout(o_tmp.layout())
        );

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) {
                o_store_reduction_view(mi, ni) =
                    o_store_reduction_view(mi, ni) * scores_scale(mi) +
                    o_tmp_reduction_view(mi, ni);
            }
        }
    }

    /**
     * R17 方案 B: 两遍 partial softmax
     *
     * Pass 1 (find_max_chunk): 只求 max，不算 exp2。纯 FMA，可与 GEMM 交错。
     * Pass 2 (exp2_sum_chunk): 用 global max 算 exp2+sum+SF。含 MUFU。
     *
     * 控制逻辑 (init, rescale) 在 mainloop 中处理，这里只做纯计算。
     */

    /**
     * Pass 1: 求一个 N-sub-tile 的 chunk max，更新 running row_max。
     * 纯 FMA (fmaxf + shuffle)，无 MUFU。约 12 条指令/chunk。
     */
    template<bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE void find_max_chunk(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        int ni
    ) {
        Tensor acc_rv = make_tensor(
            acc.data(),
            sage::convert_to_reduction_layout(acc.layout())
        );

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_rv); mi++) {
            float chunk_max = -INFINITY;
            CUTLASS_PRAGMA_UNROLL
            for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                chunk_max = fmaxf(chunk_max, acc_rv(mi, make_coord(ei, ni)));
            }
            float max_recv = __shfl_xor_sync(int32_t(-1), chunk_max, 1);
            chunk_max = fmaxf(chunk_max, max_recv);
            AbsMaxP(mi, ni) = chunk_max;

            row_max(mi) = fmaxf(row_max(mi), chunk_max);
            max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
            row_max(mi) = fmaxf(row_max(mi), max_recv);
        }
    }

    /**
     * Pass 2: 用已确定的 global row_max 算一个 sub-tile 的 exp2 + sum + SF。
     * 含 MUFU (exp2)。需要所有 find_max_chunk 完成后 row_max 已是 global max。
     */
    template<bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE void exp2_sum_chunk(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        int ni,
        const float softmax_scale_log2
    ) {
        Tensor acc_rv = make_tensor(
            acc.data(),
            sage::convert_to_reduction_layout(acc.layout())
        );

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_rv); mi++) {
            const float max_scaled = InfCheck
                ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

            CUTLASS_PRAGMA_UNROLL
            for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                float val = ptx_exp2(
                    acc_rv(mi, make_coord(ei, ni)) * softmax_scale_log2 - max_scaled
                );
                acc_rv(mi, make_coord(ei, ni)) = val;
                row_sum(mi) += val;
            }

            AbsMaxP(mi, ni) = ptx_exp2(
                AbsMaxP(mi, ni) * softmax_scale_log2 - max_scaled + fp4_scale_log2
            );
        }
    }

    /**
     * R17: Quantize after all partial softmax chunks are done.
     * Separated from softmax because quantization needs all AbsMaxP values.
     */
    template<typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE void quantize_after_partial_softmax(
        TensorAcc& acc,
        TensorMax& AbsMaxP
    ) {
        Tensor acc_cv = make_tensor(acc.data(), sage::convert_to_conversion_layout(acc.layout()));
        auto temp1 = flatten(acc_cv);
        auto temp2 = group_modes<0, 2>(temp1);
        auto acc_flat = group_modes<1, 5>(temp2);

        Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            inv_AbsMaxP(i) = 1.0f / AbsMaxP(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_flat); ++j) {
                acc_flat(j, i) *= inv_AbsMaxP(i);
            }
        }
    }

    /**
     * R17 fixed: Two-pass chunked softmax that correctly handles AbsMaxP layout.
     *
     * The original find_max_chunk/exp2_sum_chunk wrote AbsMaxP(mi, ni) using
     * reduction-layout ni, but the quantize lambda reads AbsMaxP through
     * conversion layout. This method does all softmax + AbsMaxP construction
     * in a single call, using the same approach as online_softmax_with_quant
     * but organized in two passes for future GEMM-softmax interleaving.
     *
     * @param is_first - true for first N-block of the tile
     * @param softmax_scale_log2 - log2(softmax_scale)
     */
    template<bool InfCheck = false, typename TensorAcc, typename TensorMax, typename TensorPrev>
    CUTLASS_DEVICE void chunked_softmax_fixed(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        bool is_first,
        const float softmax_scale_log2,
        TensorPrev const& scores_max_prev
    ) {
        Tensor acc_rv = make_tensor(
            acc.data(), sage::convert_to_reduction_layout(acc.layout()));

        Tensor acc_cv = make_tensor(
            acc.data(), sage::convert_to_conversion_layout(acc.layout()));
        auto acc_cv_flat = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_cv)));

        constexpr int MmaN = decltype(size<1, 1>(acc_rv))::value;

        // --- Pass 1: Find row_max across ALL chunks ---
        // (This mirrors online_softmax_with_quant's max-finding loop exactly)
        if (is_first) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_rv); mi++) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < MmaN; ni++) {
                float local_max = -INFINITY;
                CUTLASS_PRAGMA_UNROLL
                for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                    local_max = fmaxf(local_max, acc_rv(mi, make_coord(ei, ni)));
                }
                float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
                AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
            }
            float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
            row_max(mi) = fmaxf(row_max(mi), max_recv);
        }

        // --- Cross-block rescale (non-first tile) ---
        // Must happen between max finding and exp2 computation, same as default path.
        if (!is_first) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_rv); mi++) {
                float scores_max_cur = !InfCheck ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = ptx_exp2(
                    (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }

        // --- Pass 2: exp2 + sum + AbsMaxP transform ---
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_rv); mi++) {
            const float max_scaled = InfCheck
                ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

            // exp2 on all elements (flat, same as default path)
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_rv); ni++) {
                float val = ptx_exp2(acc_rv(mi, ni) * softmax_scale_log2 - max_scaled);
                acc_rv(mi, ni) = val;
                row_sum(mi) += val;
            }

            // Transform AbsMaxP (same loop as default path line 137-141)
            CUTLASS_PRAGMA_UNROLL
            for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                AbsMaxP(mi, sfi) = ptx_exp2(
                    AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
            }
        }

        // --- Quantize: divide acc by AbsMaxP in conversion-layout order ---
        // (Same as online_softmax_with_quant lines 216-227)
        Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            inv_AbsMaxP(i) = 1.0f / AbsMaxP(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_cv_flat); ++j) {
                acc_cv_flat(j, i) *= inv_AbsMaxP(i);
            }
        }
    }

    /**
     * R17 Direction C: exp2 + sum + AbsMaxP transform + quantize.
     *
     * Called AFTER all find_max_chunk() calls have populated AbsMaxP(mi, ni)
     * and row_max(mi). Does cross-block rescale, exp2, row_sum accumulation,
     * AbsMaxP transformation, and quantization.
     *
     * Uses the same layout handling as chunked_softmax_fixed (proven correct).
     *
     * @param is_first - true for first N-block of the tile
     * @param softmax_scale_log2 - log2(softmax_scale)
     * @param scores_max_prev - row_max snapshot before find_max (for rescale)
     */
    template<bool InfCheck = false, typename TensorAcc, typename TensorMax, typename TensorPrev>
    CUTLASS_DEVICE void exp2_sum_and_quantize(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        bool is_first,
        const float softmax_scale_log2,
        TensorPrev const& scores_max_prev
    ) {
        Tensor acc_rv = make_tensor(
            acc.data(), sage::convert_to_reduction_layout(acc.layout()));
        Tensor acc_cv = make_tensor(
            acc.data(), sage::convert_to_conversion_layout(acc.layout()));
        auto acc_cv_flat = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_cv)));

        // --- Cross-block rescale (non-first tile) ---
        if (!is_first) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_rv); mi++) {
                float scores_max_cur = !InfCheck ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = ptx_exp2(
                    (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }

        // --- exp2 + sum + AbsMaxP transform ---
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_rv); mi++) {
            const float max_scaled = InfCheck
                ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_rv); ni++) {
                float val = ptx_exp2(acc_rv(mi, ni) * softmax_scale_log2 - max_scaled);
                acc_rv(mi, ni) = val;
                row_sum(mi) += val;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                AbsMaxP(mi, sfi) = ptx_exp2(
                    AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
            }
        }

        // --- Quantize: divide acc by AbsMaxP in conversion-layout order ---
        Tensor inv_AbsMaxP = make_tensor_like<float>(AbsMaxP.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            inv_AbsMaxP(i) = 1.0f / AbsMaxP(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(inv_AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_cv_flat); ++j) {
                acc_cv_flat(j, i) *= inv_AbsMaxP(i);
            }
        }
    }

    /**
     * R17 方案 A: 单遍 online softmax chunk (find_max + exp2 + rescale)
     *
     * 合并 find_max 和 exp2 为一个方法。当 row_max 增加时，rescale 之前所有
     * sub-tile 的 exp2 值和 AbsMaxP。这样 exp2 可以在 GEMM gap 中执行。
     *
     * @param ni - 当前处理的 N-sub-tile (0..MmaN-1)
     * @param softmax_scale_log2 - log2(softmax_scale)
     */
    template<bool IsInit, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE void online_softmax_chunk(
        TensorAcc& acc,
        TensorMax& AbsMaxP,
        int ni,
        const float softmax_scale_log2
    ) {
        Tensor acc_rv = make_tensor(
            acc.data(),
            sage::convert_to_reduction_layout(acc.layout())
        );

        if constexpr (IsInit) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_rv); mi++) {
            // --- Find chunk max ---
            float chunk_max = -INFINITY;
            CUTLASS_PRAGMA_UNROLL
            for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                chunk_max = fmaxf(chunk_max, acc_rv(mi, make_coord(ei, ni)));
            }
            float max_recv = __shfl_xor_sync(int32_t(-1), chunk_max, 1);
            chunk_max = fmaxf(chunk_max, max_recv);
            AbsMaxP(mi, ni) = chunk_max;

            // --- Update row_max ---
            float prev_max = row_max(mi);
            row_max(mi) = fmaxf(row_max(mi), chunk_max);
            max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2);
            row_max(mi) = fmaxf(row_max(mi), max_recv);

            // --- max_scaled for exp2 (using updated global max) ---
            const float max_scaled = InfCheck
                ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);

            // --- Rescale previous sub-tiles ONLY if max actually changed ---
            // All 4 threads sharing a row see the same row_max (after shuffle),
            // so this branch is uniform within the warp → no divergence.
            if constexpr (!IsInit) {
                if (prev_max != row_max(mi)) {
                    scores_scale(mi) = ptx_exp2(
                        (prev_max - row_max(mi)) * softmax_scale_log2
                    );
                    row_sum(mi) *= scores_scale(mi);
                    CUTLASS_PRAGMA_UNROLL
                    for (int prev_ni = 0; prev_ni < ni; prev_ni++) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                            acc_rv(mi, make_coord(ei, prev_ni)) *= scores_scale(mi);
                        }
                        AbsMaxP(mi, prev_ni) *= scores_scale(mi);
                    }
                }
            }

            // --- Compute exp2 for current sub-tile ---
            CUTLASS_PRAGMA_UNROLL
            for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                float val = ptx_exp2(
                    acc_rv(mi, make_coord(ei, ni)) * softmax_scale_log2 - max_scaled
                );
                acc_rv(mi, make_coord(ei, ni)) = val;
                row_sum(mi) += val;
            }

            // --- Scale factor for quantization ---
            AbsMaxP(mi, ni) = ptx_exp2(
                AbsMaxP(mi, ni) * softmax_scale_log2 - max_scaled + fp4_scale_log2
            );
        }
    }

private:
    /**
     * PTX exp2 指令
     */
    __device__ __forceinline__ static float ptx_exp2(float x) {
        float result;
        asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
        return result;
    }
};

}  // namespace sage
