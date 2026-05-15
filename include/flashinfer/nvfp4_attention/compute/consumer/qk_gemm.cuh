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
 * QK GEMM Computer
 *
 * 职责:
 * 1. 计算 Q @ K^T GEMM (Attention Scores)
 * 2. 从 shared memory 拷贝 Q/K 到 register
 * 3. 从 shared memory 拷贝 scale factors (SFQ/SFK)
 * 4. 执行 blocked GEMM with FP4 inputs
 * 5. 应用 causal masking
 *
 * 特点:
 * - 使用 OMMA 指令 (FP4 Block GEMM)
 * - FP4 输入 + FP8 scale factors,FP32 累加器
 * - 支持 causal 和 non-causal attention
 * - 输出是 S = Q @ K^T (未归一化的 attention scores)
 *
 * 模板参数:
 * @tparam Traits - Kernel 配置 traits
 * @tparam IsCausal - 是否使用 causal masking
 */
template<typename Traits, bool IsCausal>
struct QKGemmComputer {

    using Element = typename Traits::Element;
    using ElementSF = typename Traits::ElementSF;
    using TileShape_MNK = typename Traits::TileShape_MNK;
    using TiledMmaQK = typename Traits::TiledMmaQK;
    using SmemCopyAtomQ = typename Traits::SmemCopyAtomQ;
    using SmemCopyAtomKV = typename Traits::SmemCopyAtomKV;
    using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kBlockK = get<2>(TileShape_MNK{});

    /**
     * 从 shared memory 拷贝 K block 到 register
     *
     * 此函数从 mainloop_tma_ws.h:672-677 提取
     *
     * @param smem_tiled_copy_K - K 的 tiled copy 对象
     * @param smem_tiled_copy_SFK - SFK 的 tiled copy 对象
     * @param tSsK - K 的 shared memory tensor (source)
     * @param tSsSFK - SFK 的 shared memory tensor (source)
     * @param tSrK_copy_view - K 的 register tensor (destination)
     * @param tSrSFK_copy_view - SFK 的 register tensor (destination)
     * @param smem_pipe_read_k - Pipeline 读取状态
     * @param block_id - 当前处理的 K block ID
     */
    template<
        typename SmemTiledCopyK,
        typename SmemTiledCopySFK,
        typename TensorSsK,
        typename TensorSsSFK,
        typename TensorSrK,
        typename TensorSrSFK,
        typename PipelineStateK
    >
    __device__ __forceinline__ static void copy_k_block(
        SmemTiledCopyK const& smem_tiled_copy_K,
        SmemTiledCopySFK const& smem_tiled_copy_SFK,
        TensorSsK const& tSsK,
        TensorSsSFK const& tSsSFK,
        TensorSrK& tSrK_copy_view,
        TensorSrSFK& tSrSFK_copy_view,
        PipelineStateK const& smem_pipe_read_k,
        auto block_id
    ) {
        // 获取当前 pipeline stage 的数据
        auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
        auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());

        // 从 shared memory 拷贝到 register
        copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
        copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
    }

    /**
     * 计算 Q @ K^T GEMM
     *
     * 此函数从 mainloop_tma_ws.h:720-729 提取
     *
     * 执行 blocked GEMM:
     * - 输入: Q (FP4), K (FP4), SFQ (FP8), SFK (FP8)
     * - 输出: S (FP32 accumulator)
     * - 算法: S += (Q * SFQ) @ (K * SFK)^T
     *
     * @param tiled_mma_qk - QK GEMM 的 tiled MMA 对象
     * @param tSrQ - Q 的 register tensor
     * @param tSrSFQ - SFQ 的 register tensor
     * @param tSrK - K 的 register tensor
     * @param tSrSFK - SFK 的 register tensor
     * @param tSrS - S 的 accumulator (输出)
     * @param smem_tiled_copy_K - K 的 tiled copy (用于预取)
     * @param smem_tiled_copy_SFK - SFK 的 tiled copy (用于预取)
     * @param tSsK - K 的 shared memory tensor
     * @param tSsSFK - SFK 的 shared memory tensor
     * @param tSrK_copy_view - K 的 register copy view
     * @param tSrSFK_copy_view - SFK 的 register copy view
     * @param smem_pipe_read_k - Pipeline 读取状态
     */
    template<
        typename TiledMma,
        typename TensorRQ,
        typename TensorRSFQ,
        typename TensorRK,
        typename TensorRSFK,
        typename TensorRS,
        typename SmemTiledCopyK,
        typename SmemTiledCopySFK,
        typename TensorSsK,
        typename TensorSsSFK,
        typename TensorRKView,
        typename TensorRSFKView,
        typename PipelineStateK
    >
    __device__ __forceinline__ static void compute_qk_gemm(
        TiledMma const& tiled_mma_qk,
        TensorRQ const& tSrQ,
        TensorRSFQ const& tSrSFQ,
        TensorRK const& tSrK,
        TensorRSFK const& tSrSFK,
        TensorRS& tSrS,
        SmemTiledCopyK const& smem_tiled_copy_K,
        SmemTiledCopySFK const& smem_tiled_copy_SFK,
        TensorSsK const& tSsK,
        TensorSsSFK const& tSsSFK,
        TensorRKView& tSrK_copy_view,
        TensorRSFKView& tSrSFK_copy_view,
        PipelineStateK const& smem_pipe_read_k
    ) {
        // Blocked GEMM: 循环处理所有 K blocks
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
            // 执行 GEMM: S += Q @ K^T
            // 使用 zip_tensor 将数据和 scale factors 打包在一起
            cute::gemm(
                tiled_mma_qk,
                make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                tSrS
            );

            // 预取下一个 K block (除了最后一个)
            if (k_block < size<2>(tSrQ) - 1) {
                copy_k_block(
                    smem_tiled_copy_K, smem_tiled_copy_SFK,
                    tSsK, tSsSFK,
                    tSrK_copy_view, tSrSFK_copy_view,
                    smem_pipe_read_k,
                    k_block + 1
                );
            }
        }
    }

    /**
     * 应用 Causal Masking
     *
     * 此函数从 mainloop_tma_ws.h:735-749 提取
     *
     * 对于 causal attention, 将未来的位置设为 -INFINITY
     * 对于 non-causal attention, 只 mask padding 部分
     *
     * @param tSrS - Attention scores (将被就地修改)
     * @param tScS - 坐标 tensor (用于判断位置)
     * @param n_block - 当前处理的 N block 索引
     * @param seqlen_k - K 的序列长度
     * @param unpadded_seqlen_k - K 的未 padding 序列长度
     * @param seqlen_q - Q 的序列长度
     * @param m_block - 当前处理的 M block 索引
     */
    template<typename TensorRS, typename TensorCS>
    __device__ __forceinline__ static void apply_masking(
        TensorRS& tSrS,
        TensorCS const& tScS,
        int n_block,
        int seqlen_k,
        int unpadded_seqlen_k,
        int seqlen_q,
        int m_block
    ) {
        // Lambda: 计算 causal masking 的列限制
        auto col_limit_causal = [&](int row, int n_block_idx) {
            return row + 1 + seqlen_k - n_block_idx * kBlockN - seqlen_q + m_block * kBlockM;
        };

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
            if constexpr (!IsCausal) {
                // Non-causal: 只 mask padding 部分
                if (int(get<1>(tScS(i))) >= int(unpadded_seqlen_k - n_block * kBlockN)) {
                    tSrS(i) = -INFINITY;
                }
            } else {
                // Causal: mask 未来位置和 padding
                int col_limit = std::min(
                    seqlen_k - n_block * kBlockN,
                    col_limit_causal(int(get<0>(tScS(i))), n_block)
                );
                if (int(get<1>(tScS(i))) >= col_limit) {
                    tSrS(i) = -INFINITY;
                }
            }
        }
    }

    /**
     * 完整的 QK GEMM 流程
     *
     * 包括:
     * 1. 等待 K 数据就绪
     * 2. 拷贝第一个 K block
     * 3. 添加 Delta_s correction
     * 4. 执行 GEMM
     * 5. 应用 masking
     * 6. 释放 K pipeline
     *
     * @param tiled_mma_qk - QK GEMM 的 tiled MMA 对象
     * @param tSrQ - Q 的 register tensor
     * @param tSrSFQ - SFQ 的 register tensor
     * @param tSrK - K 的 register tensor
     * @param tSrSFK - SFK 的 register tensor
     * @param tSrS - S 的 accumulator (输出)
     * @param smem_tiled_copy_K - K 的 tiled copy
     * @param smem_tiled_copy_SFK - SFK 的 tiled copy
     * @param tSsK - K 的 shared memory tensor
     * @param tSsSFK - SFK 的 shared memory tensor
     * @param tSrK_copy_view - K 的 register copy view
     * @param tSrSFK_copy_view - SFK 的 register copy view
     * @param pipeline_k - K 的 pipeline 对象
     * @param smem_pipe_read_k - Pipeline 读取状态
     * @param n_block - 当前 N block 索引
     * @param seqlen_k - K 序列长度
     * @param unpadded_seqlen_k - 未 padding K 序列长度
     * @param seqlen_q - Q 序列长度
     * @param m_block - 当前 M block 索引
     * @param add_delta_s_func - Delta_s correction 函数
     */
    template<
        typename TiledMma,
        typename TensorRQ,
        typename TensorRSFQ,
        typename TensorRK,
        typename TensorRSFK,
        typename TensorRS,
        typename SmemTiledCopyK,
        typename SmemTiledCopySFK,
        typename TensorSsK,
        typename TensorSsSFK,
        typename TensorRKView,
        typename TensorRSFKView,
        typename PipelineK,
        typename PipelineStateK,
        typename DeltaSFunc
    >
    __device__ __forceinline__ static void run(
        TiledMma const& tiled_mma_qk,
        TensorRQ const& tSrQ,
        TensorRSFQ const& tSrSFQ,
        TensorRK const& tSrK,
        TensorRSFK const& tSrSFK,
        TensorRS& tSrS,
        SmemTiledCopyK const& smem_tiled_copy_K,
        SmemTiledCopySFK const& smem_tiled_copy_SFK,
        TensorSsK const& tSsK,
        TensorSsSFK const& tSsSFK,
        TensorRKView& tSrK_copy_view,
        TensorRSFKView& tSrSFK_copy_view,
        PipelineK& pipeline_k,
        PipelineStateK& smem_pipe_read_k,
        int n_block,
        int seqlen_k,
        int unpadded_seqlen_k,
        int seqlen_q,
        int m_block,
        DeltaSFunc const& add_delta_s_func
    ) {
        // 1. 等待 K 数据就绪
        auto barrier_token = pipeline_k.consumer_try_wait(smem_pipe_read_k);
        pipeline_k.consumer_wait(smem_pipe_read_k, barrier_token);

        // 2. 拷贝第一个 K block
        copy_k_block(
            smem_tiled_copy_K, smem_tiled_copy_SFK,
            tSsK, tSsSFK,
            tSrK_copy_view, tSrSFK_copy_view,
            smem_pipe_read_k,
            _0{}
        );

        // 3. 添加 Delta_s correction (如果需要)
        add_delta_s_func(tSrS);

        // 4. 执行 GEMM
        compute_qk_gemm(
            tiled_mma_qk,
            tSrQ, tSrSFQ,
            tSrK, tSrSFK,
            tSrS,
            smem_tiled_copy_K, smem_tiled_copy_SFK,
            tSsK, tSsSFK,
            tSrK_copy_view, tSrSFK_copy_view,
            smem_pipe_read_k
        );

        // 5. 应用 masking
        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
        Tensor tScS = tiled_mma_qk.get_thread_slice(threadIdx.x).partition_C(cS);
        apply_masking(tSrS, tScS, n_block, seqlen_k, unpadded_seqlen_k, seqlen_q, m_block);

        // 6. 释放 K pipeline
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;
    }
};

}  // namespace sage
