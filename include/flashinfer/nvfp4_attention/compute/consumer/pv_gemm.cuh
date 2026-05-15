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
 * PV GEMM Computer
 *
 * 职责:
 * 1. 计算 P @ V GEMM (Attention Output)
 * 2. 从 shared memory 拷贝 V 到 register
 * 3. 量化 P 矩阵 (从 FP32 到 FP4 E2M1)
 * 4. 执行 blocked GEMM with FP4 inputs
 *
 * 特点:
 * - P 矩阵是量化后的 softmax 结果 (FP4)
 * - V 矩阵是转置的 (Vt)
 * - 使用 OMMA 指令 (FP4 Block GEMM)
 * - FP4 输入 + FP8 scale factors, FP32 累加器
 *
 * 模板参数:
 * @tparam Traits - Kernel 配置 traits
 */
template<typename Traits>
struct PVGemmComputer {

    using Element = typename Traits::Element;
    using ElementSF = typename Traits::ElementSF;
    using TileShape_MNK = typename Traits::TileShape_MNK;
    using TiledMmaPV = typename Traits::TiledMmaPV;
    using SmemCopyAtomKV = typename Traits::SmemCopyAtomKV;
    using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;
    using LayoutP = typename Traits::LayoutP;
    using LayoutSFP = typename Traits::LayoutSFP;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kBlockK = get<2>(TileShape_MNK{});

    /**
     * 从 shared memory 拷贝 V block 到 register
     *
     * 此函数从 mainloop_tma_ws.h:679-684 提取
     *
     * @param smem_tiled_copy_V - V 的 tiled copy 对象
     * @param smem_tiled_copy_SFV - SFV 的 tiled copy 对象
     * @param tOsVt - V^T 的 shared memory tensor (source)
     * @param tOsSFVt - SFV^T 的 shared memory tensor (source)
     * @param tOrVt_copy_view - V^T 的 register tensor (destination)
     * @param tOrSFVt_copy_view - SFV^T 的 register tensor (destination)
     * @param smem_pipe_read_v - Pipeline 读取状态
     * @param block_id - 当前处理的 V block ID
     */
    template<
        typename SmemTiledCopyV,
        typename SmemTiledCopySFV,
        typename TensorSsVt,
        typename TensorSsSFVt,
        typename TensorSrVt,
        typename TensorSrSFVt,
        typename PipelineStateV
    >
    __device__ __forceinline__ static void copy_v_block(
        SmemTiledCopyV const& smem_tiled_copy_V,
        SmemTiledCopySFV const& smem_tiled_copy_SFV,
        TensorSsVt const& tOsVt,
        TensorSsSFVt const& tOsSFVt,
        TensorSrVt& tOrVt_copy_view,
        TensorSrSFVt& tOrSFVt_copy_view,
        PipelineStateV const& smem_pipe_read_v,
        auto block_id
    ) {
        // 获取当前 pipeline stage 的数据
        auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
        auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());

        // 从 shared memory 拷贝到 register
        copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
        copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
    }

    /**
     * 量化 P 矩阵 (FP32 -> FP4 E2M1)
     *
     * 此函数从 mainloop_tma_ws.h:750-797 提取
     *
     * 将 softmax 结果 (FP32) 量化为 FP4 (E2M1 格式)
     * 同时计算 scale factors (FP8 E4M3)
     *
     * 算法:
     * 1. 将 scale factors (float) 转换为 FP8 E4M3
     * 2. 将 P 矩阵 (float) 转换为 FP4 E2M1
     * 3. 使用 warp shuffle 交换 scale factors
     *
     * @param acc_conversion_view - P 矩阵的转换视图 (输入: FP32)
     * @param AbsMaxP - Scale factors (输入: float)
     * @param tOrP - 量化后的 P (输出: FP4)
     * @param tOrSFP - 量化后的 scale factors (输出: FP8)
     * @param mma_k - 当前的 MMA K block 索引
     */
    template<typename TensorAcc, typename TensorMaxP, typename TensorRP, typename TensorRSFP>
    __device__ __forceinline__ static void quantize_p(
        TensorAcc const& acc_conversion_view,
        TensorMaxP const& AbsMaxP,
        TensorRP& tOrP,
        TensorRSFP& tOrSFP,
        int mma_k
    ) {
        // 获取当前 MMA stage 的数据
        Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
        Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);

        // 临时 tensor 用于存储 FP8 scale factors
        Tensor SFP = make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
        Tensor SFP_uint32_view = recast<uint32_t>(SFP);

        // 1. 将 scale factors 转换为 FP8 E4M3 (每 4 个 float 打包成 1 个 uint32)
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
            uint32_t& tmp = SFP_uint32_view(i / 4);
            packed_float_to_ue4m3(
                AbsMaxP_stagek(i),
                AbsMaxP_stagek(i + 1),
                AbsMaxP_stagek(i + 2),
                AbsMaxP_stagek(i + 3),
                tmp
            );
        }

        // 2. 将 P 矩阵转换为 FP4 E2M1
        int const quad_id = threadIdx.x & 3;
        uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);
        Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
        Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));

        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
            // 将 8 个 float 打包成 1 个 uint32 (FP4 E2M1)
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                packed_float_to_e2m1(
                    acc_conversion_stagek(make_coord(_0{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_1{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_2{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_3{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_4{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_5{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_6{}, i), mma_m),
                    acc_conversion_stagek(make_coord(_7{}, i), mma_m),
                    tOrP_uint32_view(i, mma_m)
                );
            }

            // 3. Warp shuffle: 交换 scale factors
            uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
            uint32_t peer_sfp  = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
            if ((quad_id & 1) == 0) {
                uint32_t sfp = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
                tOrSFP_uint32_view(_0{}, mma_m) = sfp;
            } else {
                uint32_t sfp = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
                tOrSFP_uint32_view(_0{}, mma_m) = sfp;
            }
        }
    }

    /**
     * 计算 P @ V GEMM
     *
     * 此函数从 mainloop_tma_ws.h:805-815 提取
     *
     * 执行 blocked GEMM:
     * - 输入: P (FP4), V^T (FP4), SFP (FP8), SFV^T (FP8)
     * - 输出: O (FP32 accumulator)
     * - 算法: O += (P * SFP) @ (V^T * SFV^T)
     *
     * @param tiled_mma_pv - PV GEMM 的 tiled MMA 对象
     * @param tOrP - P 的 register tensor (FP4)
     * @param tOrSFP - SFP 的 register tensor (FP8)
     * @param tOrVt - V^T 的 register tensor (FP4)
     * @param tOrSFVt - SFV^T 的 register tensor (FP8)
     * @param tOrO - O 的 accumulator (输出)
     * @param acc_conversion_view - P 的转换视图 (用于量化)
     * @param AbsMaxP - Scale factors
     * @param smem_tiled_copy_V - V 的 tiled copy (用于预取)
     * @param smem_tiled_copy_SFV - SFV 的 tiled copy (用于预取)
     * @param tOsVt - V^T 的 shared memory tensor
     * @param tOsSFVt - SFV^T 的 shared memory tensor
     * @param tOrVt_copy_view - V^T 的 register copy view
     * @param tOrSFVt_copy_view - SFV^T 的 register copy view
     * @param smem_pipe_read_v - Pipeline 读取状态
     */
    template<
        typename TiledMma,
        typename TensorRP,
        typename TensorRSFP,
        typename TensorRVt,
        typename TensorRSFVt,
        typename TensorRO,
        typename TensorAccConv,
        typename TensorMaxP,
        typename SmemTiledCopyV,
        typename SmemTiledCopySFV,
        typename TensorSsVt,
        typename TensorSsSFVt,
        typename TensorRVtView,
        typename TensorRSFVtView,
        typename PipelineStateV
    >
    __device__ __forceinline__ static void compute_pv_gemm(
        TiledMma const& tiled_mma_pv,
        TensorRP& tOrP,
        TensorRSFP& tOrSFP,
        TensorRVt const& tOrVt,
        TensorRSFVt const& tOrSFVt,
        TensorRO& tOrO,
        TensorAccConv const& acc_conversion_view,
        TensorMaxP const& AbsMaxP,
        SmemTiledCopyV const& smem_tiled_copy_V,
        SmemTiledCopySFV const& smem_tiled_copy_SFV,
        TensorSsVt const& tOsVt,
        TensorSsSFVt const& tOsSFVt,
        TensorRVtView& tOrVt_copy_view,
        TensorRSFVtView& tOrSFVt_copy_view,
        PipelineStateV const& smem_pipe_read_v
    ) {
        // Blocked GEMM: 循环处理所有 V blocks
        CUTLASS_PRAGMA_UNROLL
        for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
            // 执行 GEMM: O += P @ V^T
            // 使用 zip_tensor 将数据和 scale factors 打包在一起
            cute::gemm(
                tiled_mma_pv,
                make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                tOrO
            );

            // 预取下一个 V block (除了最后一个)
            if (v_block < size<2>(tOrP) - 1) {
                copy_v_block(
                    smem_tiled_copy_V, smem_tiled_copy_SFV,
                    tOsVt, tOsSFVt,
                    tOrVt_copy_view, tOrSFVt_copy_view,
                    smem_pipe_read_v,
                    v_block + 1
                );
                // 量化下一个 P block
                quantize_p(acc_conversion_view, AbsMaxP, tOrP, tOrSFP, v_block + 1);
            }
        }
    }

    /**
     * 完整的 PV GEMM 流程
     *
     * 包括:
     * 1. 等待 V 数据就绪
     * 2. 拷贝第一个 V block
     * 3. 量化第一个 P block
     * 4. 执行 GEMM
     * 5. 释放 V pipeline
     *
     * @param tiled_mma_pv - PV GEMM 的 tiled MMA 对象
     * @param tOrP - P 的 register tensor
     * @param tOrSFP - SFP 的 register tensor
     * @param tOrVt - V^T 的 register tensor
     * @param tOrSFVt - SFV^T 的 register tensor
     * @param tOrO - O 的 accumulator (输出)
     * @param acc_conversion_view - P 的转换视图
     * @param AbsMaxP - Scale factors
     * @param smem_tiled_copy_V - V 的 tiled copy
     * @param smem_tiled_copy_SFV - SFV 的 tiled copy
     * @param tOsVt - V^T 的 shared memory tensor
     * @param tOsSFVt - SFV^T 的 shared memory tensor
     * @param tOrVt_copy_view - V^T 的 register copy view
     * @param tOrSFVt_copy_view - SFV^T 的 register copy view
     * @param pipeline_v - V 的 pipeline 对象
     * @param smem_pipe_read_v - Pipeline 读取状态
     */
    template<
        typename TiledMma,
        typename TensorRP,
        typename TensorRSFP,
        typename TensorRVt,
        typename TensorRSFVt,
        typename TensorRO,
        typename TensorAccConv,
        typename TensorMaxP,
        typename SmemTiledCopyV,
        typename SmemTiledCopySFV,
        typename TensorSsVt,
        typename TensorSsSFVt,
        typename TensorRVtView,
        typename TensorRSFVtView,
        typename PipelineV,
        typename PipelineStateV
    >
    __device__ __forceinline__ static void run(
        TiledMma const& tiled_mma_pv,
        TensorRP& tOrP,
        TensorRSFP& tOrSFP,
        TensorRVt const& tOrVt,
        TensorRSFVt const& tOrSFVt,
        TensorRO& tOrO,
        TensorAccConv const& acc_conversion_view,
        TensorMaxP const& AbsMaxP,
        SmemTiledCopyV const& smem_tiled_copy_V,
        SmemTiledCopySFV const& smem_tiled_copy_SFV,
        TensorSsVt const& tOsVt,
        TensorSsSFVt const& tOsSFVt,
        TensorRVtView& tOrVt_copy_view,
        TensorRSFVtView& tOrSFVt_copy_view,
        PipelineV& pipeline_v,
        PipelineStateV& smem_pipe_read_v
    ) {
        // 1. 等待 V 数据就绪
        auto barrier_token = pipeline_v.consumer_try_wait(smem_pipe_read_v);
        pipeline_v.consumer_wait(smem_pipe_read_v, barrier_token);

        // 2. 拷贝第一个 V block
        copy_v_block(
            smem_tiled_copy_V, smem_tiled_copy_SFV,
            tOsVt, tOsSFVt,
            tOrVt_copy_view, tOrSFVt_copy_view,
            smem_pipe_read_v,
            _0{}
        );

        // 3. 量化第一个 P block
        quantize_p(acc_conversion_view, AbsMaxP, tOrP, tOrSFP, 0);

        // 4. 执行 GEMM
        compute_pv_gemm(
            tiled_mma_pv,
            tOrP, tOrSFP,
            tOrVt, tOrSFVt,
            tOrO,
            acc_conversion_view, AbsMaxP,
            smem_tiled_copy_V, smem_tiled_copy_SFV,
            tOsVt, tOsSFVt,
            tOrVt_copy_view, tOrSFVt_copy_view,
            smem_pipe_read_v
        );

        // 5. 释放 V pipeline
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
    }

private:
    /**
     * 辅助函数: 量化相关
     *
     * 注意: 这些函数应该在 quantization/ 模块中定义
     * 这里假设它们存在
     */
    __device__ __forceinline__ static void packed_float_to_ue4m3(
        float a, float b, float c, float d, uint32_t& out
    ) {
        // TODO: 实现 float -> FP8 E4M3 的转换
        // 原始代码位置: sageattn3/blackwell/utils.h
    }

    __device__ __forceinline__ static void packed_float_to_e2m1(
        float v0, float v1, float v2, float v3,
        float v4, float v5, float v6, float v7,
        uint32_t& out
    ) {
        // TODO: 实现 float -> FP4 E2M1 的转换
        // 原始代码位置: sageattn3/blackwell/utils.h
    }
};

}  // namespace sage
