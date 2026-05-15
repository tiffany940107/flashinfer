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
 * Q Tensor Loader
 *
 * 职责：
 * 1. 从 global memory 加载 Q 到 shared memory (使用 TMA)
 * 2. 同时加载 Q 的 scale factors (SFQ)
 * 3. 使用独立的 pipeline_q 进行同步
 *
 * 特点：
 * - Q 只加载一次（不在循环中）
 * - 没有 multicast（每个 block 独立加载）
 * - 形状: (kBlockM, kHeadDim)
 *
 * 模板参数：
 * @tparam Traits - Kernel 配置 traits，包含：
 *   - Element: 数据类型 (FP4)
 *   - ElementSF: Scale factor 类型 (FP8 e4m3)
 *   - TileShape_MNK: Tile 形状
 *   - SmemLayoutQ: Q 的 shared memory layout
 *   - SmemLayoutSFQ: SFQ 的 shared memory layout
 */
template<typename Traits>
struct QLoader {

    // 类型定义
    using Element = typename Traits::Element;
    using ElementSF = typename Traits::ElementSF;
    using TileShape_MNK = typename Traits::TileShape_MNK;
    using SmemLayoutQ = typename Traits::SmemLayoutQ;
    using SmemLayoutSFQ = typename Traits::SmemLayoutSFQ;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    /**
     * 加载 Q tile 到 shared memory
     *
     * 此函数从 mainloop_tma_ws.h 中的以下代码提取：
     *
     * 原始代码位置：mainloop_tma_ws.h:511-515
     * ```cpp
     * if (lane_predicate) {
     *     pipeline_q.producer_acquire(smem_pipe_write_q);
     *     copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
     *          tQgQ, tQsQ);
     *     copy(mainloop_params.tma_load_SFQ.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
     *          tQgSFQ, tQsSFQ);
     *     ++smem_pipe_write_q;
     * }
     * ```
     *
     * @param mainloop_params - 包含 TMA 描述符的参数
     * @param pipeline_q - Q 的 pipeline 对象
     * @param smem_pipe_write_q - Pipeline 写入状态
     * @param tQgQ - Q 的 global memory tensor（已分区）
     * @param tQsQ - Q 的 shared memory tensor（已分区）
     * @param tQgSFQ - SFQ 的 global memory tensor（已分区）
     * @param tQsSFQ - SFQ 的 shared memory tensor（已分区）
     * @param lane_predicate - 是否是负责 TMA 的线程
     */
    template<
        typename MainloopParams,
        typename PipelineQ,
        typename PipelineStateQ,
        typename TensorGQ,
        typename TensorSQ,
        typename TensorGSFQ,
        typename TensorSSFQ
    >
    __device__ __forceinline__ static void load_and_stage(
        const MainloopParams& mainloop_params,
        PipelineQ& pipeline_q,
        PipelineStateQ& smem_pipe_write_q,
        const TensorGQ& tQgQ,
        const TensorSQ& tQsQ,
        const TensorGSFQ& tQgSFQ,
        const TensorSSFQ& tQsSFQ,
        bool lane_predicate
    ) {
        // 只有选中的 lane 执行 TMA 操作
        if (lane_predicate) {
            // 1. 获取 pipeline barrier（等待 shared memory 可用）
            pipeline_q.producer_acquire(smem_pipe_write_q);

            // 2. 使用 TMA 加载 Q 数据到 shared memory
            copy(
                mainloop_params.tma_load_Q.with(
                    *pipeline_q.producer_get_barrier(smem_pipe_write_q),
                    0  // no multicast for Q
                ),
                tQgQ,   // source: global memory
                tQsQ    // destination: shared memory
            );

            // 3. 使用 TMA 加载 Q 的 scale factors 到 shared memory
            copy(
                mainloop_params.tma_load_SFQ.with(
                    *pipeline_q.producer_get_barrier(smem_pipe_write_q),
                    0  // no multicast for Q
                ),
                tQgSFQ, // source: global memory
                tQsSFQ  // destination: shared memory
            );

            // 4. 提交 pipeline（通知 consumer 数据已准备好）
            ++smem_pipe_write_q;
        }
    }

    /**
     * 准备 TMA 张量分区
     *
     * 从 mainloop_tma_ws.h:463, 467, 486-491 提取
     *
     * @param mainloop_params - 包含 TMA 描述符和形状信息
     * @param shared_storage - Shared memory storage
     * @param m_block - 当前处理的 M 维度 block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     * @return tuple(tQgQ, tQsQ, tQgSFQ, tQsSFQ, block_tma_q, block_tma_sfq)
     */
    template<typename MainloopParams, typename SharedStorage>
    __device__ __forceinline__ static auto prepare_tma_tensors(
        const MainloopParams& mainloop_params,
        SharedStorage& shared_storage,
        int m_block,
        int bidh,
        int bidb
    ) {
        // 1. 创建 shared memory tensors
        auto sQ = make_tensor(
            make_smem_ptr(shared_storage.smem_q.begin()),
            SmemLayoutQ{}
        );
        auto sSFQ = make_tensor(
            make_smem_ptr(shared_storage.smem_SFQ.begin()),
            SmemLayoutSFQ{}
        );

        // 2. 获取 global memory tensors
        auto mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
        auto mSFQ = mainloop_params.tma_load_SFQ.get_tma_tensor(
            shape(mainloop_params.layout_SFQ)
        );

        // 3. 根据 batch 和 head 索引选择当前的 tile
        auto gQ = local_tile(
            mQ(_, _, bidh, bidb),
            select<0, 2>(TileShape_MNK{}),
            make_coord(m_block, _0{})
        );
        auto gSFQ = local_tile(
            mSFQ(_, _, bidh, bidb),
            select<0, 2>(TileShape_MNK{}),
            make_coord(m_block, _0{})
        );

        // 4. 创建 TMA copy 对象并分区
        auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
        auto tQgQ = block_tma_q.partition_S(gQ);   // source partition
        auto tQsQ = block_tma_q.partition_D(sQ);   // destination partition

        auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});
        auto tQgSFQ = block_tma_sfq.partition_S(gSFQ);
        auto tQsSFQ = block_tma_sfq.partition_D(sSFQ);

        return cute::make_tuple(
            tQgQ, tQsQ,
            tQgSFQ, tQsSFQ,
            block_tma_q, block_tma_sfq
        );
    }

    /**
     * 预取 TMA 描述符到 L2 cache
     *
     * 从 mainloop_tma_ws.h:270, 273 提取
     */
    template<typename MainloopParams>
    __device__ __forceinline__ static void prefetch_tma_descriptors(
        const MainloopParams& mainloop_params
    ) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_Q.get_tma_descriptor()
        );
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_SFQ.get_tma_descriptor()
        );
    }
};

}  // namespace sage
