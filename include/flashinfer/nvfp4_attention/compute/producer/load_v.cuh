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
 * V Tensor Loader
 *
 * 职责：
 * 1. 从 global memory 加载 V^T (转置的 V) 到 shared memory (使用 TMA)
 * 2. 同时加载 V 的 scale factors (SFV)
 * 3. 使用 pipeline_v 进行同步
 *
 * 特点：
 * - V 在循环中重复加载（每个 n_block）
 * - 有 multicast（cluster 内的多个 block 共享）
 * - 加载的是 Vt（V transposed）
 * - 形状: (kHeadDim, kBlockN) - 注意是转置的！
 *
 * 模板参数：
 * @tparam Traits - Kernel 配置 traits
 */
template<typename Traits>
struct VLoader {

    // 类型定义
    using Element = typename Traits::Element;
    using ElementSF = typename Traits::ElementSF;
    using TileShape_MNK = typename Traits::TileShape_MNK;
    using SmemLayoutV = typename Traits::SmemLayoutV;        // 实际是 Vt 的 layout
    using SmemLayoutSFV = typename Traits::SmemLayoutSFV;    // 实际是 SFVt 的 layout

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    /**
     * 加载 V^T tile 到 shared memory
     *
     * 此函数从 mainloop_tma_ws.h 中的以下代码提取：
     *
     * 原始代码位置：mainloop_tma_ws.h:524-529 (第一次加载)
     * 以及 mainloop_tma_ws.h:545-550 (循环中的加载)
     *
     * ```cpp
     * pipeline_v.producer_acquire(smem_pipe_write_v);
     * copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
     *      tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
     * copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
     *      tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
     * ++smem_pipe_write_v;
     * ```
     *
     * @param mainloop_params - 包含 TMA 描述符的参数
     * @param pipeline_v - V 的 pipeline 对象
     * @param smem_pipe_write_v - Pipeline 写入状态
     * @param tVgVt - V^T 的 global memory tensor（已分区）
     * @param tVsVt - V^T 的 shared memory tensor（已分区）
     * @param tVgSFVt - SFV^T 的 global memory tensor（已分区）
     * @param tVsSFVt - SFV^T 的 shared memory tensor（已分区）
     * @param n_block - 当前的 N 维度 block 索引
     * @param mcast_mask_kv - Multicast mask（用于 cluster 内共享）
     * @param lane_predicate - 是否是负责 TMA 的线程
     */
    template<
        typename MainloopParams,
        typename PipelineV,
        typename PipelineStateV,
        typename TensorGVt,
        typename TensorSVt,
        typename TensorGSFVt,
        typename TensorSSFVt
    >
    __device__ __forceinline__ static void load_and_stage(
        const MainloopParams& mainloop_params,
        PipelineV& pipeline_v,
        PipelineStateV& smem_pipe_write_v,
        const TensorGVt& tVgVt,
        const TensorSVt& tVsVt,
        const TensorGSFVt& tVgSFVt,
        const TensorSSFVt& tVsSFVt,
        int n_block,
        uint16_t mcast_mask_kv,
        bool lane_predicate
    ) {
        // 只有选中的 lane 执行 TMA 操作
        if (lane_predicate) {
            // 1. 获取 pipeline barrier
            pipeline_v.producer_acquire(smem_pipe_write_v);

            // 2. 使用 TMA 加载 V^T 数据到 shared memory
            // 注意：加载的是转置后的 V，形状是 (kHeadDim, kBlockN)
            copy(
                mainloop_params.tma_load_Vt.with(
                    *pipeline_v.producer_get_barrier(smem_pipe_write_v),
                    mcast_mask_kv  // multicast for K/V
                ),
                tVgVt(_, n_block),                    // source: global memory (当前 n_block)
                tVsVt(_, smem_pipe_write_v.index())   // destination: shared memory (pipeline 索引)
            );

            // 3. 使用 TMA 加载 V 的 scale factors（也是转置的）
            copy(
                mainloop_params.tma_load_SFVt.with(
                    *pipeline_v.producer_get_barrier(smem_pipe_write_v),
                    mcast_mask_kv
                ),
                tVgSFVt(_, n_block),
                tVsSFVt(_, smem_pipe_write_v.index())
            );

            // 4. 提交 pipeline
            ++smem_pipe_write_v;
        }
    }

    /**
     * 准备 TMA 张量分区
     *
     * 从 mainloop_tma_ws.h:465, 469, 475, 485, 498-503 提取
     *
     * @param mainloop_params - 包含 TMA 描述符和形状信息
     * @param shared_storage - Shared memory storage
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     * @param cluster_local_block_id - Cluster 内的 block ID
     * @return tuple(tVgVt, tVsVt, tVgSFVt, tVsSFVt, ...)
     */
    template<typename MainloopParams, typename SharedStorage>
    __device__ __forceinline__ static auto prepare_tma_tensors(
        const MainloopParams& mainloop_params,
        SharedStorage& shared_storage,
        int bidh,
        int bidb,
        uint2 cluster_local_block_id
    ) {
        // 1. 创建 shared memory tensors
        // 注意：SmemLayoutV 实际上是 Vt 的 layout
        auto sVt = make_tensor(
            make_smem_ptr(shared_storage.smem_v.begin()),
            SmemLayoutV{}
        );
        auto sSFVt = make_tensor(
            make_smem_ptr(shared_storage.smem_SFV.begin()),
            SmemLayoutSFV{}
        );

        // 2. 获取 global memory tensors
        // 注意：已经是转置的形状
        auto mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);
        auto mSFVt = mainloop_params.tma_load_SFVt.get_tma_tensor(
            shape(mainloop_params.layout_SFVt)
        );

        // 3. 根据 batch 和 head 索引选择当前的 tile
        // Vt: (kHeadDim, kBlockN, _) - 转置后的形状
        auto gVt = local_tile(
            mVt(_, _, bidh, bidb),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            make_coord(_0{}, _)  // 第三维是 n_block，在加载时指定
        );
        auto gSFVt = local_tile(
            mSFVt(_, _, bidh, bidb),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            make_coord(_0{}, _)
        );

        // 4. 创建 TMA copy 对象并分区
        // 注意：V 使用 cluster_local_block_id.x 来支持 multicast
        auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
        auto tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));
        auto tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));

        auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
        auto tVgSFVt = group_modes<0, 3>(block_tma_sfvt.partition_S(gSFVt));
        auto tVsSFVt = group_modes<0, 3>(block_tma_sfvt.partition_D(sSFVt));

        return cute::make_tuple(
            tVgVt, tVsVt,
            tVgSFVt, tVsSFVt,
            block_tma_vt, block_tma_sfvt
        );
    }

    /**
     * 预取 TMA 描述符到 L2 cache
     *
     * 从 mainloop_tma_ws.h:272, 275 提取
     */
    template<typename MainloopParams>
    __device__ __forceinline__ static void prefetch_tma_descriptors(
        const MainloopParams& mainloop_params
    ) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_Vt.get_tma_descriptor()
        );
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_SFVt.get_tma_descriptor()
        );
    }
};

}  // namespace sage
