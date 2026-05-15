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
 * K Tensor Loader
 *
 * 职责：
 * 1. 从 global memory 加载 K 到 shared memory (使用 TMA)
 * 2. 同时加载 K 的 scale factors (SFK)
 * 3. 同时加载 Delta_s 校正值
 * 4. 使用 pipeline_k 进行同步
 *
 * 特点：
 * - K 在循环中重复加载（每个 n_block）
 * - 有 multicast（cluster 内的多个 block 共享）
 * - 需要 permute 操作
 * - 形状: (kBlockN, kHeadDim)
 * - 与 Delta_s 一起加载
 *
 * 模板参数：
 * @tparam Traits - Kernel 配置 traits
 */
template<typename Traits>
struct KLoader {

    // 类型定义
    using Element = typename Traits::Element;
    using ElementSF = typename Traits::ElementSF;
    using TileShape_MNK = typename Traits::TileShape_MNK;
    using SmemLayoutK = typename Traits::SmemLayoutK;
    using SmemLayoutSFK = typename Traits::SmemLayoutSFK;
    using SmemLayoutDS = typename Traits::SmemLayoutDS;

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr bool BlockMean = Traits::BlockMean;

    /**
     * 加载 K tile（和 Delta_s）到 shared memory
     *
     * 此函数从 mainloop_tma_ws.h 中的以下代码提取：
     *
     * 原始代码位置：mainloop_tma_ws.h:516-523 (第一次加载)
     * 以及 mainloop_tma_ws.h:537-544 (循环中的加载)
     *
     * ```cpp
     * pipeline_k.producer_acquire(smem_pipe_write_k);
     * copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
     *      tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
     * copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
     *      tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
     * copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
     *      tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
     * ++smem_pipe_write_k;
     * ```
     *
     * @param mainloop_params - 包含 TMA 描述符的参数
     * @param pipeline_k - K 的 pipeline 对象
     * @param smem_pipe_write_k - Pipeline 写入状态
     * @param tKgK - K 的 global memory tensor（已分区）
     * @param tKsK - K 的 shared memory tensor（已分区）
     * @param tKgSFK - SFK 的 global memory tensor（已分区）
     * @param tKsSFK - SFK 的 shared memory tensor（已分区）
     * @param tDSgDS - Delta_s 的 global memory tensor（已分区）
     * @param tDSsDS - Delta_s 的 shared memory tensor（已分区）
     * @param n_block - 当前的 N 维度 block 索引
     * @param mcast_mask_kv - Multicast mask（用于 cluster 内共享）
     * @param lane_predicate - 是否是负责 TMA 的线程
     */
    template<
        typename MainloopParams,
        typename PipelineK,
        typename PipelineStateK,
        typename TensorGK,
        typename TensorSK,
        typename TensorGSFK,
        typename TensorSSFK,
        typename TensorGDS,
        typename TensorSDS
    >
    __device__ __forceinline__ static void load_and_stage(
        const MainloopParams& mainloop_params,
        PipelineK& pipeline_k,
        PipelineStateK& smem_pipe_write_k,
        const TensorGK& tKgK,
        const TensorSK& tKsK,
        const TensorGSFK& tKgSFK,
        const TensorSSFK& tKsSFK,
        const TensorGDS& tDSgDS,
        const TensorSDS& tDSsDS,
        int n_block,
        uint16_t mcast_mask_kv,
        bool lane_predicate
    ) {
        // 只有选中的 lane 执行 TMA 操作
        if (lane_predicate) {
            // 1. 获取 pipeline barrier
            pipeline_k.producer_acquire(smem_pipe_write_k);

            // 2. 使用 TMA 加载 K 数据到 shared memory
            // 注意：使用 mcast_mask_kv 实现 cluster 内的 multicast
            copy(
                mainloop_params.tma_load_K.with(
                    *pipeline_k.producer_get_barrier(smem_pipe_write_k),
                    mcast_mask_kv  // multicast for K/V
                ),
                tKgK(_, n_block),                    // source: global memory (当前 n_block)
                tKsK(_, smem_pipe_write_k.index())   // destination: shared memory (pipeline 索引)
            );

            // 3. 使用 TMA 加载 K 的 scale factors
            copy(
                mainloop_params.tma_load_SFK.with(
                    *pipeline_k.producer_get_barrier(smem_pipe_write_k),
                    mcast_mask_kv
                ),
                tKgSFK(_, n_block),
                tKsSFK(_, smem_pipe_write_k.index())
            );

            // 4. 使用 TMA 加载 Delta_s 校正值
            copy(
                mainloop_params.tma_load_DS.with(
                    *pipeline_k.producer_get_barrier(smem_pipe_write_k),
                    mcast_mask_kv
                ),
                tDSgDS(_, n_block),
                tDSsDS(_, smem_pipe_write_k.index())
            );

            // 5. 提交 pipeline
            ++smem_pipe_write_k;
        }
    }

    /**
     * 准备 TMA 张量分区
     *
     * 从 mainloop_tma_ws.h:464, 468, 474, 476-482, 484, 492-506 提取
     *
     * @param mainloop_params - 包含 TMA 描述符和形状信息
     * @param shared_storage - Shared memory storage
     * @param m_block - 当前处理的 M 维度 block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     * @param cluster_local_block_id - Cluster 内的 block ID
     * @return tuple(tKgK, tKsK, tKgSFK, tKsSFK, tDSgDS, tDSsDS, ...)
     */
    template<typename MainloopParams, typename SharedStorage>
    __device__ __forceinline__ static auto prepare_tma_tensors(
        const MainloopParams& mainloop_params,
        SharedStorage& shared_storage,
        int m_block,
        int bidh,
        int bidb,
        uint2 cluster_local_block_id
    ) {
        // 1. 创建 shared memory tensors
        auto sK = make_tensor(
            make_smem_ptr(shared_storage.smem_k.begin()),
            SmemLayoutK{}
        );
        auto sSFK = make_tensor(
            make_smem_ptr(shared_storage.smem_SFK.begin()),
            SmemLayoutSFK{}
        );
        auto sDS = make_tensor(
            make_smem_ptr(shared_storage.smem_ds.begin()),
            SmemLayoutDS{}
        );

        // 2. 获取 global memory tensors
        auto mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
        auto mSFK = mainloop_params.tma_load_SFK.get_tma_tensor(
            shape(mainloop_params.layout_SFK)
        );
        auto mDS = mainloop_params.tma_load_DS.get_tma_tensor(
            shape(mainloop_params.layout_DS)
        );

        // 3. 根据 batch 和 head 索引选择当前的 tile
        // K: (N, K, _) - 第三维是 n_block，在加载时指定
        auto gK = local_tile(
            mK(_, _, bidh, bidb),
            select<1, 2>(TileShape_MNK{}),
            make_coord(_, _0{})
        );
        auto gSFK = local_tile(
            mSFK(_, _, bidh, bidb),
            select<1, 2>(TileShape_MNK{}),
            make_coord(_, _0{})
        );

        // Delta_s: 根据 per_block_mean 选择不同的索引方式
        auto gDS = [&] {
            if constexpr (BlockMean) {
                // Per-block mean: 每个 m_block 有自己的 delta_s
                return local_tile(
                    mDS(_, _, bidh, bidb),
                    select<0, 1>(TileShape_MNK{}),
                    make_coord(m_block, _)
                );
            } else {
                // Global mean: 所有 block 共享同一个 delta_s
                return local_tile(
                    mDS(_, _, bidh, bidb),
                    select<0, 1>(TileShape_MNK{}),
                    make_coord(_0{}, _)
                );
            }
        }();

        // 4. 创建 TMA copy 对象并分区
        // 注意：K/V 使用 cluster_local_block_id.x 来支持 multicast
        auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
        auto tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        auto tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));

        auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
        auto tKgSFK = group_modes<0, 3>(block_tma_sfk.partition_S(gSFK));
        auto tKsSFK = group_modes<0, 3>(block_tma_sfk.partition_D(sSFK));

        auto block_tma_ds = mainloop_params.tma_load_DS.get_slice(cluster_local_block_id.x);
        auto tDSgDS = group_modes<0, 3>(block_tma_ds.partition_S(gDS));
        auto tDSsDS = group_modes<0, 3>(block_tma_ds.partition_D(sDS));

        return cute::make_tuple(
            tKgK, tKsK,
            tKgSFK, tKsSFK,
            tDSgDS, tDSsDS,
            block_tma_k, block_tma_sfk, block_tma_ds
        );
    }

    /**
     * 预取 TMA 描述符到 L2 cache
     *
     * 从 mainloop_tma_ws.h:271, 274, 276 提取
     */
    template<typename MainloopParams>
    __device__ __forceinline__ static void prefetch_tma_descriptors(
        const MainloopParams& mainloop_params
    ) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_K.get_tma_descriptor()
        );
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_SFK.get_tma_descriptor()
        );
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_DS.get_tma_descriptor()
        );
    }
};

}  // namespace sage
