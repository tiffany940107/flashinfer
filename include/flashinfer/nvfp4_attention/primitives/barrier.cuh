/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace sage {

/**
 * Named Barriers for SageAttention3
 *
 * 用于协调不同 warp 之间的同步
 *
 * Hopper 架构支持 named barriers，允许不同的 warp group
 * 使用独立的 barrier 进行同步，避免全局同步
 */
enum class NamedBarriers {
    QueryEmpty = 1,                        // Q 数据为空
    WarpSpecializedConsumer = 2,           // Consumer warp 同步
    WarpSpecializedPingPongConsumer1 = 3,  // Ping-pong consumer 1
    WarpSpecializedPingPongConsumer2 = 4,  // Ping-pong consumer 2
    ProducerEnd = 5,                       // Producer 结束
    ConsumerEnd = 6,                       // Consumer 结束
    EpilogueBarrier = 7                    // Epilogue 同步
};

/**
 * Ordered Sequence Barrier (变长 Group Size)
 *
 * 用于管理有序的 barrier 序列，支持不同大小的 thread group
 *
 * 背景:
 * 在 FlashAttention 中，producer 和 consumer warps 需要按顺序处理数据。
 * 这个 barrier 确保：
 * 1. Producer 先完成数据加载
 * 2. Consumer 等待数据就绪后开始计算
 * 3. Producer 等待 consumer 释放 shared memory 后加载下一个 tile
 *
 * 特点:
 * - 支持可变的 group size (每个 stage 可以有不同数量的线程)
 * - 支持 pipeline depth (多个 stages 可以同时进行)
 * - 使用 ClusterBarrier (跨 CTA 同步)
 *
 * 模板参数:
 * @tparam SequenceDepth - Pipeline 深度 (stages 数量)
 * @tparam SequenceLength - Sequence 长度 (groups 数量)
 */
template<int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrier {
public:
    static constexpr int SequenceDepth = SequenceDepth_;
    static constexpr int SequenceLength = SequenceLength_;

    using Barrier = cutlass::arch::ClusterBarrier;
    using PipelineState = cutlass::PipelineState<SequenceDepth>;

    /**
     * Shared Storage for Barriers
     *
     * 存储 SequenceDepth x SequenceLength 个 barriers
     * - Depth 维度: 不同的 pipeline stages
     * - Length 维度: 不同的 thread groups
     */
    struct SharedStorage {
        Barrier barrier_[SequenceDepth][SequenceLength];
    };

    /**
     * Parameters
     *
     * @field group_id - 当前 thread group 的 ID (0 to SequenceLength-1)
     * @field group_size_list - 每个 group 的线程数量数组 (长度为 SequenceLength)
     */
    struct Params {
        uint32_t group_id;
        uint32_t* group_size_list;
    };

private:
    Params params_;
    Barrier* barrier_ptr_;
    PipelineState stage_;

public:
    // 禁用复制和移动
    OrderedSequenceBarrier() = delete;
    OrderedSequenceBarrier(const OrderedSequenceBarrier&) = delete;
    OrderedSequenceBarrier(OrderedSequenceBarrier&&) = delete;
    OrderedSequenceBarrier& operator=(const OrderedSequenceBarrier&) = delete;
    OrderedSequenceBarrier& operator=(OrderedSequenceBarrier&&) = delete;
    ~OrderedSequenceBarrier() = default;

    /**
     * 构造函数
     *
     * 初始化所有 barriers
     *
     * @param storage - Shared memory storage
     * @param params - Parameters (group_id, group_size_list)
     */
    CUTLASS_DEVICE
    OrderedSequenceBarrier(SharedStorage& storage, Params const& params) :
        params_(params),
        barrier_ptr_(&storage.barrier_[0][0]),
        // Group 0 starts with an opposite phase
        stage_({0, params.group_id == 0, 0}) {

        int warp_idx = cutlass::canonical_warp_idx_sync();
        int lane_predicate = cute::elect_one_sync();

        // Barrier initialization
        // 只有一个 elected thread 执行初始化
        if (warp_idx == 0 && lane_predicate) {
            for (int d = 0; d < SequenceDepth; ++d) {
                for (int l = 0; l < SequenceLength; ++l) {
                    barrier_ptr_[d * SequenceLength + l].init(
                        *(params.group_size_list + l)
                    );
                }
            }
        }

        // Fence: 确保 barrier 初始化对所有线程可见
        cutlass::arch::fence_barrier_init();
    }

    /**
     * Wait: 等待当前 stage 解锁
     *
     * 等待前一个 group 完成并 signal
     */
    CUTLASS_DEVICE
    void wait() {
        get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
    }

    /**
     * Arrive: Signal 完成当前 stage 并移动到下一个 stage
     *
     * 当前 group 完成后，signal 给下一个 group
     * (group_id) signals to (group_id + 1)
     */
    CUTLASS_DEVICE
    void arrive() {
        int signalling_id = (params_.group_id + 1) % SequenceLength;
        get_barrier_for_current_stage(signalling_id).arrive();
        ++stage_;
    }

    /**
     * Advance: 移动到下一个 stage (不 signal)
     *
     * 用于跳过某些 stages
     */
    CUTLASS_DEVICE
    void advance() {
        ++stage_;
    }

    /**
     * 获取当前 stage 对应的 barrier
     */
    CUTLASS_DEVICE
    Barrier& get_barrier_for_current_stage(int group_id) {
        return barrier_ptr_[stage_.index() * SequenceLength + group_id];
    }
};

/**
 * OrderedSequenceBarrierVarGroupSize - Shared Storage
 *
 * 用于可变组大小的有序屏障
 */
template<int SequenceDepth, int SequenceLength>
struct OrderedSequenceBarrierVarGroupSizeSharedStorage {
    using Barrier = cutlass::arch::ClusterBarrier;
    Barrier barrier_[SequenceDepth][SequenceLength];
};

/**
 * OrderedSequenceBarrierVarGroupSize - 可变组大小的有序屏障
 *
 * 用于 Epilogue 阶段的 Producer-Consumer 同步
 */
template<int SequenceDepth_, int SequenceLength_>
class OrderedSequenceBarrierVarGroupSize {
public:
    static constexpr int SequenceDepth = SequenceDepth_;
    static constexpr int SequenceLength = SequenceLength_;
    using Barrier = cutlass::arch::ClusterBarrier;
    using SharedStorage = OrderedSequenceBarrierVarGroupSizeSharedStorage<SequenceDepth, SequenceLength>;

    struct Params {
        uint32_t group_id;
        uint32_t* group_size_list;
    };

private:
    // In future this Params object can be replaced easily with a CG object
    Params params_;
    Barrier *barrier_ptr_;
    cutlass::PipelineState<SequenceDepth> stage_;

    static constexpr int Depth = SequenceDepth;
    static constexpr int Length = SequenceLength;

public:
    OrderedSequenceBarrierVarGroupSize() = delete;
    OrderedSequenceBarrierVarGroupSize(const OrderedSequenceBarrierVarGroupSize&) = delete;
    OrderedSequenceBarrierVarGroupSize(OrderedSequenceBarrierVarGroupSize&&) = delete;
    OrderedSequenceBarrierVarGroupSize& operator=(const OrderedSequenceBarrierVarGroupSize&) = delete;
    OrderedSequenceBarrierVarGroupSize& operator=(OrderedSequenceBarrierVarGroupSize&&) = delete;
    ~OrderedSequenceBarrierVarGroupSize() = default;

    CUTLASS_DEVICE
    OrderedSequenceBarrierVarGroupSize(SharedStorage& storage, Params const& params) :
        params_(params),
        barrier_ptr_(&storage.barrier_[0][0]),
        // Group 0 - starts with an opposite phase
        stage_({0, params.group_id == 0, 0}) {
        int warp_idx = cutlass::canonical_warp_idx_sync();
        int lane_predicate = cute::elect_one_sync();

        // Barrier FULL, EMPTY init
        // Init is done only by the one elected thread of the block
        if (warp_idx == 0 && lane_predicate) {
            for (int d = 0; d < Depth; ++d) {
                for (int l = 0; l < Length; ++l) {
                    barrier_ptr_[d * Length + l].init(*(params.group_size_list + l));
                }
            }
        }
        cutlass::arch::fence_barrier_init();
    }

    // Wait on a stage to be unlocked
    CUTLASS_DEVICE
    void wait() {
        get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
    }

    // Signal completion of Stage and move to the next stage
    // (group_id) signals to (group_id+1)
    CUTLASS_DEVICE
    void arrive() {
        int signalling_id = (params_.group_id + 1) % Length;
        get_barrier_for_current_stage(signalling_id).arrive();
        ++stage_;
    }

    CUTLASS_DEVICE
    void advance() {
        ++stage_;
    }

private:
    CUTLASS_DEVICE
    Barrier& get_barrier_for_current_stage(int group_id) {
        return barrier_ptr_[stage_.index() * Length + group_id];
    }
};

}  // namespace sage
