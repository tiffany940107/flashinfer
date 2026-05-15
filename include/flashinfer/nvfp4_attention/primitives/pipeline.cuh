/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cute/tensor.hpp"

namespace sage {

using namespace cute;

/**
 * Pipeline 管理
 *
 * Pipeline 实现了软件流水线，允许多个 stages 同时进行：
 * - Producer 加载下一个 tile 的数据
 * - Consumer 计算当前 tile
 *
 * 这样可以隐藏内存延迟，提高吞吐量
 *
 * Pipeline 模型:
 * ```
 * Stage:    0      1      2      3
 * Producer: Load0  Load1  Load2  Load3
 * Consumer:  wait  Comp0  Comp1  Comp2
 * ```
 *
 * 关键概念:
 * - Pipeline Depth: 同时进行的 stages 数量 (通常 2-4)
 * - Pipeline State: 当前的 stage index 和 phase
 * - Barrier: 用于 producer 和 consumer 之间的同步
 */

/**
 * Pipeline State 包装器
 *
 * 管理 pipeline 的当前状态
 *
 * @tparam Depth - Pipeline 深度
 */
template<int Depth>
using PipelineState = cutlass::PipelineState<Depth>;

/**
 * Producer-Consumer Pipeline
 *
 * 标准的 producer-consumer pipeline
 * Producer 加载数据到 shared memory
 * Consumer 从 shared memory 读取并计算
 *
 * @tparam Stages - Pipeline stages 数量
 */
template<int Stages>
class ProducerConsumerPipeline {
public:
    using Pipeline = cutlass::PipelineTmaAsync<Stages>;
    using PipelineState = cutlass::PipelineState<Stages>;

private:
    Pipeline pipeline_;

public:
    /**
     * 构造函数
     *
     * @param shared_storage - Shared memory storage for pipeline
     */
    template<typename SharedStorage>
    __device__ __forceinline__
    ProducerConsumerPipeline(SharedStorage& shared_storage)
        : pipeline_(shared_storage.pipeline) {}

    // ========== Producer 接口 ==========

    /**
     * Producer Acquire
     *
     * Producer 获取一个 shared memory slot
     * 如果 slot 还在被 consumer 使用，会等待
     *
     * @param state - Pipeline state
     */
    __device__ __forceinline__
    void producer_acquire(PipelineState& state) {
        pipeline_.producer_acquire(state);
    }

    /**
     * Producer Get Barrier
     *
     * 获取当前 stage 的 barrier 指针
     * 用于 TMA 操作
     *
     * @param state - Pipeline state
     * @return Barrier 指针
     */
    __device__ __forceinline__
    auto* producer_get_barrier(PipelineState& state) {
        return pipeline_.producer_get_barrier(state);
    }

    /**
     * Producer Release (Commit)
     *
     * Producer 完成数据加载，释放 slot
     * 通知 consumer 数据已就绪
     *
     * @param state - Pipeline state
     */
    __device__ __forceinline__
    void producer_commit(PipelineState& state) {
        pipeline_.producer_commit(state);
    }

    /**
     * Producer Tail
     *
     * Producer 完成所有工作后的 tail 处理
     *
     * @param state - Pipeline state
     */
    __device__ __forceinline__
    void producer_tail(PipelineState& state) {
        pipeline_.producer_tail(state);
    }

    // ========== Consumer 接口 ==========

    /**
     * Consumer Try Wait
     *
     * Consumer 尝试等待数据就绪
     * 返回一个 barrier token
     *
     * @param state - Pipeline state
     * @return Barrier token
     */
    __device__ __forceinline__
    auto consumer_try_wait(PipelineState& state) {
        return pipeline_.consumer_try_wait(state);
    }

    /**
     * Consumer Wait
     *
     * Consumer 等待数据就绪
     * 使用之前 try_wait 返回的 token
     *
     * @param state - Pipeline state
     * @param barrier_token - Barrier token (from try_wait)
     */
    template<typename BarrierToken>
    __device__ __forceinline__
    void consumer_wait(PipelineState& state, BarrierToken const& barrier_token) {
        pipeline_.consumer_wait(state, barrier_token);
    }

    /**
     * Consumer Wait (简化版)
     *
     * 直接等待，不需要显式 token
     *
     * @param state - Pipeline state
     */
    __device__ __forceinline__
    void consumer_wait(PipelineState& state) {
        auto token = consumer_try_wait(state);
        consumer_wait(state, token);
    }

    /**
     * Consumer Release
     *
     * Consumer 完成计算，释放 shared memory slot
     * 允许 producer 重新使用这个 slot
     *
     * @param state - Pipeline state
     */
    __device__ __forceinline__
    void consumer_release(PipelineState& state) {
        pipeline_.consumer_release(state);
    }
};

/**
 * Multi-Pipeline Manager
 *
 * 管理多个独立的 pipelines（例如 Q, K, V 各有一个）
 *
 * @tparam NumPipelines - Pipeline 数量
 * @tparam Stages - 每个 pipeline 的 stages 数量
 */
template<int NumPipelines, int Stages>
class MultiPipelineManager {
public:
    using Pipeline = ProducerConsumerPipeline<Stages>;
    using PipelineState = cutlass::PipelineState<Stages>;

private:
    Pipeline* pipelines_[NumPipelines];

public:
    /**
     * 构造函数
     *
     * @param pipelines - Pipeline 对象数组
     */
    __device__ __forceinline__
    MultiPipelineManager(Pipeline* pipelines[NumPipelines]) {
        for (int i = 0; i < NumPipelines; ++i) {
            pipelines_[i] = pipelines[i];
        }
    }

    /**
     * 获取指定的 pipeline
     *
     * @param idx - Pipeline 索引
     * @return Pipeline 引用
     */
    __device__ __forceinline__
    Pipeline& get_pipeline(int idx) {
        return *pipelines_[idx];
    }

    /**
     * Producer tail (所有 pipelines)
     *
     * @param states - Pipeline states 数组
     */
    __device__ __forceinline__
    void producer_tail_all(PipelineState states[NumPipelines]) {
        for (int i = 0; i < NumPipelines; ++i) {
            pipelines_[i]->producer_tail(states[i]);
        }
    }
};

/**
 * 创建 Pipeline State
 *
 * 工厂函数，初始化 pipeline state
 *
 * @tparam Stages - Pipeline stages 数量
 * @param index - 初始 index (通常是 0)
 * @param phase - 初始 phase (通常是 false)
 * @param count - 初始 count (通常是 0)
 * @return PipelineState 对象
 */
template<int Stages>
__device__ __forceinline__ auto make_pipeline_state(
    int index = 0,
    bool phase = false,
    int count = 0
) {
    return PipelineState<Stages>{index, phase, count};
}

/**
 * Pipeline 辅助函数: Producer 加载并提交
 *
 * 封装了常见的 producer 操作序列
 *
 * @param pipeline - Pipeline 对象
 * @param state - Pipeline state
 * @param load_func - 加载函数 (lambda)
 */
template<typename Pipeline, typename State, typename LoadFunc>
__device__ __forceinline__ void pipeline_producer_load(
    Pipeline& pipeline,
    State& state,
    LoadFunc const& load_func
) {
    // 1. Acquire slot
    pipeline.producer_acquire(state);

    // 2. Load data
    load_func(pipeline.producer_get_barrier(state));

    // 3. Commit (release and advance)
    pipeline.producer_commit(state);
    ++state;
}

/**
 * Pipeline 辅助函数: Consumer 等待并处理
 *
 * 封装了常见的 consumer 操作序列
 *
 * @param pipeline - Pipeline 对象
 * @param state - Pipeline state
 * @param compute_func - 计算函数 (lambda)
 */
template<typename Pipeline, typename State, typename ComputeFunc>
__device__ __forceinline__ void pipeline_consumer_compute(
    Pipeline& pipeline,
    State& state,
    ComputeFunc const& compute_func
) {
    // 1. Wait for data
    pipeline.consumer_wait(state);

    // 2. Compute
    compute_func();

    // 3. Release (allow producer to reuse)
    pipeline.consumer_release(state);
    ++state;
}

}  // namespace sage
