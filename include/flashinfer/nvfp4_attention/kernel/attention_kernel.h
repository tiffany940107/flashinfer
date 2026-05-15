/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "../common/params.h"
#include "../compute/mainloop.cuh"
#include "../compute/epilogue.cuh"
#include "../compute/producer/load_q.cuh"
#include "../compute/producer/load_k.cuh"
#include "../compute/producer/load_v.cuh"
#include "../compute/consumer/qk_gemm.cuh"
#include "../compute/consumer/softmax.cuh"
#include "../compute/consumer/pv_gemm.cuh"
#include "../compute/consumer/delta_correction.cuh"
#include "../compute/epilogue/output_writer.cuh"
#include "../compute/epilogue/lse_writer.cuh"
#include "scheduler.h"
#include "traits.h"

namespace sage {

using namespace cute;

/**
 * SageAttention3 Kernel - Warp Specialization 版本
 *
 * 使用 Warp Specialization 策略：
 * - Producer warp group: 负责 TMA 加载 Q/K/V 和存储结果
 * - Consumer warp groups: 负责计算 Q@K, Softmax, P@V
 *
 * 架构特点：
 * 1. **模块化**: 调用已重构的 producer/consumer/epilogue 模块
 * 2. **Pipeline**: 使用 TMA async pipeline 隐藏内存延迟
 * 3. **FP4 量化**: Q/K/V 使用 FP4 存储，P 进行在线量化
 * 4. **OMMA**: 使用 Hopper 的 FP4 Block GEMM 指令
 * 5. **Online Softmax**: 实现数值稳定的 online softmax
 *
 * 模板参数：
 * @tparam Ktraits: Kernel traits (定义配置)
 * @tparam Is_causal: 是否使用 causal mask
 * @tparam TileScheduler: Tile 调度器类型
 */
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    attention_kernel_ws(
        CUTE_GRID_CONSTANT Flash_fwd_params const params,
        CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
        CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
        CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
    ) {

    // ============ 类型定义 ============
    using Element = typename Ktraits::Element;
    using ElementAccum = typename Ktraits::ElementAccum;
    using SoftType = ElementAccum;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMmaQK{});  // 128 (per WG)
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;     // 128 (producer WG)
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;  // 64

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

    // ============ Warp 角色定义 ============
    enum class WarpGroupRole {
        Producer = 0,   // 负责加载和存储
        Consumer0 = 1,  // 负责计算
        Consumer1 = 2   // 负责计算
    };
    enum class ProducerWarpRole {
        Mainloop = 0,   // 负责 Q/K/V 加载
        Epilogue = 1,   // 负责结果存储
        Warp2 = 2,      // 保留
        Warp3 = 3       // 保留
    };

    // ============ Shared Memory 初始化 ============
    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    // ============ 线程角色分配 ============
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

    // ============ 预取 TMA 描述符 ============
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // ============ Pipeline 初始化 ============
    // Scheme A: all pipelines use num_consumers = 2*NumMmaThreads = 256
    // Both WGs participate in wait/release for every stage.
    // Only the "owning" WG computes; the other WG skips and releases immediately.
    static constexpr int NumAllConsumerThreads = 2 * NumMmaThreads;  // 256

    PipelineParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    pipeline_params_v.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_v.is_leader = warp_group_thread_idx == 0;
    pipeline_params_v.num_consumers = NumAllConsumerThreads;

    PipelineParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = NumAllConsumerThreads;

    PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    pipeline_params_q.role = warp_group_role == WarpGroupRole::Producer
        ? MainloopPipelineQ::ThreadCategory::Producer
        : MainloopPipelineQ::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = NumAllConsumerThreads;

    // 创建 pipeline 对象
    MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q, ClusterShape{});
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params_v, ClusterShape{});

    // ============ Epilogue Barrier 初始化 ============
#if defined(ASYNC_EPILOGUE)
    // ASYNC_EPILOGUE: barrier_o syncs between two consumer WGs only.
    // Producer warps must NOT participate (group_id doesn't matter for them,
    // but they must never call arrive/wait on barrier_o).
    uint32_t epilogue_barrier_group_size_list[2] = {cutlass::NumThreadsPerWarpGroup, cutlass::NumThreadsPerWarpGroup};
    typename EpilogueBarrier::Params params_epilogue_barrier;
    if (warp_group_role == WarpGroupRole::Consumer0) {
        params_epilogue_barrier.group_id = 0;
    } else if (warp_group_role == WarpGroupRole::Consumer1) {
        params_epilogue_barrier.group_id = 1;
    } else {
        // Producer: set group_id = 0 but NEVER call arrive/wait
        params_epilogue_barrier.group_id = 0;
    }
    params_epilogue_barrier.group_size_list = epilogue_barrier_group_size_list;
    EpilogueBarrier barrier_o(shared_storage.barrier_o, params_epilogue_barrier);
#else
    uint32_t epilogue_barrier_group_size_list[2] = {cutlass::NumThreadsPerWarp, NumAllConsumerThreads};
    typename EpilogueBarrier::Params params_epilogue_barrier;
    params_epilogue_barrier.group_id = (warp_group_role == WarpGroupRole::Producer);
    params_epilogue_barrier.group_size_list = epilogue_barrier_group_size_list;
    EpilogueBarrier barrier_o(shared_storage.barrier_o, params_epilogue_barrier);
#endif

    // ============ Math Order Barrier (ping-pong between Consumer0/Consumer1) ============
    using MathOrderBarrier = typename Ktraits::MathOrderBarrier;
    uint32_t math_order_group_sizes[2] = {cutlass::NumThreadsPerWarpGroup, cutlass::NumThreadsPerWarpGroup};
    typename MathOrderBarrier::Params math_order_params;
    // OrderedSequenceBarrier: group 1 goes first (wait passes immediately), group 0 waits.
    // Consumer0 = group 0 (waits for signal), Consumer1 = group 1 (goes first on QK GEMM).
    // Producer = group 0 (does not participate in math_order wait/arrive).
    math_order_params.group_id = (warp_group_role == WarpGroupRole::Consumer1) ? 1 : 0;
    math_order_params.group_size_list = math_order_group_sizes;
    MathOrderBarrier math_order(shared_storage.math_order, math_order_params);
#if defined(STEADY_SPLIT_QK_PV_ORDER)
    MathOrderBarrier math_order_pv(shared_storage.math_order_pv, math_order_params);
#endif

    // ============ R18 mbarrier 初始化 ============
    // Hardware mbarrier for OMMA handoff (1 arrival per signal from leader thread)
    if (warp_idx == 0 && lane_predicate) {
        shared_storage.r18_mbar[0].init(1);  // WG0→WG1
        shared_storage.r18_mbar[1].init(1);  // WG1→WG0
        shared_storage.r18_wait_phase[0] = 0;
        shared_storage.r18_wait_phase[1] = 0;
    }
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
    if (warp_idx == 0 && lane_predicate) {
        shared_storage.strict_phase_gate[0].init(cutlass::NumThreadsPerWarpGroup);
        shared_storage.strict_phase_gate[1].init(cutlass::NumThreadsPerWarpGroup);
    }
    cutlass::arch::fence_barrier_init();
#endif

    // ============ Collective 对象 ============
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    __syncthreads();

    // ============================================
    // Producer Warp Group: 负责数据加载和存储
    // ============================================
    if (warp_group_role == WarpGroupRole::Producer) {
        // 减少寄存器分配 (Producer 不需要太多寄存器)
        cutlass::arch::warpgroup_reg_dealloc<24>();

        TileScheduler scheduler;

        // --- Mainloop Warp: 加载 Q/K/V ---
        if (producer_warp_role == ProducerWarpRole::Mainloop) {
            PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            // 遍历所有工作 tiles
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

                int tile_count_semaphore = 0;

                // 调用重构的 load 模块
                collective_mainloop.load(
                    mainloop_params, scheduler_params,
                    pipeline_q, pipeline_k, pipeline_v,
                    smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v,
                    shared_storage, work_tile_info, work_idx, tile_count_semaphore
                );

                work_idx++;
            }

            // Producer tail: 等待所有传输完成
            collective_mainloop.load_tail(
                pipeline_q, pipeline_k, pipeline_v,
                smem_pipe_write_q, smem_pipe_write_k, smem_pipe_write_v
            );
        }
        // --- Epilogue Warp: 存储结果 ---
        else if (producer_warp_role == ProducerWarpRole::Epilogue) {
#if defined(ASYNC_EPILOGUE)
            // ASYNC_EPILOGUE: consumer 自己发 TMA store, epilogue warp 空闲
            // 不需要做任何事情 — TMA store 由 consumer WGs 直接发出
#else
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

                // 等待 consumer 计算完成
                barrier_o.wait();

                // 调用重构的 epilogue 模块
                collective_epilogue.tma_store(
                    shared_storage, epilogue_params,
                    work_tile_info, scheduler_params, threadIdx.x
                );

                collective_epilogue.store_tail();

                // 通知 consumer 可以写入新数据
                barrier_o.arrive();
            }
#endif
        }
    }

    // ============================================
    // Consumer Warp Groups: 负责计算
    // ============================================
    else if (warp_group_role == WarpGroupRole::Consumer0 ||
             warp_group_role == WarpGroupRole::Consumer1) {

        // 增加寄存器分配 (Consumer 需要更多寄存器)
        cutlass::arch::warpgroup_reg_alloc<232>();

        typename Ktraits::TiledMmaPV tiled_mma_pv;
        TileScheduler scheduler{};

        // === Ping-pong: per-WG 配置 ===
        // consumer_thread_idx: WG1=0-127, WG2=128-255 (用于 store_zero 等全局操作)
        // mma_thread_idx: 两个 WG 都是 0-127 (用于 MMA partition)
        int consumer_thread_idx = threadIdx.x - NumCopyThreads;
        int wg_id = (warp_group_role == WarpGroupRole::Consumer1) ? 1 : 0;
        int mma_thread_idx = consumer_thread_idx % NumMmaThreads;  // 0-127 for both WGs

        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        PipelineStateQ smem_pipe_read_q;
        // Scheme A: both WGs start at stage 0, advance by 1 (normal)

#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
        int strict_phase_wait_phase = 0;
#endif
        int work_idx = 0;

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

            // === 初始化输出累加器 (64 M-rows per WG) ===
            Tensor tOrO = partition_fragment_C(tiled_mma_pv, make_shape(Int<kBlockMPerWG>{}, Int<Ktraits::kHeadDim>{}));

            // === 初始化 Softmax ===
            sage::SoftmaxFused<2 * (2 * kBlockMPerWG / NumMmaThreads)> softmax_fused;

            // === 获取 block 坐标 ===
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, bidh, bidb] = block_coord;

            // === 计算需要处理的 N blocks 数量 (基于 full kBlockM=128) ===
            int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block);

            // === Causal attention: 提前退出 ===
            if (Is_causal && n_block_max <= 0) {
                collective_epilogue.store_zero(
                    epilogue_params, consumer_thread_idx, block_coord
                );
                continue;
            }

#if defined(ASYNC_EPILOGUE)
            // === ASYNC_EPILOGUE: 等待上一次 TMA store 完成, smem_O 可安全重用 ===
            if (work_idx > 0) {
                if (warp_group_role == WarpGroupRole::Consumer0 && warp_group_thread_idx == 0) {
                    cute::tma_store_wait<0>();
                }
                // Consumer-only sync: 256 consumer threads (128-383), producer不参与
                // NamedBarrier id=7 (EpilogueBarrier), count=256
                cutlass::arch::NamedBarrier::sync(NumAllConsumerThreads, 0 /*user_barrier_0*/);
            }
#endif

            // === 主计算循环 ===
            collective_mainloop.mma(
                mainloop_params,
                pipeline_q, pipeline_k, pipeline_v,
                smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                tOrO, softmax_fused, n_block_max,
                mma_thread_idx, work_idx, m_block, wg_id,
                shared_storage, math_order,
#if defined(STEADY_SPLIT_QK_PV_ORDER)
                math_order_pv
#else
                math_order
#endif
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
                , strict_phase_wait_phase
#endif
            );

#if defined(ASYNC_EPILOGUE)
            // === ASYNC_EPILOGUE: consumer 自己做 mma_store + tma_store ===

            // 不需要 barrier_o.wait() — 没有 epilogue warp 争用 smem_O
            // smem_O 的安全性已由上面的 tma_store_wait 保证

            // 存储结果到 shared memory (each WG writes its 64-row half)
            collective_epilogue.mma_store(
                shared_storage, tiled_mma_pv, tOrO, mma_thread_idx, wg_id
            );

            // Consumer-only sync: 确保两个 WG 的 mma_store 都完成后再发 TMA
            cutlass::arch::NamedBarrier::sync(NumAllConsumerThreads, 0 /*user_barrier_0*/);

            // Consumer0 thread 0 发起异步 TMA store (不等待完成)
            if (warp_group_role == WarpGroupRole::Consumer0 && warp_group_thread_idx == 0) {
                collective_epilogue.tma_store(
                    shared_storage, epilogue_params,
                    work_tile_info, scheduler_params, 0
                );
                // 注意: 不调 store_tail() — 异步, 下一 tile 开头再等
            }

            // 不需要 barrier_o.arrive() — 没有 epilogue warp 需要通知
#else
            // === 等待 producer 准备好存储空间 ===
            barrier_o.wait();

            // === 存储结果到 shared memory (each WG writes its 64-row half) ===
            collective_epilogue.mma_store(
                shared_storage, tiled_mma_pv, tOrO, mma_thread_idx, wg_id
            );

            // === 通知 producer 可以存储 ===
            barrier_o.arrive();
#endif

            ++work_idx;
        }

#if defined(ASYNC_EPILOGUE)
        // === ASYNC_EPILOGUE: 最后一个 tile 的 TMA store 等待完成 ===
        if (work_idx > 0) {
            if (warp_group_role == WarpGroupRole::Consumer0 && warp_group_thread_idx == 0) {
                cute::tma_store_wait<0>();
            }
        }
#endif
    }
}

} // namespace sage
