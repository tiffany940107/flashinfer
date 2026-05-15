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
 *
 * Non-WS Attention Kernel
 *
 * 256 threads. Uses CollectiveMainloopNonWS for fused produce-consume tile loop.
 * Thread 0 issues TMA inline in the tile loop (no separate producer WG).
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
#include "../compute/mainloop_nonws.cuh"
#if defined(SPLIT_Q)
#include "../compute/mainloop_nonws_splitq.cuh"
#endif
#if defined(CROSS_TILE)
#include "../compute/mainloop_nonws_crosstile.cuh"
#endif
#include "../compute/epilogue.cuh"
#include "scheduler.h"
#include "traits.h"

namespace sage {

using namespace cute;

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    attention_kernel_nonws(
        CUTE_GRID_CONSTANT Flash_fwd_params const params,
        CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
        CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
        CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
    ) {

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int NumThreads = Ktraits::kNThreads;

    using CollectiveMainloopWS = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using CollectiveMainloopNW = CollectiveMainloopNonWS<Ktraits, Is_causal>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const thread_idx = threadIdx.x;
    bool const is_tma_thread = (thread_idx == 0);

    // ============ Prefetch TMA descriptors ============
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloopWS::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // ============ Pipeline init ============
    PipelineParams pipeline_params_k;
    pipeline_params_k.transaction_bytes = CollectiveMainloopWS::TmaTransactionBytesK;
    pipeline_params_k.role = is_tma_thread
        ? MainloopPipeline::ThreadCategory::ProducerConsumer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_k.is_leader = is_tma_thread;
    pipeline_params_k.num_consumers = NumThreads;

    PipelineParams pipeline_params_v;
    pipeline_params_v.transaction_bytes = CollectiveMainloopWS::TmaTransactionBytesV;
    pipeline_params_v.role = is_tma_thread
        ? MainloopPipeline::ThreadCategory::ProducerConsumer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params_v.is_leader = is_tma_thread;
    pipeline_params_v.num_consumers = NumThreads;

    typename Ktraits::PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = CollectiveMainloopWS::TmaTransactionBytesQ;
    pipeline_params_q.role = is_tma_thread
        ? MainloopPipelineQ::ThreadCategory::ProducerConsumer
        : MainloopPipelineQ::ThreadCategory::Consumer;
    pipeline_params_q.is_leader = is_tma_thread;
    pipeline_params_q.num_consumers = NumThreads;

    MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q, ClusterShape{});
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params_k, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params_v, ClusterShape{});

    // 256 = max regs/thread for 256 threads (65536/256).
    // Cross-tile double-buffer needs the full budget.
    cutlass::arch::warpgroup_reg_alloc<256>();

    CollectiveMainloopWS collective_mainloop_ws;
    CollectiveMainloopNW collective_mainloop_nw;
    CollectiveEpilogue collective_epilogue;

    // Pipeline state persists across work tiles
    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    typename Ktraits::PipelineStateQ smem_pipe_read_q;
    auto smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
    auto smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
    auto smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

    __syncthreads();

    // ============ Per-work-tile loop ============
    typename Ktraits::TiledMmaPV tiled_mma_pv;
#if defined(CROSS_TILE)
    typename Ktraits::TiledMmaQK tiled_mma_qk;
#endif
    TileScheduler scheduler{};
    int work_idx = 0;

    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.get_next_work(scheduler_params, work_tile_info)) {

        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [m_block, bidh, bidb] = block_coord;
        int n_block_max = collective_mainloop_ws.get_n_block_max(mainloop_params, m_block);

        if (Is_causal && n_block_max <= 0) {
            collective_epilogue.store_zero(epilogue_params, thread_idx, block_coord);
            continue;
        }

        // ============ TMA tensor setup ============
        using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
        using SmemLayoutK = typename Ktraits::SmemLayoutK;
        using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
        using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
        using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
        using SmemLayoutSFVt = typename Ktraits::SmemLayoutSFVt;
        using SmemLayoutDS = typename Ktraits::SmemLayoutDS;

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
        Tensor mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);
        Tensor mDS = mainloop_params.tma_load_DS.get_tma_tensor(shape(mainloop_params.layout_DS));
        Tensor mSFQ = mainloop_params.tma_load_SFQ.get_tma_tensor(shape(mainloop_params.layout_SFQ));
        Tensor mSFK = mainloop_params.tma_load_SFK.get_tma_tensor(shape(mainloop_params.layout_SFK));
        Tensor mSFVt = mainloop_params.tma_load_SFVt.get_tma_tensor(shape(mainloop_params.layout_SFVt));

        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                         block_rank_in_cluster / cluster_shape_x};

        Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gVt = local_tile(mVt(_, _, bidh, bidb),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));
        Tensor gSFQ = local_tile(mSFQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gSFK = local_tile(mSFK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gSFVt = local_tile(mSFVt(_, _, bidh, bidb),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));
        Tensor gDS = [&] {
            if constexpr (Ktraits::BlockMean) {
                return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}), make_coord(m_block, _));
            } else {
                return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}), make_coord(_0{}, _));
            }
        }();

        auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
        auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
        auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
        auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
        auto block_tma_ds = mainloop_params.tma_load_DS.get_slice(cluster_local_block_id.x);
        auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
        auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});

        Tensor tQgQ = block_tma_q.partition_S(gQ);
        Tensor tQsQ = block_tma_q.partition_D(sQ);
        Tensor tQgSFQ = block_tma_sfq.partition_S(gSFQ);
        Tensor tQsSFQ = block_tma_sfq.partition_D(sSFQ);
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));
        Tensor tKgSFK = group_modes<0, 3>(block_tma_sfk.partition_S(gSFK));
        Tensor tKsSFK = group_modes<0, 3>(block_tma_sfk.partition_D(sSFK));
        Tensor tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));
        Tensor tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));
        Tensor tVgSFVt = group_modes<0, 3>(block_tma_sfvt.partition_S(gSFVt));
        Tensor tVsSFVt = group_modes<0, 3>(block_tma_sfvt.partition_D(sSFVt));
        Tensor tDSgDS = group_modes<0, 3>(block_tma_ds.partition_S(gDS));
        Tensor tDSsDS = group_modes<0, 3>(block_tma_ds.partition_D(sDS));
        uint16_t mcast_mask_kv = 0;

        // ============ Q load (thread 0 issues TMA) ============
        if (is_tma_thread) {
            pipeline_q.producer_acquire(smem_pipe_write_q);
            copy(mainloop_params.tma_load_Q.with(
                *pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgQ, tQsQ);
            copy(mainloop_params.tma_load_SFQ.with(
                *pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgSFQ, tQsSFQ);
            ++smem_pipe_write_q;
        }

        // ============ Compute ============
        {
            static constexpr int kBlockN = get<1>(TileShape_MNK{});

            Tensor tOrO = partition_fragment_C(tiled_mma_pv,
                make_shape(Int<kBlockMPerWG>{}, Int<kHeadDim>{}));

            sage::SoftmaxFused<2 * (2 * kBlockMPerWG /
                size(typename Ktraits::TiledMmaQK{}))> softmax_fused;

#if defined(SPLIT_Q)
            // ---- Split-Q: 2 groups × 128 threads, each processes 64 M-rows ----
            {
                int group_id = thread_idx / 128;
                int mma_thread_idx_group = thread_idx % 128;

                // MathOrder barrier for compute stagger
                using MathOrderBarrier = typename Ktraits::MathOrderBarrier;
                uint32_t mo_sizes[2] = {128, 128};  // 2 groups × 128 threads
                typename MathOrderBarrier::Params mo_params;
                mo_params.group_id = (group_id == 1) ? 1 : 0;  // Group 1 goes first
                mo_params.group_size_list = mo_sizes;
                MathOrderBarrier math_order(shared_storage.math_order, mo_params);

                // Call split-Q mainloop (group_id computed internally)
                CollectiveMainloopNonWSSplitQ<Ktraits, Is_causal>().mma_nonws_splitq(
                    mainloop_params,
                    pipeline_q, pipeline_k, pipeline_v,
                    smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                    smem_pipe_write_k, smem_pipe_write_v,
                    tOrO, softmax_fused,
                    n_block_max, thread_idx, m_block,
                    shared_storage, math_order,
                    tKgK, tKsK, tKgSFK, tKsSFK, tDSgDS, tDSsDS,
                    tVgVt, tVsVt, tVgSFVt, tVsSFVt,
                    mcast_mask_kv
                );
            }
#elif defined(CROSS_TILE)
            // ---- Cross-tile double-buffer: tSrS[0]/[1] and AbsMaxP[0]/[1] ----
            Tensor tSrS_buf0 = partition_fragment_C(tiled_mma_qk,
                make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
            Tensor tSrS_buf1 = partition_fragment_C(tiled_mma_qk,
                make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));

            auto tSrS_buf0_cv = make_tensor(tSrS_buf0.data(),
                sage::convert_to_conversion_layout(tSrS_buf0.layout()));
            auto tSrS_buf1_cv = make_tensor(tSrS_buf1.data(),
                sage::convert_to_conversion_layout(tSrS_buf1.layout()));

            Tensor AbsMaxP_buf0 = make_tensor_like<float>(
                make_layout(shape(group<1, 4>(flatten(
                    tSrS_buf0_cv.layout()(make_coord(_0{}, _), _, _))))));
            Tensor AbsMaxP_buf1 = make_tensor_like<float>(
                make_layout(shape(group<1, 4>(flatten(
                    tSrS_buf1_cv.layout()(make_coord(_0{}, _), _, _))))));

            // Call cross-tile mainloop (uses separate class)
            CollectiveMainloopNonWSCrosstile<Ktraits, Is_causal>().mma_nonws_crosstile(
                mainloop_params,
                pipeline_q, pipeline_k, pipeline_v,
                smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                smem_pipe_write_k, smem_pipe_write_v,
                tOrO, softmax_fused,
                n_block_max, thread_idx, m_block,
                shared_storage,
                tSrS_buf0, tSrS_buf1, AbsMaxP_buf0, AbsMaxP_buf1,
                tKgK, tKsK, tKgSFK, tKsSFK, tDSgDS, tDSsDS,
                tVgVt, tVsVt, tVgSFVt, tVsSFVt,
                mcast_mask_kv
            );
#else
            // ---- Baseline mainloop (no cross-tile) ----
            collective_mainloop_nw.mma_nonws(
                mainloop_params,
                pipeline_q, pipeline_k, pipeline_v,
                smem_pipe_read_q, smem_pipe_read_k, smem_pipe_read_v,
                smem_pipe_write_k, smem_pipe_write_v,
                tOrO, softmax_fused,
                n_block_max, thread_idx, m_block,
                shared_storage,
                tKgK, tKsK, tKgSFK, tKsSFK, tDSgDS, tDSsDS,
                tVgVt, tVsVt, tVgSFVt, tVsSFVt,
                mcast_mask_kv
            );
#endif

            // ============ Epilogue ============
            collective_epilogue.mma_store(
                shared_storage, tiled_mma_pv, tOrO, thread_idx, /*wg_id=*/0);
            cutlass::arch::fence_view_async_shared();
            __syncthreads();
            if (is_tma_thread) {
                collective_epilogue.tma_store(
                    shared_storage, epilogue_params,
                    work_tile_info, scheduler_params, thread_idx);
                collective_epilogue.store_tail();
            }
            __syncthreads();
        }

        work_idx++;
    }
}

}  // namespace sage
