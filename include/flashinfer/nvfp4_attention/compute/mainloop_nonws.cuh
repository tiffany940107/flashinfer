/*
 * Non-WS (Non-Warp-Specialized) Mainloop for NVFP4 Attention
 *
 * 256 threads, no dedicated producer warp group.
 * Thread 0 issues TMA loads; all threads participate in consumer_wait/release.
 *
 * Pipeline prefetch design (kStages=3):
 *   Prologue: thread 0 prefills min(n_blocks, kStages) K/V stages.
 *   Main loop: after consumer_release, thread 0 issues TMA for the next tile
 *   (kStages ahead), overlapping TMA latency with compute.
 *
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../utils/math.cuh"
#include "../utils/layout.cuh"
#include "../primitives/barrier.cuh"
#include "../quantization/fp4_convert.cuh"
#include "mainloop.cuh"  // CollectiveMainloopFwd (reuse Params, TMA transaction bytes, SF helpers)

namespace sage {

using namespace cute;

/**
 * Non-WS Mainloop: fused produce-consume tile loop.
 *
 * Key differences from CollectiveMainloopFwd (WS):
 * - No separate load() function; TMA is issued inline by thread 0.
 * - wg_id is always 0; kBlockMPerWG == kBlockM == 128.
 * - thread_idx is raw threadIdx.x (0-255), no producer offset.
 * - Uses kStages-deep pipeline prefetch: prologue prefills stages,
 *   main loop refills after consumer_release to overlap TMA with compute.
 *
 * Template parameters match CollectiveMainloopFwd so the same Ktraits works.
 */
template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopNonWS {

    // ============ Type aliases (mirrored from CollectiveMainloopFwd) ============
    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int BlockMean = Ktraits::BlockMean;

    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
    using SmemLayoutDS = typename Ktraits::SmemLayoutDS;
    using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
    using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
    using SmemLayoutSFVt = typename Ktraits::SmemLayoutSFVt;

    using SmemCopyAtomQ = typename Ktraits::SmemCopyAtomQ;
    using SmemCopyAtomKV = typename Ktraits::SmemCopyAtomKV;
    using SmemCopyAtomSF = typename Ktraits::SmemCopyAtomSF;

    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;
    static constexpr int NumMmaThreads = size(TiledMmaQK{});

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;

    using LayoutSF = typename Ktraits::LayoutSF;
    using LayoutP = typename Ktraits::LayoutP;
    using LayoutSFP = typename Ktraits::LayoutSFP;
    using SfAtom = typename Ktraits::SfAtom;

    // Reuse Params from CollectiveMainloopFwd (same TMA descriptors, same layout)
    using CollectiveMainloopWS = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using Params = typename CollectiveMainloopWS::Params;

    // Transaction bytes (same as WS mainloop)
    static constexpr uint32_t TmaTransactionBytesQ = CollectiveMainloopWS::TmaTransactionBytesQ;
    static constexpr uint32_t TmaTransactionBytesK = CollectiveMainloopWS::TmaTransactionBytesK;
    static constexpr uint32_t TmaTransactionBytesV = CollectiveMainloopWS::TmaTransactionBytesV;

    // Delegate to WS mainloop for host-side setup
    static Params to_underlying_arguments(typename CollectiveMainloopWS::Arguments const& args) {
        return CollectiveMainloopWS::to_underlying_arguments(args);
    }

    CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& mainloop_params) {
        CollectiveMainloopWS::prefetch_tma_descriptors(mainloop_params);
    }

    CUTLASS_DEVICE int get_n_block_max(Params const& mainloop_params, int m_block) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        }
        return n_block_max;
    }

    // ========================================================================
    // mma_nonws: The core fused produce-consume tile loop.
    //
    // Called from attention_kernel_nonws after pipeline init + TMA tensor setup.
    // All TMA tensors are set up in the kernel entry and passed here.
    //
    // Pipeline protocol (kStages-deep prefetch):
    //   - pipeline_k, pipeline_v: kStages-deep PipelineTmaAsync.
    //   - Thread 0 is ProducerConsumer, others are Consumer.
    //
    //   Prologue: thread 0 prefills min(n_blocks, kStages) K+V stages.
    //
    //   For each N-block i:
    //       1. All threads: consumer_wait(K[i]), smem->reg copy
    //       2. All threads: consumer_release(K[i])
    //       3. Thread 0: refill K[i+kStages] (producer_acquire + TMA)
    //       4. All threads: QK GEMM + softmax
    //       5. All threads: consumer_wait(V[i]), PV GEMM
    //       6. All threads: consumer_release(V[i])
    //       7. Thread 0: refill V[i+kStages] (producer_acquire + TMA)
    //
    // Refill after release is deadlock-free: consumer_release frees a stage,
    // so producer_acquire returns immediately (cluster_size=1).
    // ========================================================================
    template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused,
              // TMA tensor types (auto-deduced from kernel entry)
              typename TKgK_t, typename TKsK_t,
              typename TKgSFK_t, typename TKsSFK_t,
              typename TDSgDS_t, typename TDSsDS_t,
              typename TVgVt_t, typename TVsVt_t,
              typename TVgSFVt_t, typename TVsSFVt_t>
    CUTLASS_DEVICE void
    mma_nonws(
        Params const& mainloop_params,
        MainloopPipelineQ pipeline_q,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineStateQ& smem_pipe_read_q,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        // Write states: thread 0 uses these for TMA issue
        PipelineState& smem_pipe_write_k,
        PipelineState& smem_pipe_write_v,
        FrgTensorO& tOrO_store,
        SoftmaxFused& softmax_fused,
        int n_block_count,       // total N-blocks to process
        int thread_idx,          // threadIdx.x (0-255)
        int m_block,
        SharedStorage& shared_storage,
        // TMA tensors (partitioned, from kernel entry)
        TKgK_t const& tKgK, TKsK_t const& tKsK,
        TKgSFK_t const& tKgSFK, TKsSFK_t const& tKsSFK,
        TDSgDS_t const& tDSgDS, TDSsDS_t const& tDSsDS,
        TVgVt_t const& tVgVt, TVsVt_t const& tVsVt,
        TVgSFVt_t const& tVgSFVt, TVsSFVt_t const& tVsSFVt,
        uint16_t mcast_mask_kv
    ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockK = get<2>(TileShape_MNK{});
        static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;  // 128 for Non-WS

        bool const is_tma_thread = (thread_idx == 0);

        // ============ Smem tensors ============
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
        Tensor sSFQ_full = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

        // ============ Non-WS: full 128 M-rows, wg_id=0 ============
        constexpr int wg_id = 0;
        auto sQ_local = local_tile(sQ, make_shape(Int<kBlockMPerWG>{}, Int<kBlockK>{}), make_coord(wg_id, 0));

        // ============ MMA setup (256 threads, 8 atoms) ============
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

        // For Non-WS, TiledMmaQK == TiledMmaQK_Full (both 8 atoms, 256 threads).
        // SFQ partitioning uses the same thread_idx directly.
        using TiledMmaQK_Full = typename Ktraits::TiledMmaQK_Full;
        TiledMmaQK_Full tiled_mma_qk_full;
        // Non-WS: consumer_thread_idx_full == thread_idx (no WG offset)
        int consumer_thread_idx_full = thread_idx;
        auto thread_mma_qk_full = tiled_mma_qk_full.get_thread_slice(consumer_thread_idx_full);

        // ============ Fragment A/B from smem ============
        Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ_local);
        Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
        Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_, _, Int<0>{}));
        Tensor tOrP = make_tensor_like<Element>(LayoutP{});
        // SFQ: use full MMA (same as 8-atom for Non-WS)
        Tensor tSrSFQ = CollectiveMainloopWS().partition_fragment_SFA(sSFQ_full, thread_mma_qk_full);
        Tensor tSrSFK = CollectiveMainloopWS().partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
        Tensor tOrSFVt = CollectiveMainloopWS().partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
        Tensor tOrSFP = make_tensor<ElementSF>(LayoutSFP{});
        Tensor tOrSFP_flt = filter_zeros(tOrSFP);

        // ============ Smem copy setup ============
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ_local));
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        Tensor tSsK = smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
        Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_pv);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
        Tensor tOsVt = smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sVt));
        Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

        auto tile_shape_mnk = tile_shape(tiled_mma_qk);
        auto tile_shape_mnk_full = tile_shape(tiled_mma_qk_full);

        // SFQ smem copy
        auto smem_tiled_copy_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFA_TV(tiled_mma_qk_full),
                                                        make_shape(size<0>(tile_shape_mnk_full), size<2>(tile_shape_mnk_full)));
        auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(consumer_thread_idx_full);
        Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ_full));
        Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

        // SFK smem copy
        auto smem_tiled_copy_SFK = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFB_TV(tiled_mma_qk),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(thread_idx);
        Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
        Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

        // SFV smem copy
        auto smem_tiled_copy_SFV = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFB_TV(tiled_mma_pv),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(thread_idx);
        Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(as_position_independent_swizzle_tensor(sSFVt));
        Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

        // ============ Helper lambdas ============

        // consumer_wait: try_wait + wait pattern
        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int const unpadded_seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
        constexpr int wg_m_offset = 0;  // Non-WS: always 0

        // copy_k_block: load K + SFK from current smem stage to registers
        auto copy_k_block = [&](auto block_id) {
            auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
            auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());
            copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
        };

        // copy_v_block: load Vt + SFVt from current smem stage to registers
        auto copy_v_block = [&](auto block_id) {
            auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
            auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());
            copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
        };

        // add_delta_s: load delta_s correction from smem to S accumulator
        auto add_delta_s = [&](auto& acc) {
            auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
            auto acc_float4 = recast<float4>(acc);
            int quad_id = (threadIdx.x % 4) * 2;
            for (int i = 0; i < 4; i++) {
                auto num = quad_id + i * 8;
                float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
                float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
                acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
                acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
                acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
                acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
            }
        };

        // S accumulator for 128 M-rows x kBlockN
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
        Tensor tSrS_cv = make_tensor(tSrS.data(), sage::convert_to_conversion_layout(tSrS.layout()));
        Tensor AbsMaxP = make_tensor_like<float>(
            make_layout(shape(group<1, 4>(flatten(tSrS_cv.layout()(make_coord(_0{}, _), _, _))))));

        // Causal mask boundary
        auto col_limit_causal = [&](int row, int n_block) {
            return row + wg_m_offset + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };

        // apply_mask: seqlen + causal masking
        auto apply_mask = [&](auto& tSrS_local, int n_block_local) {
            Tensor cS = cute::make_identity_tensor(make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
            Tensor tScS = thread_mma_qk.partition_C(cS);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrS_local); ++i) {
                if constexpr (!Is_causal) {
                    if (int(get<1>(tScS(i))) >= int(unpadded_seqlen_k - n_block_local * kBlockN)) {
                        tSrS_local(i) = -INFINITY;
                    }
                } else {
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block_local * kBlockN,
                                                        col_limit_causal(int(get<0>(tScS(i))), n_block_local))) {
                        tSrS_local(i) = -INFINITY;
                    }
                }
            }
        };

        // quantize: convert softmax output to FP4 + compute scale factors
        auto quantize = [&](auto mma_k, auto acc_conversion_view) {
            Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
            Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);
            Tensor SFP = make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
            Tensor SFP_uint32_view = recast<uint32_t>(SFP);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
                uint32_t& tmp = SFP_uint32_view(i / 4);
                sage::packed_float_to_ue4m3(
                    AbsMaxP_stagek(i), AbsMaxP_stagek(i + 1),
                    AbsMaxP_stagek(i + 2), AbsMaxP_stagek(i + 3), tmp);
            }
            int const quad_id = threadIdx.x & 3;
            uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);
            Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
            Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    sage::packed_float_to_e2m1(
                        acc_conversion_stagek(make_coord(_0{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_1{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_2{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_3{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_4{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_5{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_6{}, i), mma_m),
                        acc_conversion_stagek(make_coord(_7{}, i), mma_m),
                        tOrP_uint32_view(i, mma_m));
                }
                uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
                uint32_t peer_sfp  = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
                if ((quad_id & 1) == 0) {
                    tOrSFP_uint32_view(_0{}, mma_m) = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
                } else {
                    tOrSFP_uint32_view(_0{}, mma_m) = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
                }
            }
        };

        // ====================================================================
        // PROLOGUE: Wait for Q in smem, then copy to registers.
        //
        // Q is loaded once per work tile. The kernel entry already issued
        // the TMA (producer_acquire + copy) for Q. Here we wait for it
        // to arrive, copy Q + SFQ to registers, then release the stage.
        // ====================================================================
        {
            auto barrier_token_q = pipeline_q.consumer_try_wait(smem_pipe_read_q);
            pipeline_q.consumer_wait(smem_pipe_read_q, barrier_token_q);
        }
        copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);
        pipeline_q.consumer_release(smem_pipe_read_q);
        ++smem_pipe_read_q;

        // ====================================================================
        // MAIN TILE LOOP with kStages pipeline prefetch
        //
        // Prologue: thread 0 prefills min(n_block_count, kStages) K+V stages.
        // Main loop: after consumer_release(K/V), thread 0 refills the next
        //            stage (kStages ahead), overlapping TMA with compute.
        //
        // Execution flow per tile i:
        //   1. consumer_wait(K[i]) -> copy K to regs -> consumer_release(K[i])
        //   2. thread 0: refill K[i+kStages] (producer_acquire + TMA)
        //   3. QK GEMM + softmax
        //   4. consumer_wait(V[i]) -> PV GEMM -> consumer_release(V[i])
        //   5. thread 0: refill V[i+kStages] (producer_acquire + TMA)
        // ====================================================================

        bool is_first_compute = true;

        // MmaN for mask identity tensor
        constexpr int MmaN_qk = decltype(size<2>(tSrS))::value;

        // Helper: issue TMA for K pipeline (K + SFK + DS) for a given n_block
        auto issue_tma_k = [&](int n_block) {
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(mainloop_params.tma_load_K.with(
                *pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
            copy(mainloop_params.tma_load_SFK.with(
                *pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
            copy(mainloop_params.tma_load_DS.with(
                *pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;
        };

        // Helper: issue TMA for V pipeline (Vt + SFVt) for a given n_block
        auto issue_tma_v = [&](int n_block) {
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(mainloop_params.tma_load_Vt.with(
                *pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
            copy(mainloop_params.tma_load_SFVt.with(
                *pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
            ++smem_pipe_write_v;
        };

        // ====================================================================
        // PROLOGUE: Thread 0 prefills min(n_block_count, kStages) stages.
        //
        // N-blocks are processed from n_block_count-1 down to 0, so prefill
        // issues TMA for n_block_count-1, n_block_count-2, ... in order.
        // Other threads will block at consumer_wait until data arrives.
        // ====================================================================
        static constexpr int kStagesLocal = Ktraits::kStages;
        int const prefill_count = cute::min(n_block_count, kStagesLocal);
        if (is_tma_thread) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < prefill_count; ++i) {
                int n_block = n_block_count - 1 - i;
                issue_tma_k(n_block);
                issue_tma_v(n_block);
            }
        }

        // ====================================================================
        // MAIN TILE LOOP (with pipeline prefetch)
        // ====================================================================
        #pragma unroll 1
        for (int tile_idx = 0; tile_idx < n_block_count; ++tile_idx) {
            int n_block = n_block_count - 1 - tile_idx;

            // The next tile to prefetch (kStages ahead of current)
            int prefetch_tile_idx = tile_idx + kStagesLocal;
            int prefetch_n_block = n_block_count - 1 - prefetch_tile_idx;
            bool has_prefetch = prefetch_tile_idx < n_block_count;

            // ================================================================
            // Step 1: All threads wait for K+SFK+DS to be ready in smem.
            // ================================================================
            consumer_wait(pipeline_k, smem_pipe_read_k);

            // ================================================================
            // Step 2: Copy K+SFK from smem to registers, load delta_s.
            //
            // Early K release: copy ALL K blocks to registers first, then
            // release the K pipeline stage so TMA can reuse the smem buffer.
            // ================================================================
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                copy_k_block(k_block);
            }
            add_delta_s(tSrS);

            // ================================================================
            // Step 3: Release K stage (early release - data is in registers).
            // ================================================================
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;

            // ================================================================
            // Step 3b: QK GEMM with staggered MMA issue.
            //
            // NCU shows Math Pipe Throttle = 0.57cy (vs WS 0.23cy) because
            // 8 warps simultaneously flood TC pipeline. Stagger back 4 warps
            // by ~100ns to spread MMA issue and reduce TC congestion.
            // ================================================================

            // Stagger: back 4 warps (threads 128-255) delay briefly
            // to avoid all 8 warps hitting TC pipeline simultaneously.
            if (thread_idx >= 128) {
                asm volatile("nanosleep.u32 100;");
            }

            // --- QK GEMM: Q[128,d] x K[d,128] -> S[128,128] ---
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk,
                    make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                    tSrS);
            }

            // ================================================================
            // Step 4: Softmax + quantize.
            //
            // Masking applied before find_max. Correct for both causal and
            // non-causal: masked positions are -INFINITY and won't affect
            // max/sum.
            // ================================================================
            apply_mask(tSrS, n_block);

            {
                auto prev_block_max = make_fragment_like(softmax_fused.row_max);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                }

                softmax_fused.template chunked_softmax_fixed<>(
                    tSrS, AbsMaxP, is_first_compute,
                    mainloop_params.softmax_scale_log2, prev_block_max);
            }

            // ================================================================
            // Step 4b: K TMA refill DURING softmax (moved from before QK GEMM).
            //
            // NCU shows MIO Throttle = 0.42cy when TMA refill runs alongside
            // QK GEMM (both access smem). Moving refill to after softmax
            // separates TMA from GEMM memory access, reducing MIO congestion.
            // ================================================================
            if (has_prefetch && is_tma_thread) {
                issue_tma_k(prefetch_n_block);
            }

            // ================================================================
            // Step 5: All threads wait for V+SFVt to be ready in smem.
            // ================================================================
            consumer_wait(pipeline_v, smem_pipe_read_v);

            // ================================================================
            // Step 6: PV GEMM with staggered issue.
            // ================================================================

            // Stagger PV GEMM too (same as QK)
            if (thread_idx >= 128) {
                asm volatile("nanosleep.u32 100;");
            }

            copy_v_block(_0{});
            quantize(_0{}, tSrS_cv);

            if (is_first_compute) {
                // First tile: write directly to output accumulator
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv,
                        make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                        make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                        tOrO_store);
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_cv);
                    }
                }
                is_first_compute = false;
            } else {
                // Subsequent tiles: accumulate to temp, then rescale + add
                Tensor tOrO = make_fragment_like(tOrO_store);
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv,
                        make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                        make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                        tOrO);
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_cv);
                    }
                }
                // O_store = O_store * scores_scale + O_new
                softmax_fused.rescale_o(tOrO_store, tOrO);
            }

            // ================================================================
            // Step 7: Release V stage.
            // ================================================================
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;

            // ================================================================
            // Step 7b: Thread 0 refills the next V stage (kStages ahead).
            //
            // Same reasoning as K refill: consumer_release just freed a stage.
            // ================================================================
            if (has_prefetch && is_tma_thread) {
                issue_tma_v(prefetch_n_block);
            }

        } // end tile loop

        // ====================================================================
        // FINALIZE: divide accumulated O by row_sum to get final softmax output.
        // ====================================================================
        softmax_fused.finalize(tOrO_store);
    }
};

}  // namespace sage
