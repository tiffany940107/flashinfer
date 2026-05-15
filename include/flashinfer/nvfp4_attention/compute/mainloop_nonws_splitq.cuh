/*
 * Non-WS Split-Q Mainloop for NVFP4 Attention
 *
 * 256 threads split into 2 groups of 128, each processing 64 M-rows.
 * group_id = threadIdx.x / 128 (0 or 1)
 * mma_thread_idx = threadIdx.x % 128 (per-group 0-127)
 *
 * Pipeline: unchanged from mainloop_nonws.cuh. Both groups synchronously
 * consumer_wait/release on the same pipeline (all 256 threads participate).
 * Thread 0 issues TMA loads.
 *
 * Phase 1: no OrderedSequenceBarrier stagger. Both groups execute
 * identically but on different M-row halves, for correctness validation.
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
 * Non-WS Split-Q Mainloop.
 *
 * Key differences from CollectiveMainloopNonWS:
 * - 256 threads are split into 2 groups of 128 threads each.
 * - Each group uses a 4-atom MMA (128 threads) covering 64 M-rows.
 * - Group 0 processes M-rows 0-63, Group 1 processes M-rows 64-127.
 * - Pipeline is shared: all 256 threads participate in consumer_wait/release.
 * - TMA is issued by thread 0 (same as non-split-Q).
 */
template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopNonWSSplitQ {

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

    // ============ Group-local MMA types ============
    // SPLIT_Q mode: Ktraits::TiledMmaQK uses AtomLayout<_4> (128 threads, 64 M-rows)
    // which is exactly what each 128-thread group needs. No need to redefine.
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;
    static constexpr int NumMmaThreads = size(TiledMmaQK{});  // 128

    // Full 8-atom MMA (256 threads) for SFQ partitioning and pipeline
    using TiledMmaQK_Full = typename Ktraits::TiledMmaQK_Full;
    using TiledMmaPV_Full = typename Ktraits::TiledMmaPV_Full;

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
    // mma_nonws_splitq: Split-Q fused produce-consume tile loop.
    //
    // 256 threads split into 2 groups of 128. Each group processes 64 M-rows
    // using a 4-atom MMA. Pipeline is shared (all 256 threads).
    // ========================================================================
    template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused,
              typename MathOrderBarrier,
              // TMA tensor types (auto-deduced from kernel entry)
              typename TKgK_t, typename TKsK_t,
              typename TKgSFK_t, typename TKsSFK_t,
              typename TDSgDS_t, typename TDSsDS_t,
              typename TVgVt_t, typename TVsVt_t,
              typename TVgSFVt_t, typename TVsSFVt_t>
    CUTLASS_DEVICE void
    mma_nonws_splitq(
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
        MathOrderBarrier& math_order,
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
        static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;  // 64 in SPLIT_Q

        // ============ Split-Q: compute group_id and per-group thread index ============
        int const group_id = thread_idx / 128;       // 0 or 1
        int const mma_thread_idx = thread_idx % 128;  // per-group 0-127
        int const wg_m_offset = group_id * 64;        // M-row offset for this group

        bool const is_tma_thread = (thread_idx == 0);

        // ============ Smem tensors ============
        Tensor sQ_full = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
        Tensor sSFQ_full = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

        // ============ Per-group Q slice (64 M-rows) ============
        auto sQ_group = local_tile(sQ_full, make_shape(Int<kBlockMPerWG>{}, Int<kBlockK>{}), make_coord(group_id, 0));

        // ============ Group-local MMA setup (128 threads, 4 atoms) ============
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(mma_thread_idx);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(mma_thread_idx);

        // Full 8-atom MMA for SFQ partitioning (uses all 256 threads).
        // consumer_thread_idx_full maps group 0 -> atoms 0-3 (rows 0-63),
        //                                group 1 -> atoms 4-7 (rows 64-127).
        TiledMmaQK_Full tiled_mma_qk_full;
        int consumer_thread_idx_full = mma_thread_idx + group_id * NumMmaThreads;
        auto thread_mma_qk_full = tiled_mma_qk_full.get_thread_slice(consumer_thread_idx_full);

        // ============ Fragment A/B from smem ============
        // Q: partitioned with group-local MMA over group's 64 M-rows
        Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ_group);
        // K: shared across groups (same kBlockN x kHeadDim tile)
        Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
        // V: shared across groups
        Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_, _, Int<0>{}));
        Tensor tOrP = make_tensor_like<Element>(LayoutP{});

        // SFQ: use full MMA (8 atoms, 256 threads) for correct SFQ partitioning
        Tensor tSrSFQ = CollectiveMainloopWS().partition_fragment_SFA(sSFQ_full, thread_mma_qk_full);
        // SFK/SFV: use group-local thread slices (partition_fragment_SFB requires ThrMma, not TiledMma)
        Tensor tSrSFK = CollectiveMainloopWS().partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
        Tensor tOrSFVt = CollectiveMainloopWS().partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
        Tensor tOrSFP = make_tensor<ElementSF>(LayoutSFP{});
        Tensor tOrSFP_flt = filter_zeros(tOrSFP);

        // ============ Smem copy setup (using group-local MMA + mma_thread_idx) ============
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(mma_thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ_group));
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(mma_thread_idx);
        Tensor tSsK = smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
        Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_pv);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(mma_thread_idx);
        Tensor tOsVt = smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sVt));
        Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

        auto tile_shape_mnk = tile_shape(tiled_mma_qk);
        auto tile_shape_mnk_full = tile_shape(tiled_mma_qk_full);

        // SFQ smem copy (full 8-atom MMA, consumer_thread_idx_full)
        auto smem_tiled_copy_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFA_TV(tiled_mma_qk_full),
                                                        make_shape(size<0>(tile_shape_mnk_full), size<2>(tile_shape_mnk_full)));
        auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(consumer_thread_idx_full);
        Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ_full));
        Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

        // SFK smem copy (group-local MMA)
        auto smem_tiled_copy_SFK = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFB_TV(tiled_mma_qk),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(mma_thread_idx);
        Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
        Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

        // SFV smem copy (group-local MMA)
        auto smem_tiled_copy_SFV = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFB_TV(tiled_mma_pv),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(mma_thread_idx);
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
        // Note: quad_id is based on mma_thread_idx (per-group), matching 4-atom MMA layout.
        auto add_delta_s = [&](auto& acc) {
            auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
            auto acc_float4 = recast<float4>(acc);
            int quad_id = (mma_thread_idx % 4) * 2;
            for (int i = 0; i < 4; i++) {
                auto num = quad_id + i * 8;
                // Offset delta_s rows by wg_m_offset for this group's 64 M-rows
                // delta_s layout: (kBlockM, kBlockN) with stride (_0, _1) => broadcast in M
                // So the same delta_s values apply (M dimension is broadcast).
                float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
                float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
                acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
                acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
                acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
                acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
            }
        };

        // S accumulator: 64 M-rows x kBlockN (per group)
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
        Tensor tSrS_cv = make_tensor(tSrS.data(), sage::convert_to_conversion_layout(tSrS.layout()));
        Tensor AbsMaxP = make_tensor_like<float>(
            make_layout(shape(group<1, 4>(flatten(tSrS_cv.layout()(make_coord(_0{}, _), _, _))))));

        // Causal mask boundary: row is group-local (0-63), add wg_m_offset for global M position
        auto col_limit_causal = [&](int row, int n_block) {
            return row + wg_m_offset + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };

        // apply_mask: seqlen + causal masking (64 M-rows per group)
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
            // quad_id within the group (per mma_thread_idx)
            int const quad_id = mma_thread_idx & 3;
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
        // PROLOGUE: Wait for Q in smem, then copy group's Q slice to registers.
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
        // ====================================================================

        bool is_first_compute = true;

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
            // Step 1: All 256 threads wait for K+SFK+DS to be ready in smem.
            // ================================================================
            consumer_wait(pipeline_k, smem_pipe_read_k);

            // ================================================================
            // Step 2: Copy K+SFK from smem to registers, load delta_s.
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
            // Step 3b: Thread 0 refills the next K stage.
            // ================================================================
            if (has_prefetch && is_tma_thread) {
                issue_tma_k(prefetch_n_block);
            }

            // --- QK GEMM: Q[64,d] x K[d,128] -> S[64,128] (per group) ---
            // Stagger: Group 1 goes first → Group 0 does softmax while Group 1 does QK
            math_order.wait();
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk,
                    make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                    tSrS);
            }
            math_order.arrive();

            // ================================================================
            // Step 4: Softmax + quantize (per group, 64 M-rows).
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
            // Step 5: All 256 threads wait for V+SFVt to be ready in smem.
            // ================================================================
            consumer_wait(pipeline_v, smem_pipe_read_v);

            // ================================================================
            // Step 6: PV GEMM: P[64,128] x V[128,d] -> O[64,d] (per group).
            // Stagger: Group 1 goes first → Group 1's PV uses TC while Group 0 still in softmax
            // ================================================================
            math_order.wait();
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

            math_order.arrive();

            // ================================================================
            // Step 7: Release V stage.
            // ================================================================
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;

            // ================================================================
            // Step 7b: Thread 0 refills the next V stage.
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
