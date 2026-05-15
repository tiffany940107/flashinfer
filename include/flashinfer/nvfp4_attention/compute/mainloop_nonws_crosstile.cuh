/*
 * Cross-Tile Double-Buffer Mainloop for NVFP4 Attention (Non-WS)
 *
 * Key idea: softmax[i-1] is interleaved with QK GEMM[i] in N-sub-tile gaps.
 *
 * Double buffer: tSrS[cur] holds the previous tile's complete scores (for softmax),
 * tSrS[nxt] accumulates the current tile's QK GEMM output.
 *
 * Softmax decomposition (Scheme 2 from crosstile_design.md):
 *   find_max_all:  full-tile max finding, done after apply_mask, before interleave
 *   exp2_sum_chunk: per-N-sub-tile exp2+sum, inserted in QK GEMM[i+1] MMA gaps
 *   quantize:       after all exp2_sum_chunk complete
 *
 * Pipeline prefetch: same kStages-deep design as mainloop_nonws.cuh.
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
#include "mainloop.cuh"

namespace sage {

using namespace cute;

/**
 * Non-WS Cross-Tile Double-Buffer Mainloop.
 *
 * Execution flow:
 *
 * Prologue:
 *   QK GEMM tile 0 -> tSrS[cur]
 *   apply_mask + find_max_all(tSrS[cur])
 *
 * Steady-state (tile i = 1..N-1):
 *   Phase A: Wait K[i], copy to regs
 *            N-sub-tile interleaved: QK GEMM[i] -> tSrS[nxt] + exp2_sum_chunk(tSrS[cur])
 *            quantize(tSrS[cur])
 *   Phase B: Wait V[i-1], PV GEMM(P[cur], V[i-1])
 *            apply_mask(tSrS[nxt]) + find_max_all(tSrS[nxt])
 *            swap(cur, nxt)
 *
 * Epilogue:
 *   find_max_all + exp2_sum + quantize(tSrS[cur])  (no GEMM to interleave)
 *   PV GEMM(P[cur], V[N-1])
 *   finalize
 */
template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopNonWSCrosstile {

    // ============ Type aliases (mirrored from CollectiveMainloopNonWS) ============
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

    using CollectiveMainloopWS = CollectiveMainloopFwd<Ktraits, Is_causal>;
    using Params = typename CollectiveMainloopWS::Params;

    static constexpr uint32_t TmaTransactionBytesQ = CollectiveMainloopWS::TmaTransactionBytesQ;
    static constexpr uint32_t TmaTransactionBytesK = CollectiveMainloopWS::TmaTransactionBytesK;
    static constexpr uint32_t TmaTransactionBytesV = CollectiveMainloopWS::TmaTransactionBytesV;

    // Softmax constants (same as SoftmaxFused)
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;
    static constexpr float fp4_scale_log2 = -2.584962500721156f;

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
    // mma_nonws_crosstile: Cross-tile double-buffer mainloop.
    //
    // tSrS_buf0/buf1 and AbsMaxP_buf0/buf1 are allocated in kernel entry
    // for register allocation visibility.
    // ========================================================================
    template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused,
              typename FrgTensorS, typename FrgTensorAMP,
              typename TKgK_t, typename TKsK_t,
              typename TKgSFK_t, typename TKsSFK_t,
              typename TDSgDS_t, typename TDSsDS_t,
              typename TVgVt_t, typename TVsVt_t,
              typename TVgSFVt_t, typename TVsSFVt_t>
    CUTLASS_DEVICE void
    mma_nonws_crosstile(
        Params const& mainloop_params,
        MainloopPipelineQ pipeline_q,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineStateQ& smem_pipe_read_q,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        PipelineState& smem_pipe_write_k,
        PipelineState& smem_pipe_write_v,
        FrgTensorO& tOrO_store,
        SoftmaxFused& softmax_fused,
        int n_block_count,
        int thread_idx,
        int m_block,
        SharedStorage& shared_storage,
        // Double-buffer score tensors (allocated in kernel entry)
        FrgTensorS& tSrS_buf0,
        FrgTensorS& tSrS_buf1,
        FrgTensorAMP& AbsMaxP_buf0,
        FrgTensorAMP& AbsMaxP_buf1,
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
        static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;

        bool const is_tma_thread = (thread_idx == 0);

        // ============ Smem tensors ============
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
        Tensor sSFQ_full = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

        constexpr int wg_id = 0;
        auto sQ_local = local_tile(sQ, make_shape(Int<kBlockMPerWG>{}, Int<kBlockK>{}), make_coord(wg_id, 0));

        // ============ MMA setup ============
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

        using TiledMmaQK_Full = typename Ktraits::TiledMmaQK_Full;
        TiledMmaQK_Full tiled_mma_qk_full;
        int consumer_thread_idx_full = thread_idx;
        auto thread_mma_qk_full = tiled_mma_qk_full.get_thread_slice(consumer_thread_idx_full);

        // ============ Fragment A/B from smem ============
        Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ_local);
        Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
        Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_, _, Int<0>{}));
        Tensor tOrP = make_tensor_like<Element>(LayoutP{});
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

        auto smem_tiled_copy_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFA_TV(tiled_mma_qk_full),
                                                        make_shape(size<0>(tile_shape_mnk_full), size<2>(tile_shape_mnk_full)));
        auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(consumer_thread_idx_full);
        Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ_full));
        Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

        auto smem_tiled_copy_SFK = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFB_TV(tiled_mma_qk),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(thread_idx);
        Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
        Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

        auto smem_tiled_copy_SFV = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        CollectiveMainloopWS().get_layoutSFB_TV(tiled_mma_pv),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(thread_idx);
        Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(as_position_independent_swizzle_tensor(sSFVt));
        Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

        // ============ Helper lambdas ============

        auto consumer_wait_fn = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int const unpadded_seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
        constexpr int wg_m_offset = 0;

        auto copy_k_block = [&](auto block_id) {
            auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
            auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());
            copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
        };

        auto copy_v_block = [&](auto block_id) {
            auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
            auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());
            copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
        };

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

        // Conversion-layout views for double-buffer score tensors
        Tensor tSrS_cv0 = make_tensor(tSrS_buf0.data(), sage::convert_to_conversion_layout(tSrS_buf0.layout()));
        Tensor tSrS_cv1 = make_tensor(tSrS_buf1.data(), sage::convert_to_conversion_layout(tSrS_buf1.layout()));

        // MmaN for N-sub-tile dispatch
        constexpr int MmaN_qk = decltype(size<2>(tSrS_buf0))::value;

        auto col_limit_causal = [&](int row, int n_block) {
            return row + wg_m_offset + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };

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

        // quantize: convert softmax output to FP4 + compute scale factors.
        // Parameterized on the conversion view and AbsMaxP buffer to support double-buffering.
        auto quantize_fn = [&](auto mma_k, auto& acc_conversion_view, auto& AbsMaxP_local) {
            Tensor AbsMaxP_stagek = AbsMaxP_local(_, make_coord(_, _, mma_k));
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

        // pv_gemm_fn: PV GEMM with interleaved V copy + quantize.
        auto pv_gemm_fn = [&](auto& tOrO_dst, auto& tSrS_cv_local, auto& AbsMaxP_local) {
            copy_v_block(_0{});
            quantize_fn(_0{}, tSrS_cv_local, AbsMaxP_local);

            CUTLASS_PRAGMA_UNROLL
            for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                cute::gemm(tiled_mma_pv,
                    make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                    make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                    tOrO_dst);
                if (v_block < size<2>(tOrP) - 1) {
                    copy_v_block(v_block + 1);
                    quantize_fn(v_block + 1, tSrS_cv_local, AbsMaxP_local);
                }
            }
        };

        // find_max_all_fn: find global row_max across all N-sub-tiles.
        // After this, softmax_fused.row_max contains the global max for
        // the tile, and AbsMaxP_local(mi, ni) contains per-sub-tile chunk max.
        auto find_max_all_fn = [&](auto& tSrS_local, auto& AbsMaxP_local) {
            Tensor acc_rv = make_tensor(
                tSrS_local.data(), sage::convert_to_reduction_layout(tSrS_local.layout()));

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_rv); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_rv); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                        local_max = fmaxf(local_max, acc_rv(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1);
                    AbsMaxP_local(mi, ni) = fmaxf(local_max, max_recv);
                    softmax_fused.row_max(mi) = fmaxf(softmax_fused.row_max(mi), AbsMaxP_local(mi, ni));
                }
                float max_recv = __shfl_xor_sync(int32_t(-1), softmax_fused.row_max(mi), 2);
                softmax_fused.row_max(mi) = fmaxf(softmax_fused.row_max(mi), max_recv);
            }
        };

        // exp2_sum_chunk_fn: exp2 + sum + AbsMaxP transform for one N-sub-tile.
        // Prerequisite: find_max_all_fn already called, softmax_fused.row_max is global max.
        // First chunk (n==0) of a non-first tile handles cross-block rescale.
        auto exp2_sum_chunk_fn = [&](auto& tSrS_local, auto& AbsMaxP_local,
                                     int n, bool is_first, auto const& prev_block_max) {
            Tensor acc_rv = make_tensor(
                tSrS_local.data(), sage::convert_to_reduction_layout(tSrS_local.layout()));

            float const scale_log2 = mainloop_params.softmax_scale_log2;

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_rv); mi++) {
                // Cross-block rescale on first chunk of non-first tile
                if (n == 0 && !is_first) {
                    float scores_max_cur = softmax_fused.row_max(mi);
                    softmax_fused.scores_scale(mi) = sage::ptx_exp2(
                        (prev_block_max(mi) - scores_max_cur) * scale_log2);
                    softmax_fused.row_sum(mi) *= softmax_fused.scores_scale(mi);
                }

                const float max_scaled = softmax_fused.row_max(mi) * scale_log2
                                         + fp8_scalexfp4_scale_log2;

                // exp2 for chunk n elements + accumulate to row_sum
                CUTLASS_PRAGMA_UNROLL
                for (int ei = 0; ei < size<1, 0>(acc_rv); ei++) {
                    float val = sage::ptx_exp2(
                        acc_rv(mi, make_coord(ei, n)) * scale_log2 - max_scaled);
                    acc_rv(mi, make_coord(ei, n)) = val;
                    softmax_fused.row_sum(mi) += val;
                }

                // AbsMaxP transform for chunk n
                AbsMaxP_local(mi, n) = sage::ptx_exp2(
                    AbsMaxP_local(mi, n) * scale_log2 - max_scaled + fp4_scale_log2);
            }
        };

        // ====================================================================
        // TMA helpers
        // ====================================================================
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
        // Double-buffer management via pointer arrays
        // ====================================================================
        FrgTensorS*   tSrS_ptrs[2]    = { &tSrS_buf0, &tSrS_buf1 };
        FrgTensorAMP* AbsMaxP_ptrs[2] = { &AbsMaxP_buf0, &AbsMaxP_buf1 };
        int cur = 0, nxt = 1;

        // ====================================================================
        // PROLOGUE: Wait for Q, copy to registers.
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
        // PROLOGUE: QK GEMM tile 0 -> tSrS[cur]
        // ====================================================================
        {
            int n_block_0 = n_block_count - 1;
            int prefetch_n_block_0 = n_block_count - 1 - kStagesLocal;
            bool has_prefetch_0 = kStagesLocal < n_block_count;

            // Wait K[0], copy to regs
            consumer_wait_fn(pipeline_k, smem_pipe_read_k);
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                copy_k_block(k_block);
            }
            add_delta_s(*tSrS_ptrs[cur]);

            // Release K stage
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;

            // Refill next K stage only. V is not consumed in prologue
            // (PV GEMM happens in steady-state for tile i-1), so V refill
            // is handled differently: see steady-state V refill below.
            if (has_prefetch_0 && is_tma_thread) {
                issue_tma_k(prefetch_n_block_0);
            }

            // QK GEMM tile 0: K-outer dispatch (no interleaving for prologue)
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk,
                    make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                    *tSrS_ptrs[cur]);
            }

            // Mask tile 0
            apply_mask(*tSrS_ptrs[cur], n_block_0);

            // Initialize softmax state for first tile
            fill(softmax_fused.row_max, -INFINITY);
            clear(softmax_fused.row_sum);
            fill(softmax_fused.scores_scale, 1.f);

            // find_max_all for tile 0 (prepares global max for interleaving in tile 1)
            find_max_all_fn(*tSrS_ptrs[cur], *AbsMaxP_ptrs[cur]);
        }

        // ====================================================================
        // Special case: n_block_count == 1 (no cross-tile, degenerate to baseline)
        // ====================================================================
        if (n_block_count == 1) {
            // find_max already done. Do exp2_sum + quantize (all serial, is_first=true).
            {
                Tensor acc_rv = make_tensor(
                    tSrS_ptrs[cur]->data(), sage::convert_to_reduction_layout(tSrS_ptrs[cur]->layout()));

                float const scale_log2 = mainloop_params.softmax_scale_log2;

                CUTLASS_PRAGMA_UNROLL
                for (int mi = 0; mi < size<0>(acc_rv); mi++) {
                    const float max_scaled = softmax_fused.row_max(mi) * scale_log2
                                             + fp8_scalexfp4_scale_log2;

                    CUTLASS_PRAGMA_UNROLL
                    for (int ni = 0; ni < size<1>(acc_rv); ni++) {
                        float val = sage::ptx_exp2(acc_rv(mi, ni) * scale_log2 - max_scaled);
                        acc_rv(mi, ni) = val;
                        softmax_fused.row_sum(mi) += val;
                    }

                    CUTLASS_PRAGMA_UNROLL
                    for (int sfi = 0; sfi < size<1>(*AbsMaxP_ptrs[cur]); sfi++) {
                        (*AbsMaxP_ptrs[cur])(mi, sfi) = sage::ptx_exp2(
                            (*AbsMaxP_ptrs[cur])(mi, sfi) * scale_log2 - max_scaled + fp4_scale_log2);
                    }
                }

                softmax_fused.quantize_after_partial_softmax(*tSrS_ptrs[cur], *AbsMaxP_ptrs[cur]);
            }

            // PV GEMM
            consumer_wait_fn(pipeline_v, smem_pipe_read_v);
            pv_gemm_fn(tOrO_store, (cur == 0) ? tSrS_cv0 : tSrS_cv1, *AbsMaxP_ptrs[cur]);
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;

            softmax_fused.finalize(tOrO_store);
            return;
        }

        // ====================================================================
        // STEADY-STATE LOOP (tile i = 1..N-1)
        //
        // Cross-block rescale correctness:
        //   exp2_sum_chunk for tile j (processed in iteration j+1) needs:
        //     prev_row_max = row_max BEFORE find_max_all(tile j) was called
        //   After find_max_all(tile j), row_max includes tile j's max.
        //   The exp2_sum uses this UPDATED row_max for exp2 (correct: global max).
        //   But cross-block rescale needs the PREVIOUS max to rescale old row_sum.
        //
        //   Solution: carry forward a `prev_block_max_saved` from before find_max_all.
        //   - After prologue find_max_all(tile 0): save row_max_before = -INF (init value)
        //   - This -INF is the "prev max" for tile 0's exp2_sum (is_first=true, skip rescale)
        //   - After find_max_all(tile i): save row_max_before = row_max before find_max
        //   - In iteration i+1: use saved value as prev_block_max for tile i's exp2_sum
        // ====================================================================
        bool is_first_pv = true;

        // prev_block_max_saved: row_max from BEFORE the most recent find_max_all.
        // For tile 0's exp2_sum (in tile_idx=1), this is the init value (-INF).
        // We never actually use it because is_first_pv=true skips rescale.
        // For tile j's exp2_sum (in tile_idx=j+1), this is row_max before find_max_all(tile j).
        auto prev_block_max_saved = make_fragment_like(softmax_fused.row_max);
        // The prologue's find_max_all(tile 0) was called when row_max was -INF.
        // After it, row_max = max(tile 0). The "before" value is -INF.
        fill(prev_block_max_saved, -INFINITY);

        #pragma unroll 1
        for (int tile_idx = 1; tile_idx < n_block_count; ++tile_idx) {
            int n_block = n_block_count - 1 - tile_idx;

            int prefetch_tile_idx = tile_idx + kStagesLocal;
            int prefetch_n_block = n_block_count - 1 - prefetch_tile_idx;
            bool has_prefetch = prefetch_tile_idx < n_block_count;

            // ================================================================
            // Phase A: QK GEMM[i] interleaved with exp2_sum[i-1]
            // ================================================================

            // Wait K[i], copy to regs
            consumer_wait_fn(pipeline_k, smem_pipe_read_k);
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                copy_k_block(k_block);
            }
            add_delta_s(*tSrS_ptrs[nxt]);

            // Release K stage
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;

            // Refill next K stage
            if (has_prefetch && is_tma_thread) {
                issue_tma_k(prefetch_n_block);
            }

            // N-sub-tile interleaved: QK GEMM[i] + exp2_sum_chunk[i-1]
            //
            // tSrS[cur] has complete scores from previous tile with global max known.
            // exp2_sum_chunk reads tSrS[cur], QK GEMM writes tSrS[nxt] -- independent.
            //
            // prev_block_max_saved: row_max from before the most recent find_max_all.
            // This is the correct "previous max" for cross-block rescale.
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaN_qk; ++n) {
                // QK GEMM sub-tile n: N-outer, K-inner dispatch
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, n),
                        (*tSrS_ptrs[nxt])(_, _0{}, n));
                }

                // exp2_sum_chunk for tSrS[cur] chunk n, hidden in MMA pipeline latency
                exp2_sum_chunk_fn(*tSrS_ptrs[cur], *AbsMaxP_ptrs[cur],
                                  n, is_first_pv, prev_block_max_saved);
            }

            // Mask new tile's scores
            apply_mask(*tSrS_ptrs[nxt], n_block);

            // Quantize old tile's softmax output (all exp2_sum done, all AbsMaxP ready)
            softmax_fused.quantize_after_partial_softmax(*tSrS_ptrs[cur], *AbsMaxP_ptrs[cur]);

            // ================================================================
            // Phase B: PV GEMM[i-1]
            // ================================================================

            consumer_wait_fn(pipeline_v, smem_pipe_read_v);

            if (is_first_pv) {
                // First PV: accumulate directly into tOrO_store
                pv_gemm_fn(tOrO_store,
                            (cur == 0) ? tSrS_cv0 : tSrS_cv1,
                            *AbsMaxP_ptrs[cur]);
                is_first_pv = false;
            } else {
                // Subsequent PV: accumulate to temp, then rescale + add
                Tensor tOrO_tmp = make_fragment_like(tOrO_store);
                pv_gemm_fn(tOrO_tmp,
                            (cur == 0) ? tSrS_cv0 : tSrS_cv1,
                            *AbsMaxP_ptrs[cur]);
                softmax_fused.rescale_o(tOrO_store, tOrO_tmp);
            }

            // Release V stage, then refill
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;

            // V refill: V is consumed 1 tile later than K (PV uses tile i-1's V).
            // So V needs refill 1 tile earlier than K.
            // V prefetch index = (tile_idx - 1) + kStages = tile_idx + kStages - 1
            {
                int v_prefetch_tile_idx = tile_idx + kStagesLocal - 1;
                int v_prefetch_n_block = n_block_count - 1 - v_prefetch_tile_idx;
                bool v_has_prefetch = v_prefetch_tile_idx < n_block_count;
                if (v_has_prefetch && is_tma_thread) {
                    issue_tma_v(v_prefetch_n_block);
                }
            }

            // ================================================================
            // Prepare next iteration: find_max_all on tSrS[nxt]
            //
            // Save row_max BEFORE find_max_all so the next iteration can use
            // it as prev_block_max for cross-block rescale.
            // Skip find_max on last iteration: epilogue uses chunked_softmax_fixed.
            // ================================================================
            if (tile_idx < n_block_count - 1) {
                // Save current row_max (state AFTER processing tile i-1's softmax,
                // BEFORE incorporating tile i's max)
                cute::copy(softmax_fused.row_max, prev_block_max_saved);
                find_max_all_fn(*tSrS_ptrs[nxt], *AbsMaxP_ptrs[nxt]);
            }

            // Swap buffers
            cur ^= 1;
            nxt ^= 1;

        }  // end steady-state loop

        // ====================================================================
        // EPILOGUE: softmax + PV for last tile
        //
        // After the last steady-state iteration:
        //   - cur points to the last tile's tSrS (swapped from nxt)
        //   - find_max_all was SKIPPED for the last tile
        //   - row_max reflects all tiles EXCEPT the last one's find_max
        //   - tSrS[cur] has masked scores, no softmax applied yet
        //
        // Use chunked_softmax_fixed which does everything in one call:
        //   find_max + cross-block rescale + exp2 + sum + AbsMaxP transform + quantize
        // ====================================================================
        {
            auto prev_block_max = make_fragment_like(softmax_fused.row_max);
            cute::copy(softmax_fused.row_max, prev_block_max);

            softmax_fused.template chunked_softmax_fixed<>(
                *tSrS_ptrs[cur], *AbsMaxP_ptrs[cur], false,
                mainloop_params.softmax_scale_log2, prev_block_max);

            // PV GEMM for last tile
            consumer_wait_fn(pipeline_v, smem_pipe_read_v);

            {
                Tensor tOrO_tmp = make_fragment_like(tOrO_store);
                pv_gemm_fn(tOrO_tmp,
                            (cur == 0) ? tSrS_cv0 : tSrS_cv1,
                            *AbsMaxP_ptrs[cur]);
                softmax_fused.rescale_o(tOrO_store, tOrO_tmp);
            }

            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;
        }

        // ====================================================================
        // FINALIZE
        // ====================================================================
        softmax_fused.finalize(tOrO_store);
    }
};

}  // namespace sage
