// PINGPONG_MATH_ORDER and PINGPONG_EARLY_RELEASE_K now controlled via setup.py.
// run_in_docker.sh defaults both to 1.
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
#if defined(MMA_SOFTMAX_INTERLEAVE) && defined(R22_MODE) && (R22_MODE == 3)
#include "../common/gemm_with_interleave.h"
#endif

#if defined(STEADY_SOFT_CREDIT_GATE) && !defined(STEADY_SOFT_CREDIT_K_BLOCKS)
#define STEADY_SOFT_CREDIT_K_BLOCKS 1
#endif
#if !defined(R19_QK_LEAD_MODE)
#define R19_QK_LEAD_MODE 0
#endif
#if defined(R19_PV_SKEW_CYCLES) && !defined(R19_PV_SKEW_WG)
#define R19_PV_SKEW_WG 1
#endif
namespace sage {

using namespace cute;

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using ElementSF = typename Ktraits::ElementSF;
    // using TMAElement = Element;
    // using TMAElementSF = typename Ktraits::ElementSF;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr int BlockMean = Ktraits::BlockMean;
    using GmemTiledCopy = typename Ktraits::GmemTiledCopy;
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
    using SmemLayoutDS = typename Ktraits::SmemLayoutDS;
    using SmemLayoutAtomDS = typename Ktraits::SmemLayoutAtomDS;
    using LayoutDS = decltype(
        blocked_product(
            SmemLayoutAtomDS{},
            make_layout(
            make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
            make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))
        )
        );
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeSF = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d // 16, head, batch)
    using LayoutSF = typename Ktraits::LayoutSF;
    using LayoutP = typename Ktraits::LayoutP;
    using LayoutSFP = typename Ktraits::LayoutSFP;
    using SfAtom = typename Ktraits::SfAtom;
    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));

    using TMA_KV = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        _1{}));

    using TMA_Vt = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
        take<0, 2>(SmemLayoutVt{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));

    using TMA_DS = decltype(make_tma_copy(
        GmemTiledCopy{},
        make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), LayoutDS{}),
        take<0, 2>(SmemLayoutDS{}),
        make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));

    using BlkScaledConfig = typename Ktraits::BlkScaledConfig;
    using GmemTiledCopySF = typename Ktraits::GmemTiledCopySF;
    using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
    using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
    using SmemLayoutSFV = typename Ktraits::SmemLayoutSFV;
    using SmemLayoutSFVt = typename Ktraits::SmemLayoutSFVt;

    using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySF{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
        SmemLayoutSFQ{},
        make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
        _1{}));  // No programmatic multicast


    using TMA_SFKV = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySF{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
        SmemLayoutSFK{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
        _1{}));

    using TMA_SFVt = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySF{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
        SmemLayoutSFVt{}(_,_,cute::Int<0>{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
        _1{}));

    using SmemCopyAtomQ = typename Ktraits::SmemCopyAtomQ;
    using SmemCopyAtomKV = typename Ktraits::SmemCopyAtomKV;
    using SmemCopyAtomSF = typename Ktraits::SmemCopyAtomSF;
    using TiledMmaQK = typename Ktraits::TiledMmaQK;
    using TiledMmaPV = typename Ktraits::TiledMmaPV;
    static constexpr int NumMmaThreads = size(TiledMmaQK{});
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
    using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
    using PipelineStateQ = typename Ktraits::PipelineStateQ;
    using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize((SmemLayoutSFQ{})) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(size((SmemLayoutQ{})) * sizeof_bits<Element>::value));

    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFK{})) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutDS{})) * cute::sizeof_bits_v<float>) +
        cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutK{})) * sizeof_bits<Element>::value));

    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFVt{})) * cute::sizeof_bits_v<ElementSF>) +
        cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutVt{})) * sizeof_bits<Element>::value));

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        ShapeQKV const unpadded_shape_K;
        Element const* ptr_Vt;
        ShapeQKV const shape_Vt;
        StrideQKV const stride_Vt;
        ElementSF const* ptr_SFQ{nullptr};
        ShapeSF const shape_SFQ{};
        ElementSF const* ptr_SFK{nullptr};
        ShapeSF const shape_SFK{};
        ElementSF const* ptr_SFVt{nullptr};
        ShapeSF const shape_SFVt{};
        float const* ptr_ds;
        ShapeQKV const shape_ds;
        StrideQKV const stride_ds;
        float const softmax_scale_log2;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        LayoutSF const layout_SFQ;
        ShapeQKV const shape_K;
        ShapeQKV const unpadded_shape_K;
        LayoutSF const layout_SFK;
        ShapeQKV const shape_Vt;
        LayoutSF const layout_SFVt;
        LayoutDS const layout_DS;
        TMA_Q tma_load_Q;
        TMA_SFQ tma_load_SFQ;
        TMA_KV tma_load_K;
        TMA_SFKV tma_load_SFK;
        TMA_Vt tma_load_Vt;
        TMA_SFVt tma_load_SFVt;
        TMA_DS tma_load_DS;
        float const softmax_scale_log2;
    };


    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopy{},
            mQ,
            SmemLayoutQ{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_KV tma_load_K = make_tma_copy(
            GmemTiledCopy{},
            mK,
            SmemLayoutK{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            _1{}); // mcast along M mode for this N load, if any
        Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_Vt), args.shape_Vt, args.stride_Vt);
        TMA_Vt tma_load_Vt = make_tma_copy(
            GmemTiledCopy{},
            mVt,
            SmemLayoutVt{}(_, _, _0{}),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{}); // mcast along M mode for this N load, if any
        auto [Seqlen_Q, Seqlen_K, HeadNum, Batch] = args.shape_ds;
        LayoutDS layout_ds = tile_to_shape(SmemLayoutAtomDS{}, make_shape(Seqlen_Q, Seqlen_K, HeadNum, Batch), Step<_2,_1,_3,_4>{});
        Tensor mDS = make_tensor(make_gmem_ptr(args.ptr_ds), layout_ds);
        TMA_DS tma_load_ds = make_tma_copy (
            GmemTiledCopy{},
            mDS,
            SmemLayoutDS{}(_, _, _0{}),
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{});
        LayoutSF layout_sfq = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFQ);
        Tensor mSFQ = make_tensor(make_gmem_ptr(args.ptr_SFQ), layout_sfq);
        TMA_SFQ tma_load_sfq = make_tma_copy<uint16_t>(
            GmemTiledCopySF{},
            mSFQ,
            SmemLayoutSFQ{},
            make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{});
        LayoutSF layout_sfk = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFK);
        Tensor mSFK = make_tensor(make_gmem_ptr(args.ptr_SFK), layout_sfk);
        TMA_SFKV tma_load_sfk = make_tma_copy<uint16_t>(
            GmemTiledCopySF{},
            mSFK,
            SmemLayoutSFK{}(_, _, _0{}),
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})),
            _1{});
        LayoutSF layout_sfvt = BlkScaledConfig::tile_atom_to_shape_SFVt(args.shape_SFVt);
        Tensor mSFVt = make_tensor(make_gmem_ptr(args.ptr_SFVt), layout_sfvt);
        TMA_SFVt tma_load_sfvt = make_tma_copy<uint16_t>(
            GmemTiledCopySF{},
            mSFVt,
            SmemLayoutSFVt{}(_, _, _0{}),
            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
            _1{});
        return {args.shape_Q, layout_sfq,
                args.shape_K, args.unpadded_shape_K, layout_sfk,
                args.shape_Vt, layout_sfvt,
                layout_ds,
                tma_load_Q, tma_load_sfq,
                tma_load_K, tma_load_sfk,
                tma_load_Vt, tma_load_sfvt,
                tma_load_ds,
                args.softmax_scale_log2};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Vt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFVt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_DS.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_n_block_max(Params const& mainloop_params, int m_block) {
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

    template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE constexpr
    auto
    thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
      CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

      using AtomShape_MNK  = typename Atom::Shape_MNK;
      using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

      auto permutation_mnk = TiledPerm{};
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

      // Reorder the tensor for the TiledAtom
      auto t_tile = make_tile(get<0>(permutation_mnk),
                              get<2>(permutation_mnk));
      auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

      // Tile the tensor for the Atom
      auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                              make_layout(size<2>(AtomShape_MNK{})));
      auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

      // Transform the Atom mode from (M,K) to (Thr,Val)
      auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

      // Tile the tensor for the Thread
      auto thr_tile = make_tile(_,
                                make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                          make_layout(size<3>(thr_layout_vmnk))));
      auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

      return thr_tensor;
    }

    template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE constexpr
    auto
    thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
      CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

      using AtomShape_MNK  = typename Atom::Shape_MNK;
      using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

      auto permutation_mnk = TiledPerm{};
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

      // Reorder the tensor for the TiledAtom
      auto t_tile = make_tile(get<1>(permutation_mnk),
                              get<2>(permutation_mnk));
      auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

      // Tile the tensor for the Atom
      auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                              make_layout(size<2>(AtomShape_MNK{})));
      auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

      // Transform the Atom mode from (M,K) to (Thr,Val)
      auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

      // Tile the tensor for the Thread
      auto thr_tile = make_tile(_,
                                make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                          make_layout(size<3>(thr_layout_vmnk))));
      auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
      return thr_tensor;
    }

    template <class SFATensor, class ThrMma>
    CUTE_HOST_DEVICE constexpr
    auto
    partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
    {
      using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
      auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
      auto thr_vmnk = thread_mma.thr_vmnk_;
      auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
      auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
      return make_fragment_like<ValTypeSF>(partition_SFA);
    }

    template <class SFBTensor, class ThrMma>
    CUTE_HOST_DEVICE constexpr
    auto
    partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
    {
      using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
      auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
      auto thr_vmnk = thread_mma.thr_vmnk_;
      auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
      auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
      return make_fragment_like<ValTypeSF>(partition_SFB);
    }

    template<class TiledMma>
    CUTE_HOST_DEVICE constexpr
    auto
    get_layoutSFA_TV(TiledMma& mma)
    {
      // (M,K) -> (M,K)
      auto tile_shape_mnk = tile_shape(mma);
      auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

      // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
      auto atile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                  make_stride(               Int<1>{} ,                Int<0>{} )),
                                      _));

      // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
      auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
      // (thr_idx,val) -> (M,K)
      return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
    }

    template<class TiledMma>
    CUTE_HOST_DEVICE constexpr
    auto
    get_layoutSFB_TV(TiledMma& mma)
    {
      // (N,K) -> (N,K)
      auto tile_shape_mnk = tile_shape(mma);
      auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

      // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
      auto btile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                  make_stride(               Int<0>{} ,                Int<1>{} )),
                                      _));

      // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
      auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
      // (thr_idx,val) -> (M,K)
      return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
    }

    template <typename SchedulerParams, typename SharedStorage, typename WorkTileInfo>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         SchedulerParams const& scheduler_params,
         MainloopPipelineQ pipeline_q,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineStateQ& smem_pipe_write_q,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage &shared_storage,
         WorkTileInfo work_tile_info,
         int& work_idx,
         int& tile_count_semaphore
         ) {

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);

        int n_block_max = get_n_block_max(mainloop_params, m_block);

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
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVt = local_tile(mVt(_, _, bidh, bidb), make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));  // (N, K, _)
        Tensor gDS = [&] {
                        if constexpr (BlockMean) {
                            return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}), make_coord(m_block, _));
                        } else {
                            return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}), make_coord(_0{}, _));
                        }
                    }();
        Tensor gSFQ = local_tile(mSFQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gSFK = local_tile(mSFK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gSFVt = local_tile(mSFVt(_, _, bidh, bidb), make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), make_coord(_0{}, _));
        auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = block_tma_q.partition_S(gQ);
        Tensor tQsQ = block_tma_q.partition_D(sQ);
        auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});
        Tensor tQgSFQ = block_tma_sfq.partition_S(gSFQ);
        Tensor tQsSFQ = block_tma_sfq.partition_D(sSFQ);
        auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));
        auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
        Tensor tKgSFK = group_modes<0, 3>(block_tma_sfk.partition_S(gSFK));
        Tensor tKsSFK = group_modes<0, 3>(block_tma_sfk.partition_D(sSFK));
        auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
        Tensor tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));
        Tensor tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));
        auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
        Tensor tVgSFVt = group_modes<0, 3>(block_tma_sfvt.partition_S(gSFVt));
        Tensor tVsSFVt = group_modes<0, 3>(block_tma_sfvt.partition_D(sSFVt));
        auto block_tma_ds = mainloop_params.tma_load_DS.get_slice(cluster_local_block_id.x);
        Tensor tDSgDS = group_modes<0, 3>(block_tma_ds.partition_S(gDS));
        Tensor tDSsDS = group_modes<0, 3>(block_tma_ds.partition_D(sDS));
        uint16_t mcast_mask_kv = 0;

        int n_block = n_block_max - 1;
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
        pipeline_q.producer_acquire(smem_pipe_write_q);
        copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgQ, tQsQ);
        copy(mainloop_params.tma_load_SFQ.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0), tQgSFQ, tQsSFQ);
        ++smem_pipe_write_q;
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
            tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
            tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
            tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
            tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
        copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
            tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
        }

        n_block--;
        if (lane_predicate) {
            // CUTLASS_PRAGMA_NO_UNROLL
            #pragma unroll 2
            for (; n_block >= 0; --n_block) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
                copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
                copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
                copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
        ++work_idx;
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineQ pipeline_q,
              MainloopPipeline pipeline_k,
              MainloopPipeline pipeline_v,
              PipelineStateQ& smem_pipe_write_q,
              PipelineState& smem_pipe_write_k,
              PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
          pipeline_q.producer_tail(smem_pipe_write_q);
          pipeline_k.producer_tail(smem_pipe_write_k);
          pipeline_v.producer_tail(smem_pipe_write_v);
        }
    }

    // ============================================================
    // R15: FA3-style True Ping-Pong MMA (Scheme A: shared pipeline)
    // Both WGs iterate ALL tiles. Only the "owning" WG computes.
    // WG1 (wg_id=0, M-rows 0-63):   computes on even tiles (N-1, N-3, ...)
    // WG2 (wg_id=1, M-rows 64-127): computes on odd tiles  (N-2, N-4, ...)
    // Pipeline protocol unchanged: num_consumers=256, advance by 1.
    // ============================================================
    // Default no-op refill for WS (producer runs separately)
    struct NoOpRefill {
        CUTLASS_DEVICE void refill_k(int) {}
        CUTLASS_DEVICE void refill_v(int) {}
    };

    template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused,
              typename MathOrderBarrier, typename TmaRefill = NoOpRefill>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipelineQ pipeline_q,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineStateQ& smem_pipe_read_q,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO_store,
        SoftmaxFused& softmax_fused,
        int n_block_count,       // total N-blocks
        int thread_idx,          // 0-127 (per-WG MMA thread)
        int work_idx,
        int m_block,
        int wg_id,               // 0 or 1
        SharedStorage& shared_storage,
        MathOrderBarrier& math_order,
        MathOrderBarrier& math_order_pv,
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
        int& strict_phase_wait_phase,
#endif
        TmaRefill tma_refill = {}
        ) {

        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockK = get<2>(TileShape_MNK{});
        static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;  // 64 (WS) or 128 (Non-WS)

        // ============ Smem tensors ============
        Tensor sQ_full = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
        Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
        Tensor sSFQ_full = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
        Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
        Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

        // ============ Per-WG Q slice (64 M-rows) ============
        auto sQ = local_tile(sQ_full, make_shape(Int<kBlockMPerWG>{}, Int<kBlockK>{}), make_coord(wg_id, 0));

        // ============ MMA setup ============
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

        // 8-atom MMA (256 threads) for SFQ partitioning only.
        // consumer_thread_idx: WG1=0-127 → atoms 0-3 (rows 0-63),
        //                      WG2=128-255 → atoms 4-7 (rows 64-127).
        using TiledMmaQK_Full = typename Ktraits::TiledMmaQK_Full;
        TiledMmaQK_Full tiled_mma_qk_full;
        int consumer_thread_idx_full = thread_idx + wg_id * NumMmaThreads;
        auto thread_mma_qk_full = tiled_mma_qk_full.get_thread_slice(consumer_thread_idx_full);

        // Fragment A from WG's Q half (64 M-rows)
        Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
        Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_,_,Int<0>{}));
        Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_,_,Int<0>{}));
        Tensor tOrP = make_tensor_like<Element>(LayoutP{});
        // SFQ uses 8-atom MMA so each WG gets SF for its own 64 M-rows
        Tensor tSrSFQ = partition_fragment_SFA(sSFQ_full, thread_mma_qk_full);
        Tensor tSrSFK = partition_fragment_SFB(sSFK(_,_,Int<0>{}), thread_mma_qk);
        Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_,_,Int<0>{}), thread_mma_pv);
        Tensor tOrSFP = make_tensor<ElementSF>(LayoutSFP{});
        Tensor tOrSFP_flt = filter_zeros(tOrSFP);

        // ============ Smem copy setup ============
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
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
        // SFQ smem copy: use 8-atom MMA layout + consumer_thread_idx_full
        auto tile_shape_mnk_full = tile_shape(tiled_mma_qk_full);
        auto smem_tiled_copy_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        get_layoutSFA_TV(tiled_mma_qk_full),
                                                        make_shape(size<0>(tile_shape_mnk_full), size<2>(tile_shape_mnk_full)));
        auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(consumer_thread_idx_full);
        Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ_full));
        Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

        auto smem_tiled_copy_SFK = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        get_layoutSFB_TV(tiled_mma_qk),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(thread_idx);
        Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
        Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

        auto smem_tiled_copy_SFV = make_tiled_copy_impl(SmemCopyAtomSF{},
                                                        get_layoutSFB_TV(tiled_mma_pv),
                                                        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
        auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(thread_idx);
        Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(as_position_independent_swizzle_tensor(sSFVt));
        Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

        // ============ Helpers ============
        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int const seqlen_q = get<0>(mainloop_params.shape_Q);
        int const seqlen_k = get<0>(mainloop_params.shape_K);
        int const unpadded_seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
        int const wg_m_offset = wg_id * kBlockMPerWG;  // 0 or 64

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

        // S accumulator for 64 M-rows × kBlockN (declared here so lambdas can capture)
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
        Tensor tSrS_converion_view = make_tensor(tSrS.data(), sage::convert_to_conversion_layout(tSrS.layout()));
        Tensor AbsMaxP = make_tensor_like<float>(
            make_layout(shape(group<1, 4>(flatten(tSrS_converion_view.layout()(make_coord(_0{}, _), _, _))))) );

#if defined(CROSS_TILE_DOUBLE_BUF)
        // Second score buffer for cross-tile pipelining: softmax reads tSrS[buf]
        // while next tile's QK GEMM writes tSrS[1-buf]. +64 regs per thread.
        Tensor tSrS_buf1 = partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
        Tensor tSrS_buf1_cv = make_tensor(tSrS_buf1.data(), sage::convert_to_conversion_layout(tSrS_buf1.layout()));
        Tensor AbsMaxP_buf1 = make_tensor_like<float>(
            make_layout(shape(group<1, 4>(flatten(tSrS_buf1_cv.layout()(make_coord(_0{}, _), _, _))))) );
        // buf=0: tSrS/AbsMaxP, buf=1: tSrS_buf1/AbsMaxP_buf1
        int cross_tile_buf = 0;
#endif

        // Causal mask boundary: row is local (0-63), add wg_m_offset for global position
        auto col_limit_causal = [&](int row, int n_block) {
            return row + wg_m_offset + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };

        // Masking lambda (applies seqlen + causal mask)
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

        // ============ Load Q (both WGs, each reads its 64-row half) ============
        consumer_wait(pipeline_q, smem_pipe_read_q);
        copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);
        pipeline_q.consumer_release(smem_pipe_read_q);
        ++smem_pipe_read_q;

        bool is_first_compute = true;

#if defined(R18_PINGPONG_OVERLAP)
        // ============================================================
        // R18: 4-Step Precise Ping-Pong GEMM-Softmax Overlap
        //
        // Each tile has 4 steps. At every step, one WG does GEMM (OMMA)
        // while the other does softmax (MUFU). Zero OMMA contention.
        //
        // Step 0: WG0=QK_i(OMMA)       WG1=soft_{i-1}_2nd+quant(MUFU)
        // Step 1: WG0=soft_i_1st(MUFU)  WG1=PV_{i-1}(OMMA)
        // Step 2: WG0=soft_i_2nd+quant  WG1=QK_i(OMMA)
        // Step 3: WG0=PV_i(OMMA)        WG1=soft_i_1st(MUFU)
        //
        // Consumer-only barrier (256 threads, PTX bar.sync).
        // Per-WG pipeline read pointers (方案 2).
        // ============================================================

        // R18 sync: hardware mbarrier for OMMA handoff.
        //
        // r18_mbar[0]: WG0→WG1. WG0 leader arrives after QK. WG1 all threads wait.
        // r18_mbar[1]: WG1→WG0. WG1 leader arrives after QK. WG0 all threads wait.
        //
        // Uses mbarrier.try_wait.parity (hardware-backed deschedule).
        // Waiting threads are desceduled by hardware — no warp scheduler starvation.
        // Phase alternates each tile: 0, 1, 0, 1, ...
        // Phase persists across blocks via shared_storage.r18_wait_phase[].
        auto* r18_mbar = shared_storage.r18_mbar;
        int r18_phase[2] = {shared_storage.r18_wait_phase[0],
                            shared_storage.r18_wait_phase[1]};

        // R18 sync: NO explicit synchronization needed!
        //
        // Timing analysis proves natural OMMA exclusion:
        //   softmax = ~1500cy > GEMM = ~750cy, so the softmax WG can NEVER
        //   catch up to the GEMM WG. Each step has exactly one OMMA user:
        //     Step 0: WG0=QK(OMMA)  WG1=soft(MUFU)  — no contention
        //     Step 1: WG1=PV(OMMA)  WG0=soft(MUFU)  — no contention
        //     Step 2: WG1=QK(OMMA)  WG0=soft(MUFU)  — no contention
        //     Step 3: WG0=PV(OMMA)  WG1=soft(MUFU)  — no contention
        //
        //   At transitions (e.g. Step 0→1): WG0 finishes QK at ~750cy,
        //   WG1 finishes soft_prev at ~750cy, then WG1 needs V pipeline
        //   wait + copy (~100cy) before PV OMMA issue at ~850cy.
        //   WG0's QK finishes at ~800cy (750 + drain). 50cy margin.
        //
        // Previous attempts with mbarrier/bar.sync killed tensor core
        // pipeline by forcing full-completion waits (ncu: 43%→14% TC util).
        // Issue-ordering (like math_order) adds ~20cy but gains nothing
        // here since natural timing already provides ordering.
        //
        // Signal/wait are now no-ops (kept as markers for readability):
        auto r18_signal_partner = [&](int) { /* natural timing */ };
        auto r18_wait_partner = [&](int) { /* natural timing */ };

        // Per-WG pipeline read state (both start at same initial position)
        // WG0 advances K at Step 0, V at Step 3
        // WG1 advances K at Step 2, V at Step 1 (V is 1 tile behind WG0)
        auto my_pipe_k = smem_pipe_read_k;
        auto my_pipe_v = smem_pipe_read_v;

        // Per-WG copy lambdas (use my_pipe_k/v instead of shared pipe state)
        auto r18_copy_k_block = [&](auto block_id) {
            auto tSsK_stage = tSsK(_, _, _, my_pipe_k.index());
            auto tSsSFK_stage = tSsSFK(_, _, _, my_pipe_k.index());
            copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
        };
        auto r18_copy_v_block = [&](auto block_id) {
            auto tOsVt_stage = tOsVt(_, _, _, my_pipe_v.index());
            auto tOsSFVt_stage = tOsSFVt(_, _, _, my_pipe_v.index());
            copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
            copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
        };
        auto r18_add_delta_s = [&](auto& acc) {
            auto tSsDS_stage = recast<float4>(sDS(_, _, my_pipe_k.index()));
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

        // Reuse tSrS (line 730) and AbsMaxP (line 732) declared at function scope.
        auto& tSrS_local = tSrS;
        Tensor tSrS_local_cv = make_tensor(tSrS_local.data(), sage::convert_to_conversion_layout(tSrS_local.layout()));
        constexpr int MmaN_qk = decltype(size<2>(tSrS_local))::value;

        // Save prev_block_max for non-first N-block rescale
        auto prev_block_max = make_fragment_like(softmax_fused.row_max);

        #pragma unroll 1
        for (int tile_idx = 0; tile_idx < n_block_count; ++tile_idx) {
            int n_block = n_block_count - 1 - tile_idx;
            bool is_first_tile = (tile_idx == 0);

#ifdef R18_DEBUG
            if (thread_idx == 0 && tile_idx == 0 && blockIdx.x == 0) {
                printf("R18 START wg=%d work=%d n_blocks=%d K_stage=%d K_phase=%d V_stage=%d V_phase=%d\n",
                       wg_id, work_idx, n_block_count,
                       my_pipe_k.index(), my_pipe_k.phase(), my_pipe_v.index(), my_pipe_v.phase());
            }
#endif

            // ============================================================
            // R18 v8: 3-step design — every step has OMMA, no TC gap.
            //
            // WG0: StepA=QK_i         StepB=soft_full+quant   StepC=PV_i
            // WG1: StepA=PV_{i-1}     StepB=QK_i+soft_1st     StepC=soft_2nd+quant
            //
            // TC: StepA=[QK_WG0+PV_WG1 concurrent] StepB=[QK_WG1] StepC=[PV_WG0]
            // No 0-OMMA step! TC always has work.
            // ============================================================

            // ============ Step A: WG0=QK_i(OMMA), WG1=PV_{i-1}(OMMA) ← CONCURRENT! ============
            if (wg_id == 0) {
                // --- WG0: QK GEMM for tile i ---
                consumer_wait(pipeline_k, my_pipe_k);
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                    r18_copy_k_block(k_block);
                }
                r18_add_delta_s(tSrS_local);
                pipeline_k.consumer_release(my_pipe_k);
                ++my_pipe_k;

                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                        tSrS_local);
                }
            } else {
                // --- WG1: PV GEMM for tile i-1 (P ready from prev tile's StepC) ---
                if (!is_first_tile) {
                    consumer_wait(pipeline_v, my_pipe_v);
                    r18_copy_v_block(_0{});
                    quantize(_0{}, tSrS_local_cv);

                    if (is_first_compute) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                            cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                                    make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
                            if (v_block < size<2>(tOrP) - 1) {
                                r18_copy_v_block(v_block + 1);
                                quantize(v_block + 1, tSrS_local_cv);
                            }
                        }
                        is_first_compute = false;
                    } else {
                        Tensor tOrO = make_fragment_like(tOrO_store);
                        CUTLASS_PRAGMA_UNROLL
                        for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                            cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                                    make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
                            if (v_block < size<2>(tOrP) - 1) {
                                r18_copy_v_block(v_block + 1);
                                quantize(v_block + 1, tSrS_local_cv);
                            }
                        }
                        softmax_fused.rescale_o(tOrO_store, tOrO);
                    }
                    pipeline_v.consumer_release(my_pipe_v);
                    ++my_pipe_v;
                } else {
                    // First tile: init softmax state (no previous PV to do)
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }
            }

            // ============ Step B: WG0=soft_full+quant(MUFU) + cross-tile QK_{i+1}(OMMA) ============
            //                     WG1=QK_i(OMMA)+soft_1st(MUFU)
            if (wg_id == 0) {
                // --- WG0: FULL softmax + quantize for tile i ---
                apply_mask(tSrS_local, n_block);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                } else {
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n);
                }
                if (!is_first_compute) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mi = 0; mi < size(softmax_fused.row_max); ++mi) {
                        float _ea = (prev_block_max(mi) - softmax_fused.row_max(mi)) * mainloop_params.softmax_scale_log2;
                        float _ev; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(_ev) : "f"(_ea));
                        softmax_fused.scores_scale(mi) = _ev;
                        softmax_fused.row_sum(mi) *= _ev;
                    }
                }
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 0, mainloop_params.softmax_scale_log2);
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 1, mainloop_params.softmax_scale_log2);
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 2, mainloop_params.softmax_scale_log2);
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 3, mainloop_params.softmax_scale_log2);
                // NOTE: Do NOT call quantize_after_partial_softmax here.
                // exp2_sum_chunk already transformed AbsMaxP to exp2 scale factors.
                // The PV GEMM's quantize lambda will read acc + AbsMaxP directly.
                // quantize_after_partial_softmax divides acc by AbsMaxP which corrupts
                // the probability values (bug found 2026-03-28).
                // softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);

                // WG0: PV GEMM for tile i (quantize uses current buffer's cv)
                consumer_wait(pipeline_v, my_pipe_v);
                r18_copy_v_block(_0{});
                quantize(_0{}, tSrS_local_cv);

                if (is_first_compute) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                        cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                                make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
                        if (v_block < size<2>(tOrP) - 1) {
                            r18_copy_v_block(v_block + 1);
                            quantize(v_block + 1, tSrS_local_cv);
                        }
                    }
                    is_first_compute = false;
                } else {
                    Tensor tOrO = make_fragment_like(tOrO_store);
                    CUTLASS_PRAGMA_UNROLL
                    for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                        cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                                make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
                        if (v_block < size<2>(tOrP) - 1) {
                            r18_copy_v_block(v_block + 1);
                            quantize(v_block + 1, tSrS_local_cv);
                        }
                    }
                    softmax_fused.rescale_o(tOrO_store, tOrO);
                }
                pipeline_v.consumer_release(my_pipe_v);
                ++my_pipe_v;
            } else {
                // --- WG1: QK GEMM for tile i + FULL softmax + quant ---
                // QK GEMM (OMMA — overlaps with WG0's softmax on MUFU)
                consumer_wait(pipeline_k, my_pipe_k);
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                    r18_copy_k_block(k_block);
                }
                r18_add_delta_s(tSrS_local);
                pipeline_k.consumer_release(my_pipe_k);
                ++my_pipe_k;

                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                        tSrS_local);
                }

                // Full softmax + quantize (MUFU — prepares P for NEXT tile's Step A PV)
                apply_mask(tSrS_local, n_block);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                } else {
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n);
                }
                if (!is_first_compute) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mi = 0; mi < size(softmax_fused.row_max); ++mi) {
                        float _ea = (prev_block_max(mi) - softmax_fused.row_max(mi)) * mainloop_params.softmax_scale_log2;
                        float _ev; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(_ev) : "f"(_ea));
                        softmax_fused.scores_scale(mi) = _ev;
                        softmax_fused.row_sum(mi) *= _ev;
                    }
                }
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 0, mainloop_params.softmax_scale_log2);
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 1, mainloop_params.softmax_scale_log2);
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 2, mainloop_params.softmax_scale_log2);
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 3, mainloop_params.softmax_scale_log2);
                // NOTE: Do NOT call quantize_after_partial_softmax here.
                // exp2_sum_chunk already transformed AbsMaxP to exp2 scale factors.
                // The PV GEMM's quantize lambda will read acc + AbsMaxP directly.
                // quantize_after_partial_softmax divides acc by AbsMaxP which corrupts
                // the probability values (bug found 2026-03-28).
                // softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);
            }
        } // end tile loop

        // ============ Tail: WG1 finishes last tile ============
        // WG1 still needs: soft_{last}_2nd + quant + PV_{last}
        if (wg_id == 1) {
            // soft 2nd half + quantize
            softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 2, mainloop_params.softmax_scale_log2);
            softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, 3, mainloop_params.softmax_scale_log2);
            // softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);  // bug: corrupts probabilities

            // PV GEMM for last tile
            consumer_wait(pipeline_v, my_pipe_v);
            r18_copy_v_block(_0{});
            quantize(_0{}, tSrS_local_cv);

            if (is_first_compute) {
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
                    if (v_block < size<2>(tOrP) - 1) {
                        r18_copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_local_cv);
                    }
                }
            } else {
                Tensor tOrO = make_fragment_like(tOrO_store);
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
                    if (v_block < size<2>(tOrP) - 1) {
                        r18_copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_local_cv);
                    }
                }
                softmax_fused.rescale_o(tOrO_store, tOrO);
            }
            pipeline_v.consumer_release(my_pipe_v);
            ++my_pipe_v;
        }

        // Write back pipeline state for the persistent kernel's next block.
        smem_pipe_read_k = my_pipe_k;
        smem_pipe_read_v = my_pipe_v;

        // Write back mbarrier wait phase for the next block.
        // CRITICAL: WG0 only waits on channel 1 (WG1→WG0), so only WG0
        // correctly tracks r18_phase[1]. WG1 only waits on channel 0
        // (WG0→WG1), so only WG1 correctly tracks r18_phase[0].
        // Each WG must ONLY write the channel it tracked — otherwise the
        // other WG's stale value (always 0) overwrites the correct phase,
        // causing try_wait phase mismatch → deadlock on the next block.
        if (thread_idx == 0) {
            int my_channel = 1 - wg_id;  // WG0→channel 1, WG1→channel 0
            shared_storage.r18_wait_phase[my_channel] = r18_phase[my_channel];
        }

#ifdef R18_DEBUG
        if (thread_idx == 0 && blockIdx.x == 0) {
            printf("R18 END wg=%d K_stage=%d K_phase=%d V_stage=%d V_phase=%d phase0=%d phase1=%d\n",
                   wg_id, my_pipe_k.index(), my_pipe_k.phase(), my_pipe_v.index(), my_pipe_v.phase(),
                   r18_phase[0], r18_phase[1]);
        }
#endif

        softmax_fused.finalize(tOrO_store);
        return;
#endif  // R18_PINGPONG_OVERLAP

#if defined(CROSS_TILE_DOUBLE_BUF)
        // ============================================================
        // Cross-tile double-buffer register pressure probe.
        //
        // Forces tSrS_buf1 + AbsMaxP_buf1 to remain allocated alongside
        // the primary buffers across the tile loop. The compiler must keep
        // both sets of registers live because we read from buf1 on each
        // tile iteration (simulating the cross-tile overlap pattern where
        // softmax[i] reads tSrS_buf1 while GEMM writes tSrS).
        //
        // Build with PTXAS_VERBOSE=1 to check spill.
        // If spill > 0: Direction D is WS-infeasible for WS architecture.
        // ============================================================
#endif

        // ============================================================
        // Original ping-pong (R16/R17): Both WGs iterate ALL tiles together.
        // ============================================================

        #pragma unroll 1
#if defined(PHASE_TRACE_GAPS)
        uint64_t _trace_prev_pv_end = 0;
        int _trace_count = 0;
        int _trace_tile[PHASE_TRACE_GAPS];
        int _trace_n[PHASE_TRACE_GAPS];
        uint64_t _trace_prev_loop[PHASE_TRACE_GAPS];
        uint64_t _trace_prev_qk[PHASE_TRACE_GAPS];
        uint64_t _trace_k_wait[PHASE_TRACE_GAPS];
        uint64_t _trace_k_prep[PHASE_TRACE_GAPS];
        uint64_t _trace_qk[PHASE_TRACE_GAPS];
        uint64_t _trace_soft[PHASE_TRACE_GAPS];
        uint64_t _trace_soft_to_pv_order[PHASE_TRACE_GAPS];
        uint64_t _trace_pv_order_wait[PHASE_TRACE_GAPS];
        uint64_t _trace_v_wait[PHASE_TRACE_GAPS];
        uint64_t _trace_v_copy_quant[PHASE_TRACE_GAPS];
        uint64_t _trace_pv_gate_wait[PHASE_TRACE_GAPS];
        uint64_t _trace_soft_to_pv_omma[PHASE_TRACE_GAPS];
        uint64_t _trace_pv[PHASE_TRACE_GAPS];
        uint64_t _trace_total[PHASE_TRACE_GAPS];
#endif
        for (int tile_idx = 0; tile_idx < n_block_count; ++tile_idx) {
            int n_block = n_block_count - 1 - tile_idx;

#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_loop_start = clock64();
            uint64_t _trace_prev_pv_to_loop = _trace_prev_pv_end == 0 ? 0 : _trace_loop_start - _trace_prev_pv_end;
            uint64_t _trace_k_wait_start = _trace_loop_start;
#endif
#if defined(STEADY_SOFT_CREDIT_GATE)
            bool soft_credit_gate_arrived = false;
            auto arrive_soft_credit_gate = [&]() {
                if (!soft_credit_gate_arrived && tile_idx > 0) {
                    shared_storage.strict_phase_gate[1 - wg_id].arrive();
                    soft_credit_gate_arrived = true;
                }
            };
#endif
            // --- K: both WGs wait for data ready ---
            consumer_wait(pipeline_k, smem_pipe_read_k);
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_k_ready = clock64();
#endif

#if defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_QK_ORDER) && !defined(STEADY_STRICT_PHASE_GATE) && (!defined(STEADY_SOFT_CREDIT_GATE) || defined(STEADY_SOFT_CREDIT_KEEP_QK_ORDER))
#if defined(CAUSAL_DISABLE_QK_ORDER)
            // R20: causal tiles skip QK-side ordering; PV ordering remains intact.
            if constexpr (!Is_causal) {
                math_order.wait();
            }
#else
            math_order.wait();
#endif
#elif defined(PINGPONG_NANOSLEEP)
            if (wg_id == 1 && tile_idx == 0) {
                asm volatile("nanosleep.u32 500;");
            }
#endif
            // --- QK GEMM ---
            Tensor tSrS_local = partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
            Tensor tSrS_local_cv = make_tensor(tSrS_local.data(), sage::convert_to_conversion_layout(tSrS_local.layout()));

#if defined(R17_NTILE_INTERLEAVE)
            // R17: Manual N-sub-tile dispatch with GEMM-softmax interleaving.
            // NOTE: R17 path subsumes PINGPONG_EARLY_RELEASE_K (does its own early K release).
            //
            // Swaps loop order to N-outer / K-inner so that after each N-sub-tile
            // completes (all k_blocks), we can do partial online softmax on those
            // 32 columns while the NEXT N-sub-tile's MMAs are on the OMMA pipeline.
            //
            // Pre-load ALL K data to registers, release pipeline early.
#if defined(PHASE_TIMING)
            uint64_t _pt_qk_start = clock64();
#endif
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_qk_start = clock64();
#endif
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                copy_k_block(k_block);
            }
            add_delta_s(tSrS_local);
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;
            tma_refill.refill_k(tile_idx);
#if defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_QK_ORDER) && !defined(STEADY_STRICT_PHASE_GATE) && (!defined(STEADY_SOFT_CREDIT_GATE) || defined(STEADY_SOFT_CREDIT_KEEP_QK_ORDER))
#if defined(CAUSAL_DISABLE_QK_ORDER)
            if constexpr (!Is_causal) {
                math_order.arrive();
            }
#else
            math_order.arrive();
#endif
#endif
            constexpr int MmaN_qk = decltype(size<2>(tSrS_local))::value;

#if defined(R17_OVERLAP)
            // ============================================================
            // Direction C: N-sub-tile GEMM + find_max interleaving
            //
            // Interleave find_max (Pass 1, pure FMA) between N-sub-tile GEMMs.
            // find_max hides in MMA pipeline latency (~128cy per 4 MMA).
            // After all GEMMs: apply_mask, rescale, exp2+sum+quantize.
            // ============================================================
#if defined(PHASE_TIMING)
            uint64_t _pt_qk_end;  // assigned inside: marks end of GEMM (before softmax exp2)
#endif
            {
                auto prev_block_max = make_fragment_like(softmax_fused.row_max);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                }

                if constexpr (!Is_causal) {
                    // ============================================================
                    // Non-causal Direction C: GEMM + softmax interleaving
                    // ============================================================
#if defined(R17_ONLINE_OVERLAP)
                    // Variant: online_softmax_chunk (find_max+exp2+rescale per sub-tile)
                    // MUFU exp2 hidden in MMA latency. Tested: -5% due to rescale overhead.
                    CUTLASS_PRAGMA_UNROLL
                    for (int n = 0; n < MmaN_qk; ++n) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                            cute::gemm(tiled_mma_qk,
                                make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                                make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, n),
                                tSrS_local(_, _0{}, n));
                        }
                        if (n > 0) {
                            softmax_fused.template online_softmax_chunk</*IsInit=*/false>(
                                tSrS_local, AbsMaxP, n - 1, mainloop_params.softmax_scale_log2);
                        } else if (is_first_compute) {
                            fill(softmax_fused.row_max, -INFINITY);
                            clear(softmax_fused.row_sum);
                            fill(softmax_fused.scores_scale, 1.f);
                        }
                    }
                    if (is_first_compute && MmaN_qk == 1) {
                        softmax_fused.template online_softmax_chunk</*IsInit=*/true>(
                            tSrS_local, AbsMaxP, 0, mainloop_params.softmax_scale_log2);
                    } else {
                        softmax_fused.template online_softmax_chunk</*IsInit=*/false>(
                            tSrS_local, AbsMaxP, MmaN_qk - 1, mainloop_params.softmax_scale_log2);
                    }
#if defined(PHASE_TIMING)
                    _pt_qk_end = clock64();  // GEMM + online softmax interleaved
#endif
                    apply_mask(tSrS_local, n_block);
                    // Cross-block rescale
                    if (!is_first_compute) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int mi = 0; mi < size(softmax_fused.row_max); ++mi) {
                            float _exp_arg = (prev_block_max(mi) - softmax_fused.row_max(mi))
                                             * mainloop_params.softmax_scale_log2;
                            float _exp_val;
                            asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(_exp_val) : "f"(_exp_arg));
                            softmax_fused.scores_scale(mi) = _exp_val;
                        }
                    }
                    softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);
#else
                    // Default: find_max-only interleaving (tested: -2% vs baseline)
                    if (is_first_compute) {
                        fill(softmax_fused.row_max, -INFINITY);
                        clear(softmax_fused.row_sum);
                        fill(softmax_fused.scores_scale, 1.f);
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int n = 0; n < MmaN_qk; ++n) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                            cute::gemm(tiled_mma_qk,
                                make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                                make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, n),
                                tSrS_local(_, _0{}, n));
                        }
                        if (n > 0) {
                            softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n - 1);
                        }
                    }
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, MmaN_qk - 1);
#if defined(PHASE_TIMING)
                    _pt_qk_end = clock64();  // GEMM + find_max interleaved
#endif
                    apply_mask(tSrS_local, n_block);
                    softmax_fused.template exp2_sum_and_quantize<>(
                        tSrS_local, AbsMaxP, is_first_compute,
                        mainloop_params.softmax_scale_log2, prev_block_max);
#endif  // R17_ONLINE_OVERLAP
                } else {
                    // ============================================================
                    // Causal: K-outer GEMM then full softmax (no interleaving).
                    // Mask must precede softmax; per-sub-tile masking not feasible.
                    // ============================================================
                    CUTLASS_PRAGMA_UNROLL
                    for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                        cute::gemm(tiled_mma_qk,
                            make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                            make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                            tSrS_local);
                    }
#if defined(PHASE_TIMING)
                    _pt_qk_end = clock64();  // GEMM only (no interleaving)
#endif
                    apply_mask(tSrS_local, n_block);
                    softmax_fused.template chunked_softmax_fixed<>(
                        tSrS_local, AbsMaxP, is_first_compute,
                        mainloop_params.softmax_scale_log2, prev_block_max);
                }
            }

#else  // !R17_OVERLAP: separate GEMM then softmax

            // QK GEMM: choose between full K-outer dispatch and N-sub-tile dispatch.
#if defined(R17_SUBTILE_GEMM)
            // N-outer / K-inner: manual sub-tile dispatch
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaN_qk; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, n),
                        tSrS_local(_, _0{}, n));
                }
            }
#else
            // Full K-outer dispatch (correct, matches default path)
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS_local);
            }
#endif

#if defined(PHASE_TIMING)
            uint64_t _pt_qk_end = clock64();
#endif

            // Non-overlap path: chunked_softmax_fixed (complete softmax after GEMM)
            {
                auto prev_block_max = make_fragment_like(softmax_fused.row_max);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                }

                apply_mask(tSrS_local, n_block);

                softmax_fused.template chunked_softmax_fixed<>(
                    tSrS_local, AbsMaxP, is_first_compute,
                    mainloop_params.softmax_scale_log2, prev_block_max);
            }
#endif  // R17_OVERLAP

#if 0  // OLD: broken chunked softmax (find_max_chunk/exp2_sum_chunk) — kept for reference
#if defined(R17_OVERLAP)
            // ============================================================
            // R17 方案 A: GEMM-softmax overlap (on-the-fly rescale)
            //
            // Non-causal: interleave online_softmax_chunk with QK GEMM.
            // Causal: falls back to 方案 B (two-pass) because causal mask
            //   changes row_max after QK GEMM, breaking online rescale.
            // ============================================================

            if constexpr (!Is_causal) {
                    // --- NON-CAUSAL: online softmax interleaved with GEMM ---
                    auto prev_block_max = make_fragment_like(softmax_fused.row_max);
                    if (!is_first_compute) {
                        cute::copy(softmax_fused.row_max, prev_block_max);
                    }

                    CUTLASS_PRAGMA_UNROLL
                    for (int n = 0; n < MmaN_qk; ++n) {
                        {
                            auto A_zip = make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ(_, _, _0{}));
                            auto B_zip = make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{}));
                            cute::gemm(tiled_mma_qk, A_zip(_, _0{}), B_zip(_, n), tSrS_local(_, _0{}, n));
                        }

                        if (n > 0) {
                            softmax_fused.template online_softmax_chunk</*IsInit=*/false>(
                                tSrS_local, AbsMaxP, n - 1, mainloop_params.softmax_scale_log2);
                        } else if (is_first_compute) {
                            fill(softmax_fused.row_max, -INFINITY);
                            clear(softmax_fused.row_sum);
                            fill(softmax_fused.scores_scale, 1.f);
                        }

                        if constexpr (size<2>(tSrQ) > 1) {
                            auto A_zip = make_zip_tensor(tSrQ(_, _, _1{}), tSrSFQ(_, _, _1{}));
                            auto B_zip = make_zip_tensor(tSrK(_, _, _1{}), tSrSFK(_, _, _1{}));
                            cute::gemm(tiled_mma_qk, A_zip(_, _0{}), B_zip(_, n), tSrS_local(_, _0{}, n));
                        }
                    }
                    if (is_first_compute && MmaN_qk == 1) {
                        softmax_fused.template online_softmax_chunk</*IsInit=*/true>(
                            tSrS_local, AbsMaxP, 0, mainloop_params.softmax_scale_log2);
                    } else {
                        softmax_fused.template online_softmax_chunk</*IsInit=*/false>(
                            tSrS_local, AbsMaxP, MmaN_qk - 1, mainloop_params.softmax_scale_log2);
                    }

                    if (!is_first_compute) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int mi = 0; mi < size(softmax_fused.row_max); ++mi) {
                            float _exp_arg = (prev_block_max(mi) - softmax_fused.row_max(mi)) * mainloop_params.softmax_scale_log2;
                            float _exp_val;
                            asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(_exp_val) : "f"(_exp_arg));
                            softmax_fused.scores_scale(mi) = _exp_val;
                        }
                    }

                    // NOTE: Do NOT call quantize_after_partial_softmax here.
                // exp2_sum_chunk already transformed AbsMaxP to exp2 scale factors.
                // The PV GEMM's quantize lambda will read acc + AbsMaxP directly.
                // quantize_after_partial_softmax divides acc by AbsMaxP which corrupts
                // the probability values (bug found 2026-03-28).
                // softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);
                } else {
                // --- CAUSAL: use 方案 B two-pass (mask must be applied before softmax) ---
                // This path is identical to the R17 方案 B below, but inlined here
                // so that R17_OVERLAP builds correctly for causal configs.
                auto prev_block_max = make_fragment_like(softmax_fused.row_max);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                }
                if (is_first_compute) {
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }

                // QK GEMM + Pass 1 find_max interleaved
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    {
                        auto A_zip = make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ(_, _, _0{}));
                        auto B_zip = make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{}));
                        cute::gemm(tiled_mma_qk, A_zip(_, _0{}), B_zip(_, n), tSrS_local(_, _0{}, n));
                    }
                    if (n > 0) {
                        softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n - 1);
                    }
                    if constexpr (size<2>(tSrQ) > 1) {
                        auto A_zip = make_zip_tensor(tSrQ(_, _, _1{}), tSrSFQ(_, _, _1{}));
                        auto B_zip = make_zip_tensor(tSrK(_, _, _1{}), tSrSFK(_, _, _1{}));
                        cute::gemm(tiled_mma_qk, A_zip(_, _0{}), B_zip(_, n), tSrS_local(_, _0{}, n));
                    }
                }
                softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, MmaN_qk - 1);

                apply_mask(tSrS_local, n_block);

                // Causal: recompute max after masking
                if (is_first_compute) {
                    fill(softmax_fused.row_max, -INFINITY);
                } else {
                    cute::copy(prev_block_max, softmax_fused.row_max);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n);
                }

                if (!is_first_compute) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mi = 0; mi < size(softmax_fused.row_max); ++mi) {
                        float _exp_arg = (prev_block_max(mi) - softmax_fused.row_max(mi)) * mainloop_params.softmax_scale_log2;
                        float _exp_val;
                        asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(_exp_val) : "f"(_exp_arg));
                        softmax_fused.scores_scale(mi) = _exp_val;
                        softmax_fused.row_sum(mi) *= softmax_fused.scores_scale(mi);
                    }
                }

                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, n, mainloop_params.softmax_scale_log2);
                }

                // NOTE: Do NOT call quantize_after_partial_softmax here.
                // exp2_sum_chunk already transformed AbsMaxP to exp2 scale factors.
                // The PV GEMM's quantize lambda will read acc + AbsMaxP directly.
                // quantize_after_partial_softmax divides acc by AbsMaxP which corrupts
                // the probability values (bug found 2026-03-28).
                // softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);
            }

#else
            // ============================================================
            // R17 方案 B: 两遍 partial softmax (correctness baseline)
            //
            // Pass 1 (find_max): 纯 FMA, 与 GEMM gap 交错
            // Pass 2 (exp2+sum): 用 global max, GEMM 后执行
            // ============================================================

            auto prev_block_max = make_fragment_like(softmax_fused.row_max);
            if (!is_first_compute) {
                cute::copy(softmax_fused.row_max, prev_block_max);
            }
            if (is_first_compute) {
                fill(softmax_fused.row_max, -INFINITY);
                clear(softmax_fused.row_sum);
                fill(softmax_fused.scores_scale, 1.f);
            }

            // QK GEMM + Pass 1 find_max 交错
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaN_qk; ++n) {
                {
                    auto A_zip = make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ(_, _, _0{}));
                    auto B_zip = make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{}));
                    cute::gemm(tiled_mma_qk, A_zip(_, _0{}), B_zip(_, n), tSrS_local(_, _0{}, n));
                }
                if (n > 0) {
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n - 1);
                }
                if constexpr (size<2>(tSrQ) > 1) {
                    auto A_zip = make_zip_tensor(tSrQ(_, _, _1{}), tSrSFQ(_, _, _1{}));
                    auto B_zip = make_zip_tensor(tSrK(_, _, _1{}), tSrSFK(_, _, _1{}));
                    cute::gemm(tiled_mma_qk, A_zip(_, _0{}), B_zip(_, n), tSrS_local(_, _0{}, n));
                }
            }
            softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, MmaN_qk - 1);

            apply_mask(tSrS_local, n_block);

            if constexpr (Is_causal) {
                if (is_first_compute) {
                    fill(softmax_fused.row_max, -INFINITY);
                } else {
                    cute::copy(prev_block_max, softmax_fused.row_max);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n);
                }
            }

            if (!is_first_compute) {
                CUTLASS_PRAGMA_UNROLL
                for (int mi = 0; mi < size(softmax_fused.row_max); ++mi) {
                    float _exp_arg = (prev_block_max(mi) - softmax_fused.row_max(mi)) * mainloop_params.softmax_scale_log2;
                    float _exp_val;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(_exp_val) : "f"(_exp_arg));
                    softmax_fused.scores_scale(mi) = _exp_val;
                    softmax_fused.row_sum(mi) *= softmax_fused.scores_scale(mi);
                }
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaN_qk; ++n) {
                softmax_fused.exp2_sum_chunk(tSrS_local, AbsMaxP, n, mainloop_params.softmax_scale_log2);
            }

            // softmax_fused.quantize_after_partial_softmax(tSrS_local, AbsMaxP);  // bug: corrupts probabilities

#if defined(PHASE_TIMING)
            uint64_t _pt_soft_end = clock64();
#endif
#endif  // R17_OVERLAP
#endif  // #if 0 (chunked softmax disabled)

#elif defined(PINGPONG_EARLY_RELEASE_K)
            // Early K release + full GEMM (R19 default when not using R17 interleaving)
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
                copy_k_block(k_block);
            }
            add_delta_s(tSrS_local);
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;
#if defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_QK_ORDER) && !defined(STEADY_STRICT_PHASE_GATE) && (!defined(STEADY_SOFT_CREDIT_GATE) || defined(STEADY_SOFT_CREDIT_KEEP_QK_ORDER)) && ((R19_QK_LEAD_MODE == 0) || defined(MMA_SOFTMAX_INTERLEAVE))
#if defined(CAUSAL_DISABLE_QK_ORDER)
            // R20: causal QK does not participate in this release token.
            if constexpr (!Is_causal) {
                math_order.arrive();
            }
#else
            math_order.arrive();
#endif
#endif
#if defined(PHASE_TIMING)
            uint64_t _pt_qk_start = clock64();
#endif
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_qk_start = clock64();
#endif

#if defined(MMA_SOFTMAX_INTERLEAVE)
            // ============================================================
            // R22: QK GEMM with find_max interleaving
            //
            // R22_MODE=0: Full N-outer/K-inner + find_max between subtiles (=R17 style)
            // R22_MODE=1: K-outer k=0, N-outer k=1, NO find_max (dispatch overhead only)
            // R22_MODE=2: K-outer k=0, N-outer k=1, find_max between subtiles
            // Default (no R22_MODE): same as mode 0
            // ============================================================
            {
                constexpr int MmaN_qk = decltype(size<2>(tSrS_local))::value;

#if !defined(R22_MODE) || (R22_MODE == 0)
                // Mode 0: Full N-outer/K-inner + find_max (same as before)
                if (is_first_compute) {
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                        cute::gemm(tiled_mma_qk,
                            make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                            make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, n),
                            tSrS_local(_, _0{}, n));
                    }
                    if (n > 0) {
                        softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n - 1);
                    }
                }
                softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, MmaN_qk - 1);

#elif (R22_MODE == 1)
                // Mode 1: K-outer k=0..K-2, N-outer last k_block (per-subtile), NO find_max
                // For D=64 (1 k_block): just do N-outer for the only k_block
                // For D=128 (2 k_blocks): K-outer k=0, N-outer k=1
                if constexpr (size<2>(decltype(tSrQ){}) > 1) {
                    // D>=128: K-outer for k=0
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ(_, _, _0{})),
                        make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{})),
                        tSrS_local);
                }
                // N-outer for last k_block
                {
                    constexpr int last_k = size<2>(decltype(tSrQ){}) - 1;
                    CUTLASS_PRAGMA_UNROLL
                    for (int n = 0; n < MmaN_qk; ++n) {
                        cute::gemm(tiled_mma_qk,
                            make_zip_tensor(tSrQ(_, _, Int<last_k>{}), tSrSFQ(_, _, Int<last_k>{}))(_, _0{}),
                            make_zip_tensor(tSrK(_, _, Int<last_k>{}), tSrSFK(_, _, Int<last_k>{}))(_, n),
                            tSrS_local(_, _0{}, n));
                    }
                }

#elif (R22_MODE == 3)
                // Mode 3: K-outer k=0, N-outer k=1, find_max INSIDE ATOM GAPS
                // Hand-written MMA unpack + fma_with_interleave.
                // Gap callback executes find_max_chunk(n-1) between the 4 hardware MMAs
                // of subtile n's atom, while those MMAs are in the TC pipeline.
                if (is_first_compute) {
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }
                if constexpr (size<2>(decltype(tSrQ){}) > 1) {
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ(_, _, _0{})),
                        make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{})),
                        tSrS_local);
                }
                {
                    constexpr int last_k = size<2>(decltype(tSrQ){}) - 1;
                    CUTLASS_PRAGMA_UNROLL
                    for (int n = 0; n < MmaN_qk; ++n) {
                        auto A_zip = make_zip_tensor(
                            tSrQ(_, _, Int<last_k>{}), tSrSFQ(_, _, Int<last_k>{}))(_, _0{});
                        auto B_zip = make_zip_tensor(
                            tSrK(_, _, Int<last_k>{}), tSrSFK(_, _, Int<last_k>{}))(_, n);
                        auto C_n = tSrS_local(_, _0{}, n);

                        if (n > 0) {
                            // Hand-written MMA with find_max in atom-internal gaps.
                            // Replicate mma_unpack from mma_traits_sm120.hpp:77-113
                            // but call fma_with_interleave instead of fma.
                            using RegTypeSF = cute::uint_bit_t<32>;
                            auto [A_data, SFA_data] = unzip_tensor(A_zip);
                            auto [B_data, SFB_data] = unzip_tensor(B_zip);
                            auto rA   = recast<uint32_t>(A_data);
                            auto rB   = recast<uint32_t>(B_data);
                            auto rD   = recast<float>(C_n);
                            auto rSFA = recast<RegTypeSF>(filter_zeros(SFA_data));
                            auto rSFB = recast<RegTypeSF>(filter_zeros(SFB_data));

                            const int prev_n = n - 1;
                            cute::SM120::BLOCKSCALED::fma_with_interleave(
                                rD(0),  rD(1),  rD(2),  rD(3),
                                rD(4),  rD(5),  rD(6),  rD(7),
                                rD(8),  rD(9),  rD(10), rD(11),
                                rD(12), rD(13), rD(14), rD(15),
                                rA(0),  rA(1),  rA(2),  rA(3),
                                rB(0),  rB(1),  rB(2),  rB(3),
                                rB(4),  rB(5),  rB(6),  rB(7),
                                rD(0),  rD(1),  rD(2),  rD(3),
                                rD(4),  rD(5),  rD(6),  rD(7),
                                rD(8),  rD(9),  rD(10), rD(11),
                                rD(12), rD(13), rD(14), rD(15),
                                rSFA(0), rSFB(0),
                                [&](int gap_idx) {
                                    // find_max on previous subtile — only in first gap
                                    if (gap_idx == 0) {
                                        softmax_fused.find_max_chunk(
                                            tSrS_local, AbsMaxP, prev_n);
                                    }
                                });
                        } else {
                            // n==0: no previous subtile, use standard cute::gemm
                            cute::gemm(tiled_mma_qk, C_n, A_zip, B_zip, C_n);
                        }
                    }
                    softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, MmaN_qk - 1);
                }

#elif (R22_MODE == 2)
                // Mode 2: K-outer k=0, N-outer k=1 + find_max between subtiles
                if (is_first_compute) {
                    fill(softmax_fused.row_max, -INFINITY);
                    clear(softmax_fused.row_sum);
                    fill(softmax_fused.scores_scale, 1.f);
                }
                cute::gemm(tiled_mma_qk,
                    make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ(_, _, _0{})),
                    make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{})),
                    tSrS_local);
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < MmaN_qk; ++n) {
                    cute::gemm(tiled_mma_qk,
                        make_zip_tensor(tSrQ(_, _, _1{}), tSrSFQ(_, _, _1{}))(_, _0{}),
                        make_zip_tensor(tSrK(_, _, _1{}), tSrSFK(_, _, _1{}))(_, n),
                        tSrS_local(_, _0{}, n));
                    if (n > 0) {
                        softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, n - 1);
                    }
                }
                softmax_fused.find_max_chunk(tSrS_local, AbsMaxP, MmaN_qk - 1);
#endif
            }
#else
            // R19 default: K-outer QK GEMM (no interleave)
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS_local);
#if defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_QK_ORDER) && !defined(STEADY_STRICT_PHASE_GATE) && (!defined(STEADY_SOFT_CREDIT_GATE) || defined(STEADY_SOFT_CREDIT_KEEP_QK_ORDER)) && (R19_QK_LEAD_MODE == 1)
                if (k_block == 0) {
#if defined(CAUSAL_DISABLE_QK_ORDER)
                    if constexpr (!Is_causal) {
                        math_order.arrive();
                    }
#else
                    math_order.arrive();
#endif
                }
#endif
#if defined(STEADY_SOFT_CREDIT_GATE)
                if (k_block + 1 >= STEADY_SOFT_CREDIT_K_BLOCKS) {
                    arrive_soft_credit_gate();
                }
#endif
            }
#if defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_QK_ORDER) && !defined(STEADY_STRICT_PHASE_GATE) && (!defined(STEADY_SOFT_CREDIT_GATE) || defined(STEADY_SOFT_CREDIT_KEEP_QK_ORDER)) && (R19_QK_LEAD_MODE == 2)
#if defined(CAUSAL_DISABLE_QK_ORDER)
            if constexpr (!Is_causal) {
                math_order.arrive();
            }
#else
            math_order.arrive();
#endif
#endif
#endif  // MMA_SOFTMAX_INTERLEAVE

#if defined(PHASE_TIMING)
            uint64_t _pt_qk_end = clock64();
#endif
#else
            // Default path: streaming K copy + GEMM (no early release)
            copy_k_block(_0{});
            add_delta_s(tSrS_local);
#if defined(PHASE_TIMING)
            uint64_t _pt_qk_start = clock64();
#endif
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_qk_start = clock64();
#endif
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
                cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                                        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS_local);
                if (k_block < size<2>(tSrQ) - 1) {
                    copy_k_block(k_block + 1);
                }
            }
#if defined(PHASE_TIMING)
            uint64_t _pt_qk_end = clock64();
#endif
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;
            // Non-WS refill: thread 0 issues TMA for the next K tile
            tma_refill.refill_k(tile_idx);
#if defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_QK_ORDER) && !defined(STEADY_STRICT_PHASE_GATE) && (!defined(STEADY_SOFT_CREDIT_GATE) || defined(STEADY_SOFT_CREDIT_KEEP_QK_ORDER))
#if defined(CAUSAL_DISABLE_QK_ORDER)
            if constexpr (!Is_causal) {
                math_order.arrive();
            }
#else
            math_order.arrive();
#endif
#endif
#endif

#if defined(STEADY_SOFT_CREDIT_GATE)
            arrive_soft_credit_gate();
#elif defined(STEADY_STRICT_PHASE_GATE)
            if (tile_idx > 0) {
                shared_storage.strict_phase_gate[1 - wg_id].arrive();
            }
#endif
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_qk_end = clock64();
#endif

            // --- Softmax (MUFU) ---
#if !defined(R17_NTILE_INTERLEAVE)

#if defined(MMA_SOFTMAX_INTERLEAVE) && (!defined(R22_MODE) || (R22_MODE == 0) || (R22_MODE == 2))
            // R22 Mode 0/2: find_max already done in QK loop above.
            // Apply mask, then use exp2_sum_and_quantize (no find_max needed).
            apply_mask(tSrS_local, n_block);
            {
                auto prev_block_max = make_fragment_like(softmax_fused.row_max);
                if (!is_first_compute) {
                    cute::copy(softmax_fused.row_max, prev_block_max);
                }
                softmax_fused.template exp2_sum_and_quantize<>(
                    tSrS_local, AbsMaxP, is_first_compute,
                    mainloop_params.softmax_scale_log2, prev_block_max);
            }
#else
            // Default softmax (non-R17 path). R17 uses chunked_softmax_fixed above.
            apply_mask(tSrS_local, n_block);

#if defined(SMEM_SCORE_BUFFER)
            // ============================================================
            // Phase 1: smem score buffer round-trip validation
            //
            // Dump tSrS_local (FP32 regs) → smem_o (reinterpreted as float*)
            // Then load back → validate that softmax produces identical results.
            //
            // smem_o = 128×128×BF16 = 32KB. Reinterpreted as FP32:
            //   can hold 64×128 floats = 32KB = exactly one WG's scores.
            //
            // Both WGs share smem_o, so they take turns (WG0 first, then WG1).
            // Phase 2 will naturally avoid this by WG alternation.
            // ============================================================
            {
                float* score_buf = reinterpret_cast<float*>(shared_storage.smem_o.begin());

                // Get per-thread (row, col) coordinate mapping from MMA partition
                Tensor cS_buf = cute::make_identity_tensor(
                    make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
                Tensor tScS_buf = thread_mma_qk.partition_C(cS_buf);

                // WG0 goes first, then WG1
                // Consumer-wide barrier ensures both WGs are synchronized
                // before and after each WG's dump+load cycle.
                for (int dump_wg = 0; dump_wg < 2; ++dump_wg) {
                    if (wg_id == dump_wg) {
                        // --- Dump: regs → smem ---
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < size(tSrS_local); ++i) {
                            int row = get<0>(tScS_buf(i));
                            int col = get<1>(tScS_buf(i));
                            score_buf[row * kBlockN + col] = tSrS_local(i);
                        }
                    }

                    // All consumer threads sync (ensures dump is visible)
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::sync(2 * NumMmaThreads, 1 /*smem_score_barrier*/);

                    if (wg_id == dump_wg) {
                        // --- Load: smem → regs ---
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < size(tSrS_local); ++i) {
                            int row = get<0>(tScS_buf(i));
                            int col = get<1>(tScS_buf(i));
                            tSrS_local(i) = score_buf[row * kBlockN + col];
                        }
                    }

                    // Sync again before the other WG uses smem_o
                    cutlass::arch::NamedBarrier::sync(2 * NumMmaThreads, 1 /*smem_score_barrier*/);
                }
            }
#endif  // SMEM_SCORE_BUFFER

#if defined(FP16_SOFTMAX)
            // ============================================================
            // FP16 Softmax — Step 3: True half register storage
            //
            // 1. Copy FP32 tSrS_local → __half tSrS_f16[] (32 regs vs 64)
            // 2. Clobber tSrS_local to hint compiler it's dead (free 64 regs)
            // 3. Softmax reads from tSrS_f16, computes in FP32 (MUFU), writes to tSrS_f16
            // 4. Copy tSrS_f16 → tSrS_local before quantize (restore FP32 for downstream)
            //
            // Net register change during softmax:
            //   tSrS_f16: 32 regs (half[64] packed)
            //   tSrS_local: dead → 0 regs (compiler can reuse)
            //   Savings: 64 - 32 = 32 regs freed for compiler optimization
            //
            // Precision: cos_sim >= 0.99999988 (verified Step 1 & 2).
            // ============================================================
            {
                // Allocate FP16 score storage
                constexpr int N_ELEM = decltype(size(tSrS_local))::value;
                __half tSrS_f16[N_ELEM];

                // Step 1: FP32 → FP16 copy
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N_ELEM; ++i) {
                    tSrS_f16[i] = __float2half(tSrS_local(i));
                }

                // Step 2: Hint compiler that tSrS_local is dead for now.
                // Use asm volatile to prevent compiler from keeping FP32 regs alive.
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N_ELEM; ++i) {
                    asm volatile("" : "+f"(tSrS_local(i)));
                }

                // Step 3: Softmax on FP16 data
                // Read from tSrS_f16 → FP32 → compute → FP32 → write to tSrS_f16
                // We create a temporary FP32 view for the softmax function,
                // but the canonical storage is in tSrS_f16.
                // Since online_softmax_with_quant_fp16 operates on tSrS_local (FP32),
                // we first restore from f16, call softmax, then save back to f16.
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N_ELEM; ++i) {
                    tSrS_local(i) = __half2float(tSrS_f16[i]);
                }

                if (is_first_compute) {
                    softmax_fused.template online_softmax_with_quant_fp16</*Is_first=*/true>(
                        tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
                } else {
                    softmax_fused.template online_softmax_with_quant_fp16</*Is_first=*/false>(
                        tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
                }

                // Step 4: tSrS_local now has softmax+quantized FP32 values.
                // No need to copy back from f16 — softmax wrote directly to tSrS_local.
                // tSrS_f16 goes out of scope here, compiler frees those 32 regs.
            }
#else
            if (is_first_compute) {
                softmax_fused.template online_softmax_with_quant</*Is_first=*/true>(tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
            } else {
                softmax_fused.template online_softmax_with_quant</*Is_first=*/false>(tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
            }
#endif  // FP16_SOFTMAX
#endif  // MMA_SOFTMAX_INTERLEAVE
#endif  // !R17_NTILE_INTERLEAVE
#if defined(PHASE_TIMING)
            uint64_t _pt_soft_end = clock64();
#endif
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_soft_end = clock64();
            uint64_t _trace_pv_order_wait_start = 0;
            uint64_t _trace_pv_order_wait_end = 0;
            uint64_t _trace_v_wait_start = 0;
            uint64_t _trace_v_ready = 0;
            uint64_t _trace_pv_gate_wait_start = 0;
            uint64_t _trace_pv_omma_start = 0;
#endif

#if defined(CROSS_TILE_DOUBLE_BUF)
            // Force tSrS_buf1 + AbsMaxP_buf1 live across tile boundary.
            // Simulates cross-tile pattern where next-tile GEMM writes buf1
            // while current-tile softmax reads from tSrS_local.
            // The asm volatile prevents compiler from scheduling these away.
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrS_buf1); ++i) {
                asm volatile("" : "+f"(tSrS_buf1(i)));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(AbsMaxP_buf1); ++i) {
                asm volatile("" : "+f"(AbsMaxP_buf1(i)));
            }
#endif

#if defined(STEADY_PREQUANTIZE_P) && !defined(SMEM_P_BUFFER)
            // Steady-state pipeline prerequisite: materialize all P/SFP before
            // the V wait so the score accumulator can be reused by a future
            // cross-tile QK pre-issue path.
            CUTLASS_PRAGMA_UNROLL
            for (int pv_block = 0; pv_block < size<2>(tOrP); ++pv_block) {
                quantize(pv_block, tSrS_local_cv);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrS_local); ++i) {
                asm volatile("" : "+f"(tSrS_local(i)));
            }
#endif

            // --- V: wait, PV GEMM, release ---
            // R19 PV ordering: WG1 first (it finished QK+softmax first via math_order).
            // WG1's PV overlaps with WG0's remaining softmax → TC stays busy.

#if defined(SMEM_P_BUFFER)
            // ============================================================
            // SMEM_P_BUFFER: store post-softmax P (FP32) to smem_o, one WG at a time.
            // tSrS_local is freed after store → shorter register live range across V wait.
            // After V wait, reload P from smem, quantize interleaved with PV GEMM.
            //
            // FP32: 64×128×4B = 32KB per WG. smem_o ≥ 32KB (D≥128: 32KB, D=64: 16KB).
            // For D=64 (smem_o=16KB < 32KB needed), fall through to original path.
            // ============================================================
            if constexpr (kHeadDim >= kBlockN) {
                float* p_smem = reinterpret_cast<float*>(shared_storage.smem_o.begin());

                Tensor cS_pbuf = cute::make_identity_tensor(
                    make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
                Tensor tScS_pbuf = thread_mma_qk.partition_C(cS_pbuf);

                // WG0 and WG1 serialize: each dumps 32KB FP32 to smem_o
                for (int dump_wg = 0; dump_wg < 2; ++dump_wg) {
                    if (wg_id == dump_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < size(tSrS_local); ++i) {
                            int row = get<0>(tScS_pbuf(i));
                            int col = get<1>(tScS_pbuf(i));
                            p_smem[row * kBlockN + col] = tSrS_local(i);
                        }
                    }
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::sync(2 * NumMmaThreads, 2 /*smem_p_barrier*/);
                    cutlass::arch::NamedBarrier::sync(2 * NumMmaThreads, 2 /*smem_p_barrier*/);
                }
            }
#endif  // SMEM_P_BUFFER

#if defined(R19_PV_SKEW_CYCLES)
            if (tile_idx > 0 && (R19_PV_SKEW_WG == 2 || wg_id == R19_PV_SKEW_WG)) {
                asm volatile("nanosleep.u32 %0;" :: "n"(R19_PV_SKEW_CYCLES));
            }
#endif

#if defined(PHASE_TRACE_GAPS)
            _trace_pv_order_wait_start = clock64();
#endif
#if defined(PV_PREFETCH_BEFORE_ORDER) && !defined(SMEM_P_BUFFER)
#if defined(PHASE_TRACE_GAPS)
            _trace_v_wait_start = _trace_pv_order_wait_start;
#endif
            consumer_wait(pipeline_v, smem_pipe_read_v);
#if defined(PHASE_TRACE_GAPS)
            _trace_v_ready = clock64();
#endif
            copy_v_block(_0{});
#if !defined(STEADY_PREQUANTIZE_P)
            quantize(_0{}, tSrS_local_cv);
#endif
#endif
#if !defined(STEADY_STRICT_PHASE_GATE) && !defined(STEADY_SOFT_CREDIT_GATE) && defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_PV_ORDER)
#if defined(STEADY_SPLIT_QK_PV_ORDER)
            math_order_pv.wait();
#else
            math_order.wait();
#endif
#endif
#if defined(PHASE_TRACE_GAPS)
            _trace_pv_order_wait_end = clock64();
#if !defined(PV_PREFETCH_BEFORE_ORDER) || defined(SMEM_P_BUFFER)
            _trace_v_wait_start = _trace_pv_order_wait_end;
#endif
#endif
#if !defined(PV_PREFETCH_BEFORE_ORDER) || defined(SMEM_P_BUFFER)
            consumer_wait(pipeline_v, smem_pipe_read_v);
#if defined(PHASE_TRACE_GAPS)
            _trace_v_ready = clock64();
#endif
#endif

#if defined(SMEM_P_BUFFER)
            if constexpr (kHeadDim >= kBlockN) {
                // Reload P from smem_o (FP32, same mapping as dump)
                float* p_smem = reinterpret_cast<float*>(shared_storage.smem_o.begin());

                Tensor cS_pbuf = cute::make_identity_tensor(
                    make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
                Tensor tScS_pbuf = thread_mma_qk.partition_C(cS_pbuf);

                // Serialize reload: same order as dump
                for (int load_wg = 0; load_wg < 2; ++load_wg) {
                    if (wg_id == load_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < size(tSrS_local); ++i) {
                            int row = get<0>(tScS_pbuf(i));
                            int col = get<1>(tScS_pbuf(i));
                            tSrS_local(i) = p_smem[row * kBlockN + col];
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(2 * NumMmaThreads, 2 /*smem_p_barrier*/);
                    cutlass::arch::NamedBarrier::sync(2 * NumMmaThreads, 2 /*smem_p_barrier*/);
                }
            }
            // Now quantize + PV GEMM with the standard interleaved path
            copy_v_block(_0{});
            quantize(_0{}, tSrS_local_cv);

#if defined(PHASE_TRACE_GAPS)
            _trace_pv_gate_wait_start = clock64();
#endif
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
            if (!(wg_id == 1 && tile_idx == 0)) {
                shared_storage.strict_phase_gate[wg_id].wait(strict_phase_wait_phase);
                strict_phase_wait_phase ^= 1;
            }
#endif
#if defined(PHASE_TRACE_GAPS)
            _trace_pv_omma_start = clock64();
#endif

            if (is_first_compute) {
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
#if defined(PINGPONG_MATH_ORDER) && defined(JERRY_PV_ISSUE_TOKEN) && !defined(JERRY_DISABLE_PV_ORDER)
                    if (v_block == 0) {
#if defined(STEADY_SPLIT_QK_PV_ORDER)
                        math_order_pv.arrive();
#else
                        math_order.arrive();
#endif
                    }
#endif
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_local_cv);
                    }
                }
                is_first_compute = false;
            } else {
                Tensor tOrO = make_fragment_like(tOrO_store);
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
#if defined(PINGPONG_MATH_ORDER) && defined(JERRY_PV_ISSUE_TOKEN) && !defined(JERRY_DISABLE_PV_ORDER)
                    if (v_block == 0) {
#if defined(STEADY_SPLIT_QK_PV_ORDER)
                        math_order_pv.arrive();
#else
                        math_order.arrive();
#endif
                    }
#endif
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
                        quantize(v_block + 1, tSrS_local_cv);
                    }
                }
                softmax_fused.rescale_o(tOrO_store, tOrO);
            }
#else
            // Original R19 path: quantize interleaved with V copy + PV GEMM
#if !defined(PV_PREFETCH_BEFORE_ORDER)
            copy_v_block(_0{});
#if !defined(STEADY_PREQUANTIZE_P)
            quantize(_0{}, tSrS_local_cv);
#endif
#endif

#if defined(PHASE_TRACE_GAPS)
            _trace_pv_gate_wait_start = clock64();
#endif
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
            if (!(wg_id == 1 && tile_idx == 0)) {
                shared_storage.strict_phase_gate[wg_id].wait(strict_phase_wait_phase);
                strict_phase_wait_phase ^= 1;
            }
#endif
#if defined(PHASE_TRACE_GAPS)
            _trace_pv_omma_start = clock64();
#endif

            if (is_first_compute) {
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
#if defined(PINGPONG_MATH_ORDER) && defined(JERRY_PV_ISSUE_TOKEN) && !defined(JERRY_DISABLE_PV_ORDER)
                    if (v_block == 0) {
#if defined(STEADY_SPLIT_QK_PV_ORDER)
                        math_order_pv.arrive();
#else
                        math_order.arrive();
#endif
                    }
#endif
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
#if !defined(STEADY_PREQUANTIZE_P)
                        quantize(v_block + 1, tSrS_local_cv);
#endif
                    }
                }
                is_first_compute = false;
            } else {
                Tensor tOrO = make_fragment_like(tOrO_store);
                CUTLASS_PRAGMA_UNROLL
                for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
                    cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                                            make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
#if defined(PINGPONG_MATH_ORDER) && defined(JERRY_PV_ISSUE_TOKEN) && !defined(JERRY_DISABLE_PV_ORDER)
                    if (v_block == 0) {
#if defined(STEADY_SPLIT_QK_PV_ORDER)
                        math_order_pv.arrive();
#else
                        math_order.arrive();
#endif
                    }
#endif
                    if (v_block < size<2>(tOrP) - 1) {
                        copy_v_block(v_block + 1);
#if !defined(STEADY_PREQUANTIZE_P)
                        quantize(v_block + 1, tSrS_local_cv);
#endif
                    }
                }
                softmax_fused.rescale_o(tOrO_store, tOrO);
            }
#endif
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
            if (wg_id == 1 && tile_idx == n_block_count - 1) {
                shared_storage.strict_phase_gate[0].arrive();
            }
#elif defined(PINGPONG_MATH_ORDER) && !defined(JERRY_DISABLE_PV_ORDER) && !defined(JERRY_PV_ISSUE_TOKEN)
#if defined(STEADY_SPLIT_QK_PV_ORDER)
            math_order_pv.arrive();
#else
            math_order.arrive();
#endif
#endif
            pipeline_v.consumer_release(smem_pipe_read_v);
            ++smem_pipe_read_v;
            // Non-WS refill: thread 0 issues TMA for the next V tile
            tma_refill.refill_v(tile_idx);
#if defined(PHASE_TRACE_GAPS)
            uint64_t _trace_pv_end = clock64();
            if (thread_idx == 0 && blockIdx.x == 0 && tile_idx < PHASE_TRACE_GAPS) {
                int _idx = _trace_count++;
                _trace_tile[_idx] = tile_idx;
                _trace_n[_idx] = n_block;
                _trace_prev_loop[_idx] = _trace_prev_pv_to_loop;
                _trace_prev_qk[_idx] = _trace_prev_pv_end == 0 ? 0 : _trace_qk_start - _trace_prev_pv_end;
                _trace_k_wait[_idx] = _trace_k_ready - _trace_k_wait_start;
                _trace_k_prep[_idx] = _trace_qk_start - _trace_k_ready;
                _trace_qk[_idx] = _trace_qk_end - _trace_qk_start;
                _trace_soft[_idx] = _trace_soft_end - _trace_qk_end;
                _trace_soft_to_pv_order[_idx] = _trace_pv_order_wait_start - _trace_soft_end;
                _trace_pv_order_wait[_idx] = _trace_pv_order_wait_end - _trace_pv_order_wait_start;
                _trace_v_wait[_idx] = _trace_v_ready - _trace_v_wait_start;
                _trace_v_copy_quant[_idx] = _trace_pv_gate_wait_start - _trace_v_ready;
                _trace_pv_gate_wait[_idx] = _trace_pv_omma_start - _trace_pv_gate_wait_start;
                _trace_soft_to_pv_omma[_idx] = _trace_pv_omma_start - _trace_soft_end;
                _trace_pv[_idx] = _trace_pv_end - _trace_pv_omma_start;
                _trace_total[_idx] = _trace_pv_end - _trace_loop_start;
            }
            _trace_prev_pv_end = _trace_pv_end;
#endif
#if defined(PHASE_TIMING)
            uint64_t _pt_pv_end = clock64();
            // Print per-tile phase timing from thread 0 of block 0, tile 1 only
            if (thread_idx == 0 && blockIdx.x == 0
                && n_block_count - 1 - n_block == 1) {
                printf("PHASE_TIMING wg=%d QK=%llu soft=%llu PV=%llu total=%llu\n",
                       wg_id,
                       (unsigned long long)(_pt_qk_end - _pt_qk_start),
                       (unsigned long long)(_pt_soft_end - _pt_qk_end),
                       (unsigned long long)(_pt_pv_end - _pt_soft_end),
                       (unsigned long long)(_pt_pv_end - _pt_qk_start));
            }
#endif
        }

#if defined(PHASE_TRACE_GAPS)
        if (thread_idx == 0 && blockIdx.x == 0) {
            for (int i = 0; i < _trace_count; ++i) {
                printf("PHASE_TRACE_GAPS wg=%d tile=%d n=%d "
                       "prev_pv_to_loop=%llu prev_pv_to_qk=%llu "
                       "k_wait=%llu k_prep=%llu qk=%llu soft=%llu "
                       "soft_to_pv_order=%llu pv_order_wait=%llu "
                       "v_wait=%llu v_copy_quant=%llu pv_gate_wait=%llu "
                       "soft_to_pv_omma=%llu pv=%llu total=%llu\n",
                       wg_id, _trace_tile[i], _trace_n[i],
                       (unsigned long long)(_trace_prev_loop[i]),
                       (unsigned long long)(_trace_prev_qk[i]),
                       (unsigned long long)(_trace_k_wait[i]),
                       (unsigned long long)(_trace_k_prep[i]),
                       (unsigned long long)(_trace_qk[i]),
                       (unsigned long long)(_trace_soft[i]),
                       (unsigned long long)(_trace_soft_to_pv_order[i]),
                       (unsigned long long)(_trace_pv_order_wait[i]),
                       (unsigned long long)(_trace_v_wait[i]),
                       (unsigned long long)(_trace_v_copy_quant[i]),
                       (unsigned long long)(_trace_pv_gate_wait[i]),
                       (unsigned long long)(_trace_soft_to_pv_omma[i]),
                       (unsigned long long)(_trace_pv[i]),
                       (unsigned long long)(_trace_total[i]));
            }
        }
#endif

        softmax_fused.finalize(tOrO_store);
        return;
    }

};

}  // namespace sage
