/*
 * Copyright (c) 2025 by SageAttention team.
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/int_tuple.hpp"
#include "cutlass/layout/matrix.h"

namespace qk_mxfp8_pv_nvfp4_attention {

using namespace cute;















template <int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct BlockScaledBasicChunk;




template <UMMA::Major major>
struct BlockScaledBasicChunk<16, major> {
  using Blk_MN = _64;
  using Blk_SF = _4;
  static constexpr int MMA_NSF = 4;

  using mnBasicBlockShape = Shape<_16, _4>;
  using mnBasicBlockStride = Stride<_16, _4>;
  using kBasicBlockShape = Shape<_16, _4>;
  using kBasicBlockStride = Stride<_0, _1>;


  using SfAtom = Layout<Shape<mnBasicBlockShape, kBasicBlockShape>,
                        Stride<mnBasicBlockStride, kBasicBlockStride>>;
};





template <UMMA::Major major>
struct BlockScaledBasicChunk<32, major> {
  using Blk_MN = _128;
  using Blk_SF = _4;
  static constexpr int MMA_NSF = 4;

  using mnBasicBlockShape = Shape<_32, _4>;
  using mnBasicBlockStride = Stride<_16, _4>;
  using kBasicBlockShape = Shape<_32, _4>;
  using kBasicBlockStride = Stride<_0, _1>;

  using SfAtom =
      Layout<Shape<Shape<_32, _4>, Shape<_32, _4>>, Stride<Stride<_16, _4>, Stride<_0, _1>>>;
};






template <int SFVecSize_>
struct BlockScaledConfig {
  static constexpr int SFVecSize = SFVecSize_;

  using BlkScaledChunk = BlockScaledBasicChunk<SFVecSize>;


  static constexpr int MMA_NSF = BlkScaledChunk::MMA_NSF;

  using Blk_MN = typename BlkScaledChunk::Blk_MN;
  using Blk_SF = typename BlkScaledChunk::Blk_SF;

  using mnBasicBlockShape = typename BlkScaledChunk::mnBasicBlockShape;
  using mnBasicBlockStride = typename BlkScaledChunk::mnBasicBlockStride;
  using kBasicBlockShape = typename BlkScaledChunk::kBasicBlockShape;
  using kBasicBlockStride = typename BlkScaledChunk::kBasicBlockStride;

  using SfAtom = typename BlkScaledChunk::SfAtom;


  using LayoutSF = decltype(blocked_product(
      SfAtom{}, make_layout(make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                            make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));


  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});


  using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));








  template <class ProblemShape>
  CUTE_HOST_DEVICE static constexpr auto tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
    auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch), Step<_2, _1, _3, _4>{});
  }








  template <class ProblemShape>
  CUTE_HOST_DEVICE static constexpr auto tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
    auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch), Step<_2, _1, _3, _4>{});
  }






  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFQ(TiledMma tiled_mma,
                                                               TileShape_MNK tileshape_mnk) {

    using sSFQ_shapeK =
        decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
                         kBasicBlockShape{}));


    using sSFQ_shapeM = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));


    using sSFQ_strideM = sSF_strideMN;
    using sSFQ_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{}, size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));


    using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
    using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
    using SmemLayoutAtomSFQ = decltype(make_layout(sSFQ_shape{}, sSFQ_stride{}));

    return SmemLayoutAtomSFQ{};
  }






  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFKV(TiledMma tiled_mma,
                                                                TileShape_MNK tileshape_mnk) {

    using sSFK_shapeK =
        decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
                         kBasicBlockShape{}));


    using sSFK_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));


    using sSFK_strideN = sSF_strideMN;
    using sSFK_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));


    using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
    using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
    using SmemLayoutAtomSFK = decltype(make_layout(sSFK_shape{}, sSFK_stride{}));

    return SmemLayoutAtomSFK{};
  }






  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFVt(TiledMma tiled_mma,
                                                                TileShape_MNK tileshape_mnk) {

    using sSFVt_shapeK =
        decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
                         kBasicBlockShape{}));


    using sSFVt_shapeN =
        decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));


    using sSFVt_strideN = sSF_strideMN;
    using sSFVt_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));


    using sSFVt_shape = decltype(make_shape(sSFVt_shapeN{}, sSFVt_shapeK{}));
    using sSFVt_stride = decltype(make_stride(sSFVt_strideN{}, sSFVt_strideK{}));
    using SmemLayoutAtomSFVt = decltype(make_layout(sSFVt_shape{}, sSFVt_stride{}));

    return SmemLayoutAtomSFVt{};
  }
};

}
