/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cutlass/layout/matrix.h"
#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace sage {

using namespace cute;

/**
 * FP4 Block-Scaled Layout Configuration
 *
 * FP4 量化使用 block-scaled 方案:
 * - 每个 block (64 elements) 共享一个 scale factor
 * - Scale factors 存储为 FP8 E4M3
 * - 数据存储为 FP4 E2M1
 *
 * Block Structure:
 * - Blk_MN = 64: 每个 block 包含 64 个元素
 * - Blk_SF = 4: 每 4 个 scale factors 组成一个单元
 *
 * 这个配置适用于 Hopper 架构的 OMMA (FP4 Block GEMM) 指令
 *
 * @tparam SFVecSize_ - Scale factor 向量大小
 */

/**
 * Block-Scaled 基本 Chunk
 *
 * 定义了 scale factor 的基本存储单元
 *
 * @tparam SFVecSize - Scale factor 向量大小
 * @tparam major - Major order (K-major by default)
 */
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct BlockScaledBasicChunk {
    // 每个 block 包含 64 个元素
    using Blk_MN = _64;

    // 4 个 scale factors 组成一个单元
    using Blk_SF = _4;

    // Scale factor atom layout
    // Shape: ((16, 4), (SFVecSize, 4))
    // Stride: ((16, 4), (0, 1))
    using SfAtom = Layout<
        Shape<Shape<_16, _4>, Shape<Int<SFVecSize>, _4>>,
        Stride<Stride<_16, _4>, Stride<_0, _1>>
    >;
};

/**
 * Block-Scaled 配置
 *
 * 管理 FP4 block-scaled 量化的 layout 配置
 *
 * 功能:
 * 1. 定义 scale factor 的 layout
 * 2. 提供 shared memory layout 推导
 * 3. 支持 Q/K/V 的不同形状
 *
 * @tparam SFVecSize_ - Scale factor 向量大小
 */
template<int SFVecSize_>
struct BlockScaledConfig {
    static constexpr int SFVecSize = SFVecSize_;
    static constexpr int MMA_NSF = 4;  // Scale factors per MMA

    using BlkScaledChunk = BlockScaledBasicChunk<SFVecSize>;
    using Blk_MN = _64;   // 64 elements per block
    using Blk_SF = _4;    // 4 scale factors per unit

    // Basic block shape and stride for M/N dimensions
    using mnBasicBlockShape = Shape<_16, _4>;
    using mnBasicBlockStride = Stride<_16, _4>;

    // Basic block shape and stride for K dimension
    using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0, _1>;

    // Scale factor atom layout
    using SfAtom = Layout<
        Shape<mnBasicBlockShape, kBasicBlockShape>,
        Stride<mnBasicBlockStride, kBasicBlockStride>
    >;

    // Global memory layout for scale factors
    using LayoutSF = decltype(
        blocked_product(
            SfAtom{},
            make_layout(
                make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0))
            )
        )
    );

    // Elements per block
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    // Shared memory stride for M/N dimensions
    using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));

    /**
     * Tile Scale Factor Atom to Q/K/V Shape
     *
     * 将 scale factor atom 扩展到实际的 Q/K/V 形状
     *
     * @param problem_shape - (Seqlen, Dim, HeadNum, Batch)
     * @return Tiled layout
     */
    template<class ProblemShape>
    CUTE_HOST_DEVICE
    static constexpr auto tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
        auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
        return tile_to_shape(
            SfAtom{},
            make_shape(Seqlen, Dim, HeadNum, Batch),
            Step<_2, _1, _3, _4>{}
        );
    }

    /**
     * Tile Scale Factor Atom to V^T Shape
     *
     * V^T 的形状是转置的
     *
     * @param problem_shape - (Dim, Seqlen, HeadNum, Batch)
     * @return Tiled layout
     */
    template<class ProblemShape>
    CUTE_HOST_DEVICE
    static constexpr auto tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
        auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
        return tile_to_shape(
            SfAtom{},
            make_shape(Dim, Seqlen, HeadNum, Batch),
            Step<_2, _1, _3, _4>{}
        );
    }

    /**
     * 推导 Shared Memory Layout for Scale Factors of Q
     *
     * 根据 tile shape 和 MMA 配置推导 SFQ 的 shared memory layout
     *
     * @param tiled_mma - Tiled MMA 对象
     * @param tileshape_mnk - Tile 形状 (M, N, K)
     * @return Shared memory layout
     */
    template<class TiledMma, class TileShape_MNK>
    CUTE_HOST_DEVICE
    static constexpr auto deduce_smem_layoutSFQ(
        TiledMma tiled_mma,
        TileShape_MNK tileshape_mnk
    ) {
        // K dimension shape
        using sSFQ_shapeK = decltype(
            prepend(
                make_shape(
                    Blk_SF{} / Int<MMA_NSF>{},
                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}
                ),
                kBasicBlockShape{}
            )
        );

        // M dimension shape
        using sSFQ_shapeM = decltype(
            prepend(size<0>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{})
        );

        // Strides
        using sSFQ_strideM = sSF_strideMN;
        using sSFQ_strideK = decltype(
            prepend(
                make_stride(
                    Int<MMA_NSF>{},
                    size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}
                ),
                kBasicBlockStride{}
            )
        );

        // Complete layout
        using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
        using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
        using SmemLayoutAtomSFQ = decltype(make_layout(sSFQ_shape{}, sSFQ_stride{}));

        return SmemLayoutAtomSFQ{};
    }

    /**
     * 推导 Shared Memory Layout for Scale Factors of K
     *
     * K 的形状是 (N, K)
     *
     * @param tiled_mma - Tiled MMA 对象
     * @param tileshape_mnk - Tile 形状
     * @return Shared memory layout
     */
    template<class TiledMma, class TileShape_MNK>
    CUTE_HOST_DEVICE
    static constexpr auto deduce_smem_layoutSFKV(
        TiledMma tiled_mma,
        TileShape_MNK tileshape_mnk
    ) {
        // K dimension shape
        using sSFK_shapeK = decltype(
            prepend(
                make_shape(
                    Blk_SF{} / Int<MMA_NSF>{},
                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}
                ),
                kBasicBlockShape{}
            )
        );

        // N dimension shape
        using sSFK_shapeN = decltype(
            prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{})
        );

        // Strides
        using sSFK_strideN = sSF_strideMN;
        using sSFK_strideK = decltype(
            prepend(
                make_stride(
                    Int<MMA_NSF>{},
                    size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}
                ),
                kBasicBlockStride{}
            )
        );

        // Complete layout
        using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
        using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
        using SmemLayoutAtomSFK = decltype(make_layout(sSFK_shape{}, sSFK_stride{}));

        return SmemLayoutAtomSFK{};
    }

    /**
     * 推导 Shared Memory Layout for Scale Factors of V^T
     *
     * V^T 的形状是 (K, N)
     *
     * @param tiled_mma - Tiled MMA 对象
     * @param tileshape_mnk - Tile 形状
     * @return Shared memory layout
     */
    template<class TiledMma, class TileShape_MNK>
    CUTE_HOST_DEVICE
    static constexpr auto deduce_smem_layoutSFVt(
        TiledMma tiled_mma,
        TileShape_MNK tileshape_mnk
    ) {
        // K dimension shape
        using sSFVt_shapeK = decltype(
            prepend(
                make_shape(
                    Blk_SF{} / Int<MMA_NSF>{},
                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}
                ),
                kBasicBlockShape{}
            )
        );

        // N dimension shape (for V^T)
        using sSFVt_shapeN = decltype(
            prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{})
        );

        // Strides
        using sSFVt_strideN = sSF_strideMN;
        using sSFVt_strideK = decltype(
            prepend(
                make_stride(
                    Int<MMA_NSF>{},
                    size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}
                ),
                kBasicBlockStride{}
            )
        );

        // Complete layout
        using sSFVt_shape = decltype(make_shape(sSFVt_shapeN{}, sSFVt_shapeK{}));
        using sSFVt_stride = decltype(make_stride(sSFVt_strideN{}, sSFVt_strideK{}));
        using SmemLayoutAtomSFVt = decltype(make_layout(sSFVt_shape{}, sSFVt_stride{}));

        return SmemLayoutAtomSFVt{};
    }
};

}  // namespace sage
