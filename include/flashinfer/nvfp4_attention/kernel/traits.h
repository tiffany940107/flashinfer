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

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "../common/cute_extension.h"  // Blackwell MMA types
#include "../quantization/fp4_layout.h"
#include "../primitives/barrier.cuh"

using namespace cute;

namespace sage {

/**
 * Shared Storage 定义
 *
 * 包含所有 kernel 需要的 shared memory 布局：
 * - Q, K, V 的数据和 scale factors
 * - Delta_s correction 值
 * - 输出 O
 * - Pipeline 和 Barrier 同步原语
 */
template <
    int kStages,        // Pipeline stages for K/V
    int EpiStages,      // Epilogue pipeline stages
    typename Element,   // FP4 element type (float_e2m1_t)
    typename ElementSF, // Scale factor type (float_ue4m3_t)
    typename OutputType,// Output type (bfloat16_t)
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutDS,  // Delta_s layout
    typename SmemLayoutO,
    typename SmemLayoutSFQ,
    typename SmemLayoutSFK,
    typename SmemLayoutSFV
>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0>{

    // Q 数据 (单 stage，因为只需要一份)
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;

    // K 数据 (多 stage，用于 pipeline)
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;

    // Scale factors for Q, K, V
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_SFV;

    // Delta_s correction 值 (多 stage)
    alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;

    // V 数据 (转置，多 stage)
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;

    // 输出 O
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;

    // 同步原语
    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename sage::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        // Ping-pong barrier: orders QK GEMM between Consumer0 and Consumer1
        alignas(16) typename sage::OrderedSequenceBarrier<2, 2>::SharedStorage math_order;
#if defined(STEADY_SPLIT_QK_PV_ORDER)
        // Optional split barrier: orders PV independently from QK for steady-state experiments.
        alignas(16) typename sage::OrderedSequenceBarrier<2, 2>::SharedStorage math_order_pv;
#endif
#if defined(STEADY_STRICT_PHASE_GATE) || defined(STEADY_SOFT_CREDIT_GATE)
        // Cross-tile phase gate:
        // channel 0 releases Consumer0's PV phase, channel 1 releases Consumer1's PV phase.
        alignas(16) cutlass::arch::ClusterBarrier strict_phase_gate[2];
#endif
        int tile_count_semaphore;
        // R18: hardware mbarrier for OMMA handoff between WG0 and WG1
        // r18_mbar[0]: WG0→WG1 (WG0 arrives after QK, WG1 waits before PV)
        // r18_mbar[1]: WG1→WG0 (WG1 arrives after QK, WG0 waits before PV)
        // Uses hardware try_wait (deschedules waiting warps, no starvation)
        alignas(16) cutlass::arch::ClusterBarrier r18_mbar[2];
        // Phase tracking for mbarrier wait side (persists across blocks)
        int r18_wait_phase[2];
    };
};

/**
 * Flash Attention Forward Kernel Traits
 *
 * 定义 kernel 的所有编译时配置，包括：
 * - Tile 形状 (M, N, K)
 * - Pipeline stages
 * - 数据类型
 * - MMA 指令配置 (OMMA for FP4)
 * - Shared memory layouts
 * - TMA copy 配置
 *
 * 模板参数：
 * @param kHeadDim_: Head dimension (必须是 32 的倍数)
 * @param kBlockM_: M 维度 tile 大小 (64 or 128)
 * @param kBlockN_: N 维度 tile 大小
 * @param kStages_: Pipeline stages for K/V
 * @param kClusterM_: Cluster size in M dimension
 * @param BlockMean_: 是否使用 block mean
 * @param ElementPairType_: FP4 pair type
 * @param ElementOut_: Output type (bfloat16/float16)
 */
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kStages_,
    int kClusterM_,
    bool BlockMean_,
    typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits {
    // ============ 基本配置 ============
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;  // 是否对 Q 进行 smooth

    // 约束检查
    static_assert(kHeadDim % 32 == 0, "Head dim must be multiple of 32");
    static_assert(kBlockM == 64 || kBlockM == 128, "BlockM must be 64 or 128");

    // ============ Warp 配置 ============
#if defined(NON_WS)
  #if defined(SPLIT_Q)
    // Split-Q: 2 groups × 128 threads, each processes 64 M-rows
    static constexpr int kNWarps = 8;  // 8 warps total
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;  // 256
    static constexpr int kBlockMPerWG = kBlockM / 2;  // 64 per group (same as WS)
    // AtomLayout: keep <_8> for full-CTA operations (pipeline, epilogue)
    // Group-local MMA uses <_4> defined in mainloop_splitq.cuh
  #else
    // Non-WS full-block: 8 warps, 128 M-rows
    static constexpr int kNWarps = 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;  // 256
    // Non-WS: all 8 warps process full kBlockM (no WG split)
    static constexpr int kBlockMPerWG = kBlockM;  // 128 (full block)
  #endif
#else
    // WS: kBlockM=128 时用 12 warps, kBlockM=64 时用 8 warps
    static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    // Ping-pong: 每个 WG 独立处理 kBlockMPerWG M-rows
    static constexpr int kBlockMPerWG = kBlockM / 2;  // 64
#endif

    // ============ Cluster 和 Pipeline ============
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;  // Epilogue stages

    // ============ Scale Factor 配置 ============
    // FP4 quantization: 每 16 个元素共享一个 scale factor
    static constexpr int NumSFQK = kHeadDim / 16;   // Q/K 的 SF 数量
    static constexpr int NumSFPV = kBlockN / 16;    // P/V 的 SF 数量
    static constexpr auto SFVectorSize = 16;        // SF vector size

    // ============ 数据类型 ============
    using ElementSF = cutlass::float_ue4m3_t;  // Scale factor: UE4M3
    using Element = cutlass::float_e2m1_t;     // Data: E2M1 (FP4)
    using ElementAccum = float;                 // Accumulator: FP32
    using ElementOut = ElementOut_;             // Output: BF16/FP16
    using index_t = int64_t;

    // ============ Tile 形状 ============
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;

    // Permutation tile sizes (用于 OMMA 指令)
    using PermTileM = Int<kBlockMPerWG>;  // 64 (WS) or 128 (Non-WS)
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;

    // ============ MMA 配置 ============
    // OMMA (FP4 Block GEMM) 输入元素类型
    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

#if defined(NON_WS) && !defined(SPLIT_Q)
    // Non-WS full-block: 8 atoms in M (8 warps × 16M/atom = 128M total)
    using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;
#else
    // WS or SPLIT_Q: 4 atoms in M per group (4 warps × 16M/atom = 64M)
    using AtomLayoutMNK = Layout<Shape<_4, _1, _1>>;
#endif

    // Q@K GEMM 的 Tiled MMA (使用 OMMA - FP4 Block GEMM)
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
      ));

    // P@V GEMM 的 Tiled MMA (同样使用 OMMA)
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, _32, PermTileK>{}
      ));

    // Full 128-row MMA (8 atoms, 256 threads).
    // Used for SFQ partitioning and epilogue O store.
    // With consumer_thread_idx 0-255, WG1 maps to atoms 0-3 (rows 0-63),
    // WG2 maps to atoms 4-7 (rows 64-127).
    using AtomLayoutMNK_Full = Layout<Shape<_8, _1, _1>>;
    using TiledMmaQK_Full = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK_Full{},
        Tile<Int<kBlockM>, PermTileN, PermTileK>{}
      ));
    using TiledMmaPV_Full = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK_Full{},
        Tile<Int<kBlockM>, _32, PermTileK>{}
      ));

    // MMA 的 scale factor 数量
    static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

    // ============ TMA 配置 ============
    using GmemTiledCopy = SM90_TMA_LOAD;    // 数据的 TMA load
    using GmemTiledCopySF = SM90_TMA_LOAD;  // Scale factor 的 TMA load

    // ============ Shared Memory Layouts ============

    // --- Q/K/V 数据的 smem layout ---
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<1>(TileShape_MNK{}))>());

    // Q layout (单 stage)
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    // K layout (多 stage)
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    // V layout (多 stage)
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    // V^T layout (transposed, 多 stage)
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // --- Delta_s layout ---
    using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
    using SmemLayoutDS =
        decltype(tile_to_shape(SmemLayoutAtomDS{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // ============ Shared Memory Copy Atoms ============
    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;

    // ============ Scale Factor Layouts ============
    using BlkScaledConfig = sage::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;

    // Q 的 scale factor layout
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));

    // K 的 scale factor layout
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));

    // V 的 scale factor layout
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));

    // V^T 的 scale factor layout
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));

    // P 的 scale factor layout
    using LayoutSFP = decltype(
      make_layout(
          make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
          make_stride(make_stride(_0{}, _1{}), _0{}, _4{})
      )
    );

    // P 的 layout
    using LayoutP = decltype(
      make_layout(
        make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
        make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
      )
    );

    // 完整的 smem layouts (包含 stages)
    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
      ));

    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))
      ));

    using SmemLayoutSFV = decltype(make_layout(
        append(shape(SmemLayoutAtomSFV{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFV{}), size(filter_zeros(SmemLayoutAtomSFV{})))
      ));

    using SmemLayoutSFVt = decltype(make_layout(
        append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFVt{}), size(filter_zeros(SmemLayoutAtomSFVt{})))
      ));

    // ============ 输出 O 的 Layout ============
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));
    // Per-WG half of O (64 M-rows × kHeadDim) for ping-pong mma_store
    using SmemLayoutO_Half = decltype(tile_to_shape(SmemLayoutAtomO{}, make_shape(Int<kBlockMPerWG>{}, Int<kHeadDim>{}), Step<_1, _2>{}));

    // ============ 完整的 Shared Storage ============
    using SharedStorage = SharedStorageQKVOwithSF<kStages, EpiStages, Element, ElementSF, ElementOut,
        SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutDS,
        SmemLayoutO, SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;

    // ============ Pipeline 类型 ============
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;

    // Q 单独的 pipeline (只有 1 stage)
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;

    // Epilogue barrier
    using EpilogueBarrier = typename sage::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;

    // Ping-pong math order barrier between Consumer0 and Consumer1
    using MathOrderBarrier = sage::OrderedSequenceBarrier<2, 2>;
};

} // namespace sage
