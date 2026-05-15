/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace sage {

using namespace cute;

/**
 * Output Writer (使用 TMA)
 *
 * 职责:
 * 1. 将 attention 输出从 register 写入到 shared memory
 * 2. 使用 TMA 将输出从 shared memory 写入到 global memory
 * 3. 处理数据类型转换 (FP32 -> FP16/BF16)
 *
 * 特点:
 * - 使用 SM90 TMA Store 指令进行高效的内存写入
 * - 支持 FP16/BF16 输出格式
 * - 自动处理 swizzled shared memory layout
 *
 * 模板参数:
 * @tparam Traits - Kernel 配置 traits
 */
template<typename Traits>
struct OutputWriter {

    using Element = typename Traits::ElementOut;
    using TileShape_MNK = typename Traits::TileShape_MNK;
    using SmemLayoutO = typename Traits::SmemLayoutO;
    using SmemCopyAtomO = typename Traits::SmemCopyAtomO;
    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kNWarps = Traits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

    // 输出张量的形状和 stride
    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;

    // TMA copy 类型
    using TMA_O = decltype(make_tma_copy(
        GmemTiledCopyOTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), repeat_like(StrideO{}, int32_t(0)), StrideO{}),
        SmemLayoutO{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for O

    /**
     * 预取 TMA 描述符到 L2 cache
     *
     * 此函数从 epilogue_tma_ws.h:105-107 提取
     *
     * @param tma_store_O - TMA Store 描述符
     */
    template<typename TMA>
    __device__ __forceinline__ static void prefetch_tma_descriptor(TMA const& tma_store_O) {
        cute::prefetch_tma_descriptor(tma_store_O.get_tma_descriptor());
    }

    /**
     * 第一步: 将输出从 register 拷贝到 shared memory
     *
     * 此函数从 epilogue_tma_ws.h:109-129 提取
     *
     * 流程:
     * 1. 将输出从 FP32 转换为目标类型 (FP16/BF16)
     * 2. 使用 smem copy 将数据拷贝到 shared memory
     * 3. 执行 fence,确保 shared memory 写入对 TMA 可见
     *
     * @param shared_storage - Shared memory storage
     * @param tiled_mma - Tiled MMA 对象 (用于确定线程 layout)
     * @param tOrO - Register 中的输出 (FP32)
     * @param thread_idx - 当前线程索引
     */
    template<typename SharedStorage, typename TiledMma, typename FrgTensorO>
    __device__ __forceinline__ static void register_to_smem(
        SharedStorage& shared_storage,
        TiledMma const& tiled_mma,
        FrgTensorO const& tOrO,
        int thread_idx
    ) {
        // 1. 创建 shared memory tensor (使用 swizzled layout)
        Tensor sO = cute::as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(shared_storage.smem_o.begin()), SmemLayoutO{})
        );

        // 2. 创建 smem tiled copy
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

        // 3. 数据类型转换: FP32 -> Element (FP16/BF16)
        constexpr int numel = decltype(size(tOrO))::value;
        cutlass::NumericArrayConverter<Element, float, numel> convert_op;
        // HACK: 这里要求 tensor 是连续的
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel>*>(tOrO.data()));
        auto tOrO_out = make_tensor(make_rmem_ptr<Element>(&frag), tOrO.layout());

        // 4. 分区并拷贝到 shared memory
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);  // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum), PIPE_M, PIPE_N)
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        // 5. Fence: 确保 shared memory 写入完成,对 TMA 可见
        cutlass::arch::fence_view_async_shared();
    }

    /**
     * 第二步: 使用 TMA 将输出从 shared memory 写入到 global memory
     *
     * 此函数从 epilogue_tma_ws.h:131-179 提取
     *
     * 流程:
     * 1. 根据 work tile info 确定当前处理的 block 坐标
     * 2. 创建 global memory 和 shared memory 的 tensor 视图
     * 3. 使用 TMA 执行异步拷贝
     * 4. 发出 TMA store arrive 信号
     *
     * @param shared_storage - Shared memory storage
     * @param tma_store_O - TMA Store 描述符
     * @param shape_O - 输出的形状
     * @param stride_O - 输出的 stride
     * @param m_block - M 维度的 block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     */
    template<typename SharedStorage, typename TMA, typename Shape, typename Stride>
    __device__ __forceinline__ static void smem_to_gmem(
        SharedStorage& shared_storage,
        TMA const& tma_store_O,
        Shape const& shape_O,
        Stride const& stride_O,
        int m_block,
        int bidh,
        int bidb
    ) {
        // 1. 创建 shared memory tensor
        Tensor sO = cute::as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(shared_storage.smem_o.begin()), SmemLayoutO{})
        );

        // 2. 创建 global memory tensor 并选择当前的 tile
        Tensor mO = tma_store_O.get_tma_tensor(shape_O);
        Tensor gO = local_tile(
            mO(_, _, bidh, bidb),
            select<0, 2>(TileShape_MNK{}),
            make_coord(m_block, _0{})
        );  // (M, K)

        // 3. 分区 global 和 shared memory tensors
        auto block_tma_O = tma_store_O.get_slice(_0{});
        Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
        Tensor tOsO = block_tma_O.partition_S(sO);  // (TMA, TMA_M, TMA_K)

        // 4. 使用 TMA 拷贝: shared memory -> global memory
        cute::copy(tma_store_O, tOsO, tOgO);

        // 5. 发出 TMA store arrive 信号
        tma_store_arrive();
    }

    /**
     * 完整的输出写入流程
     *
     * 包括:
     * 1. Register -> Shared memory
     * 2. Shared memory -> Global memory (使用 TMA)
     *
     * @param shared_storage - Shared memory storage
     * @param tiled_mma - Tiled MMA 对象
     * @param tOrO - Register 中的输出
     * @param tma_store_O - TMA Store 描述符
     * @param shape_O - 输出形状
     * @param stride_O - 输出 stride
     * @param thread_idx - 线程索引
     * @param m_block - M block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     */
    template<
        typename SharedStorage,
        typename TiledMma,
        typename FrgTensorO,
        typename TMA,
        typename Shape,
        typename Stride
    >
    __device__ __forceinline__ static void run(
        SharedStorage& shared_storage,
        TiledMma const& tiled_mma,
        FrgTensorO const& tOrO,
        TMA const& tma_store_O,
        Shape const& shape_O,
        Stride const& stride_O,
        int thread_idx,
        int m_block,
        int bidh,
        int bidb
    ) {
        // 第一步: Register -> Shared memory
        register_to_smem(shared_storage, tiled_mma, tOrO, thread_idx);

        // 第二步: Shared memory -> Global memory (TMA)
        smem_to_gmem(
            shared_storage,
            tma_store_O,
            shape_O, stride_O,
            m_block, bidh, bidb
        );
    }

    /**
     * 等待所有 TMA stores 完成
     *
     * 此函数从 epilogue_tma_ws.h:181-184 提取
     *
     * 在 kernel 结束前调用,确保所有输出都已写入完成
     */
    __device__ __forceinline__ static void wait_all_stores() {
        tma_store_wait<0>();
    }

    /**
     * 将输出置零 (用于 padding 的 blocks)
     *
     * 此函数从 epilogue_tma_ws.h:187-218 提取
     *
     * 对于超出序列长度的 blocks,需要将输出置零
     * 这个函数不使用 TMA,而是使用普通的 global memory 写入
     *
     * @param ptr_O - 输出指针
     * @param shape_O - 输出形状
     * @param stride_O - 输出 stride
     * @param thread_idx - 线程索引
     * @param m_block - M block 索引
     * @param bidh - Batch head 索引
     * @param bidb - Batch 索引
     */
    template<typename Shape, typename Stride>
    __device__ __forceinline__ static void store_zero(
        Element* ptr_O,
        Shape const& shape_O,
        Stride const& stride_O,
        int thread_idx,
        int m_block,
        int bidh,
        int bidb
    ) {
        // 1. 创建 global memory tensor
        Tensor mO = make_tensor(make_gmem_ptr(ptr_O), shape_O, stride_O);
        Tensor gO = local_tile(
            mO(_, _, bidh, bidb),
            select<0, 2>(TileShape_MNK{}),
            make_coord(m_block, _0{})
        );  // (M, K)

        // 2. 创建 non-TMA gmem copy
        // 每个线程一次写入 128 bits (kGmemElemsPerLoad 个元素)
        static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
        static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
        static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;
        static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");

        using GmemLayoutAtom = Layout<
            cute::Shape <cute::Int<NumMmaThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
            cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>
        >;
        using GmemTiledCopyO = decltype(
            make_tiled_copy(
                Copy_Atom<DefaultCopy, Element>{},
                GmemLayoutAtom{},
                Layout<cute::Shape<cute::_1, cute::Int<kGmemElemsPerLoad>>>{}  // Val layout
            )
        );

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        // 3. 创建零值 fragment
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        clear(tOrO);  // 置零

        // 4. 创建 predicate tensor (处理边界情况)
        Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(shape_O);
        }

        // 5. 拷贝到 global memory (处理边界)
        // Clear_OOB_K 必须是 false,因为我们不想写入 gmem 的 OOB 区域
        copy_with_predicate<
            /*Is_even_MN=*/false,
            /*Is_even_K=*/false,
            /*Clear_OOB_MN=*/false,
            /*Clear_OOB_K=*/false
        >(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
            get<0>(shape_O) - m_block * kBlockM
        );
    }

private:
    /**
     * 辅助函数: 带 predicate 的拷贝
     *
     * 注意: 这个函数应该在 utils/ 中定义
     * 这里假设它存在
     */
    template<bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN, bool Clear_OOB_K,
             typename TiledCopy, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4>
    __device__ __forceinline__ static void copy_with_predicate(
        TiledCopy const& tiled_copy,
        Tensor1 const& src,
        Tensor2& dst,
        Tensor3 const& coord,
        Tensor4 const& pred,
        int max_m
    ) {
        // TODO: 实现实际的拷贝逻辑
        // 原始代码位置: sageattn3/blackwell/utils.h
    }
};

}  // namespace sage
