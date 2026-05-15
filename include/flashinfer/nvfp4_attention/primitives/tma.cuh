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
 * TMA (Tensor Memory Accelerator) 操作封装
 *
 * TMA 是 Hopper 架构的硬件加速器，用于高效的内存传输
 *
 * 优势:
 * 1. 高带宽 - 接近 HBM 理论峰值
 * 2. 异步 - 不阻塞 SM，与计算重叠
 * 3. Multicast - 一次加载，多个 CTA 共享
 * 4. 自动处理 swizzled layout
 *
 * 主要操作:
 * - TMA Load: Global memory → Shared memory
 * - TMA Store: Shared memory → Global memory
 * - TMA Prefetch: 预取描述符到 L2 cache
 */

/**
 * TMA Load 辅助函数
 *
 * 从 global memory 加载数据到 shared memory
 *
 * @param tma_desc - TMA 描述符 (包含 source/dest 信息)
 * @param barrier - Barrier 指针 (用于同步)
 * @param mcast_mask - Multicast mask (0 表示不使用 multicast)
 * @param src - Source tensor (global memory)
 * @param dst - Destination tensor (shared memory)
 */
template<typename TMADesc, typename TensorSrc, typename TensorDst>
__device__ __forceinline__ void tma_load(
    TMADesc const& tma_desc,
    void* barrier,
    uint16_t mcast_mask,
    TensorSrc const& src,
    TensorDst& dst
) {
    // 使用 TMA copy，配合 barrier 和 multicast
    cute::copy(
        tma_desc.with(barrier, mcast_mask),
        src,
        dst
    );
}

/**
 * TMA Store 辅助函数
 *
 * 从 shared memory 存储数据到 global memory
 *
 * @param tma_desc - TMA 描述符
 * @param src - Source tensor (shared memory)
 * @param dst - Destination tensor (global memory)
 */
template<typename TMADesc, typename TensorSrc, typename TensorDst>
__device__ __forceinline__ void tma_store(
    TMADesc const& tma_desc,
    TensorSrc const& src,
    TensorDst& dst
) {
    // TMA Store 不需要 barrier (store 是 fire-and-forget)
    cute::copy(tma_desc, src, dst);
}

/**
 * TMA Store Arrive
 *
 * 发送 TMA store 完成信号
 * 必须在每次 TMA store 后调用
 */
__device__ __forceinline__ void tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/**
 * TMA Store Wait
 *
 * 等待所有 TMA stores 完成
 *
 * @tparam N - 等待的 store groups 数量 (0 表示等待全部)
 */
template<int N = 0>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

/**
 * TMA 描述符预取
 *
 * 将 TMA 描述符预取到 L2 cache
 * 应该在 kernel 开始时调用，减少首次访问延迟
 *
 * @param tma_desc - TMA 描述符
 */
template<typename TMADesc>
__device__ __forceinline__ void prefetch_tma_descriptor(TMADesc const& tma_desc) {
    cute::prefetch_tma_descriptor(tma_desc.get_tma_descriptor());
}

/**
 * TMA Loader 类
 *
 * 封装 TMA Load 相关操作的类
 * 提供更高层的接口
 *
 * @tparam TMADesc - TMA 描述符类型
 */
template<typename TMADesc>
struct TMALoader {
    TMADesc const& tma_desc_;

    /**
     * 构造函数
     *
     * @param tma_desc - TMA 描述符
     */
    __device__ __forceinline__
    TMALoader(TMADesc const& tma_desc) : tma_desc_(tma_desc) {}

    /**
     * 预取描述符
     */
    __device__ __forceinline__ void prefetch() const {
        sage::prefetch_tma_descriptor(tma_desc_);
    }

    /**
     * 加载数据
     *
     * @param barrier - Barrier 指针
     * @param mcast_mask - Multicast mask
     * @param src - Source tensor
     * @param dst - Destination tensor
     */
    template<typename TensorSrc, typename TensorDst>
    __device__ __forceinline__ void load(
        void* barrier,
        uint16_t mcast_mask,
        TensorSrc const& src,
        TensorDst& dst
    ) const {
        sage::tma_load(tma_desc_, barrier, mcast_mask, src, dst);
    }

    /**
     * 加载数据 (不使用 multicast)
     *
     * @param barrier - Barrier 指针
     * @param src - Source tensor
     * @param dst - Destination tensor
     */
    template<typename TensorSrc, typename TensorDst>
    __device__ __forceinline__ void load(
        void* barrier,
        TensorSrc const& src,
        TensorDst& dst
    ) const {
        sage::tma_load(tma_desc_, barrier, 0, src, dst);
    }
};

/**
 * TMA Storer 类
 *
 * 封装 TMA Store 相关操作的类
 *
 * @tparam TMADesc - TMA 描述符类型
 */
template<typename TMADesc>
struct TMAStorer {
    TMADesc const& tma_desc_;

    /**
     * 构造函数
     */
    __device__ __forceinline__
    TMAStorer(TMADesc const& tma_desc) : tma_desc_(tma_desc) {}

    /**
     * 预取描述符
     */
    __device__ __forceinline__ void prefetch() const {
        sage::prefetch_tma_descriptor(tma_desc_);
    }

    /**
     * 存储数据
     *
     * @param src - Source tensor (shared memory)
     * @param dst - Destination tensor (global memory)
     */
    template<typename TensorSrc, typename TensorDst>
    __device__ __forceinline__ void store(
        TensorSrc const& src,
        TensorDst& dst
    ) const {
        sage::tma_store(tma_desc_, src, dst);
    }

    /**
     * 发送 store 完成信号
     */
    __device__ __forceinline__ void arrive() const {
        sage::tma_store_arrive();
    }

    /**
     * 等待所有 stores 完成
     *
     * @tparam N - 等待的 groups 数量
     */
    template<int N = 0>
    __device__ __forceinline__ void wait() const {
        sage::tma_store_wait<N>();
    }
};

/**
 * 创建 TMA Loader
 *
 * 工厂函数，方便创建 TMALoader
 *
 * @param tma_desc - TMA 描述符
 * @return TMALoader 对象
 */
template<typename TMADesc>
__device__ __forceinline__ auto make_tma_loader(TMADesc const& tma_desc) {
    return TMALoader<TMADesc>(tma_desc);
}

/**
 * 创建 TMA Storer
 *
 * 工厂函数，方便创建 TMAStorer
 *
 * @param tma_desc - TMA 描述符
 * @return TMAStorer 对象
 */
template<typename TMADesc>
__device__ __forceinline__ auto make_tma_storer(TMADesc const& tma_desc) {
    return TMAStorer<TMADesc>(tma_desc);
}

}  // namespace sage
