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
 * Warp Group 操作
 *
 * Warp Group 是 Hopper 架构的新概念
 * 多个 warps 组成一个 group 协同工作
 *
 * 典型配置:
 * - 1 Warp Group = 4 Warps = 128 Threads
 * - OMMA (FP4 Block GEMM) 需要 warp group 执行
 *
 * 主要功能:
 * - 线程索引计算
 * - 数据分区
 * - Warp group 内同步
 */

/**
 * Warp Group 常量
 */
struct WarpGroupConstants {
    static constexpr int kThreadsPerWarp = 32;
    static constexpr int kWarpsPerGroup = 4;
    static constexpr int kThreadsPerGroup = kThreadsPerWarp * kWarpsPerGroup;  // 128
};

/**
 * Warp Group 索引计算
 *
 * 提供各种线程索引计算功能
 */
struct WarpGroupIndex {
    /**
     * 获取当前线程在 warp 中的 lane ID
     *
     * @return Lane ID (0-31)
     */
    __device__ __forceinline__ static int lane_id() {
        return threadIdx.x % WarpGroupConstants::kThreadsPerWarp;
    }

    /**
     * 获取当前线程所在的 warp ID (在 block 中)
     *
     * @return Warp ID
     */
    __device__ __forceinline__ static int warp_id() {
        return threadIdx.x / WarpGroupConstants::kThreadsPerWarp;
    }

    /**
     * 获取当前线程所在的 warp ID (在 warp group 中)
     *
     * @return Warp ID in group (0-3)
     */
    __device__ __forceinline__ static int warp_id_in_group() {
        return warp_id() % WarpGroupConstants::kWarpsPerGroup;
    }

    /**
     * 获取当前线程所在的 warp group ID
     *
     * @return Warp group ID
     */
    __device__ __forceinline__ static int warp_group_id() {
        return warp_id() / WarpGroupConstants::kWarpsPerGroup;
    }

    /**
     * 获取当前线程在 warp group 中的线程 ID
     *
     * @return Thread ID in warp group (0-127)
     */
    __device__ __forceinline__ static int thread_id_in_group() {
        return warp_id_in_group() * WarpGroupConstants::kThreadsPerWarp + lane_id();
    }

    /**
     * 判断当前线程是否是 warp 的第一个线程
     *
     * @return true if lane_id() == 0
     */
    __device__ __forceinline__ static bool is_first_lane() {
        return lane_id() == 0;
    }

    /**
     * 判断当前线程是否是 warp group 的第一个线程
     *
     * @return true if thread_id_in_group() == 0
     */
    __device__ __forceinline__ static bool is_first_thread_in_group() {
        return thread_id_in_group() == 0;
    }
};

/**
 * Warp Group 同步
 *
 * Warp group 内的线程同步操作
 */
struct WarpGroupSync {
    /**
     * Warp group 内同步
     *
     * 同步 warp group 内的所有线程
     */
    __device__ __forceinline__ static void sync() {
        __syncwarp();  // 同步 warp 内的所有线程
    }

    /**
     * Warp 内的 shuffle
     *
     * 在 warp 内交换数据
     *
     * @param var - 要交换的变量
     * @param src_lane - 源 lane ID
     * @return 从 src_lane 获取的值
     */
    template<typename T>
    __device__ __forceinline__ static T shuffle(T var, int src_lane) {
        return __shfl_sync(0xffffffff, var, src_lane);
    }

    /**
     * Warp 内的 XOR shuffle
     *
     * 与 XOR 后的 lane 交换数据
     *
     * @param var - 要交换的变量
     * @param lane_mask - XOR mask
     * @return 交换后的值
     */
    template<typename T>
    __device__ __forceinline__ static T shuffle_xor(T var, int lane_mask) {
        return __shfl_xor_sync(0xffffffff, var, lane_mask);
    }

    /**
     * Warp 内的 reduction (max)
     *
     * 计算 warp 内所有线程的最大值
     *
     * @param val - 当前线程的值
     * @return Warp 内的最大值
     */
    __device__ __forceinline__ static float reduce_max(float val) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val = fmaxf(val, shuffle_xor(val, mask));
        }
        return val;
    }

    /**
     * Warp 内的 reduction (sum)
     *
     * 计算 warp 内所有线程的和
     *
     * @param val - 当前线程的值
     * @return Warp 内的和
     */
    __device__ __forceinline__ static float reduce_sum(float val) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val += shuffle_xor(val, mask);
        }
        return val;
    }
};

/**
 * Warp Group 选举
 *
 * 在 warp 或 warp group 中选举一个代表线程
 */
struct WarpGroupElect {
    /**
     * Warp 内选举一个线程
     *
     * @return true if 当前线程被选中
     */
    __device__ __forceinline__ static bool elect_one_in_warp() {
        return cute::elect_one_sync();
    }

    /**
     * Warp group 内选举一个线程
     *
     * @return true if 当前线程被选中
     */
    __device__ __forceinline__ static bool elect_one_in_group() {
        // 只有每个 warp 的第一个线程参与选举
        bool is_first = WarpGroupIndex::is_first_lane();
        // 在所有 warps 的第一个线程中选举一个
        return is_first && (WarpGroupIndex::warp_id_in_group() == 0);
    }
};

/**
 * Warp Group Layout 辅助
 *
 * 用于计算 warp group 内的数据分区
 */
template<typename TiledMMA>
struct WarpGroupLayout {
    /**
     * 获取 warp group 的 MMA layout
     *
     * @param tiled_mma - Tiled MMA 对象
     * @param thread_idx - 线程索引
     * @return Thread MMA slice
     */
    __device__ __forceinline__ static auto get_thread_slice(
        TiledMMA const& tiled_mma,
        int thread_idx
    ) {
        return tiled_mma.get_thread_slice(thread_idx);
    }

    /**
     * 计算线程在 accumulator 中的位置
     *
     * 用于确定每个线程负责哪些输出元素
     *
     * @param thread_mma - Thread MMA slice
     * @param coord_tensor - 坐标 tensor
     * @return Partitioned 坐标 tensor
     */
    template<typename ThreadMMA, typename CoordTensor>
    __device__ __forceinline__ static auto partition_accumulator(
        ThreadMMA const& thread_mma,
        CoordTensor const& coord_tensor
    ) {
        return thread_mma.partition_C(coord_tensor);
    }
};

/**
 * Warp Group MMA 辅助类
 *
 * 封装 warp group 级别的 MMA 操作
 *
 * @tparam TiledMMA - Tiled MMA 类型
 */
template<typename TiledMMA>
struct WarpGroupMMA {
    using Traits = TiledMMA;

    /**
     * 获取 Tiled MMA 对象
     *
     * @return TiledMMA 对象
     */
    __device__ __forceinline__ static TiledMMA get_tiled_mma() {
        return TiledMMA{};
    }

    /**
     * 获取当前线程的 MMA slice
     *
     * @param thread_idx - 线程索引
     * @return Thread MMA slice
     */
    __device__ __forceinline__ static auto get_thread_slice(int thread_idx) {
        return get_tiled_mma().get_thread_slice(thread_idx);
    }

    /**
     * 执行 MMA (GEMM)
     *
     * @param A - A 矩阵 (register)
     * @param B - B 矩阵 (register)
     * @param C - C 矩阵 (register, accumulator)
     */
    template<typename TensorA, typename TensorB, typename TensorC>
    __device__ __forceinline__ static void gemm(
        TensorA const& A,
        TensorB const& B,
        TensorC& C
    ) {
        cute::gemm(get_tiled_mma(), A, B, C);
    }
};

/**
 * Canonical Warp Index
 *
 * 获取规范化的 warp 索引（跨 CTA 一致）
 *
 * @return Canonical warp index
 */
__device__ __forceinline__ int canonical_warp_idx() {
    return cutlass::canonical_warp_idx_sync();
}

}  // namespace sage
