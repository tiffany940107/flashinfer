/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * This code is based on code from FlashAttention3, https://github.com/Dao-AILab/flash-attention
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
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

#include "cutlass/fast_math.h"
#include "cute/tensor.hpp"

namespace sage {

///////////////////////////////////////////////////////////////////////////////

/**
 * Single Tile Scheduler
 *
 * 最简单的调度器：每个 thread block 只处理一个 tile
 * 使用 blockIdx 直接映射到 (M, H, B) 坐标
 *
 * 适用场景：
 * - 调试和验证
 * - Tile 数量 >= SM 数量的情况
 *
 * Grid 维度：(num_blocks_m, num_head, num_batch)
 */
class SingleTileScheduler {

public:

    // Host 端参数
    struct Arguments {
        int const num_blocks_m;  // M 维度的 block 数量
        int const num_head;      // Head 数量
        int const num_batch;     // Batch 数量
        int const* tile_count_semaphore = nullptr;  // 未使用
    };

    // Device 端参数 (SingleTile 不需要额外参数)
    struct Params {};

    // 转换 host 参数到 device 参数
    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    // 获取 grid 维度
    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
    }

    /**
     * Work Tile Info
     *
     * 描述一个工作 tile 的信息：
     * - M_idx: M 维度的 block 索引
     * - H_idx: Head 索引
     * - B_idx: Batch 索引
     */
    struct WorkTileInfo {
        int M_idx = 0;
        int H_idx = 0;
        int B_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, H_idx, B_idx};
        }

        CUTLASS_DEVICE
        WorkTileInfo
        get_next_work(Params const& params) const {
            // SingleTile 模式下，没有下一个工作
            return {-1, -1, -1, false};
        }

    };

    // 获取初始工作 tile (直接使用 blockIdx)
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    // 获取下一个工作 tile (SingleTile 没有下一个)
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};

///////////////////////////////////////////////////////////////////////////////

/**
 * Static Persistent Tile Scheduler
 *
 * 静态持久化调度器：每个 thread block 处理多个 tiles
 * 使用固定的 stride (gridDim.x) 来跳转到下一个 tile
 *
 * 适用场景：
 * - Tile 数量 >> SM 数量
 * - 负载均衡良好的情况
 * - 减少 kernel 启动开销
 *
 * 优点：
 * - 简单高效
 * - 无需原子操作
 * - 可预测的负载分配
 *
 * Grid 维度：(num_sm)
 * 每个 block 的工作 tiles：tile_idx, tile_idx + gridDim.x, tile_idx + 2*gridDim.x, ...
 */
class StaticPersistentTileScheduler {

public:

    // Host 端参数
    struct Arguments {
        int const num_blocks_m;
        int const num_head;
        int const num_batch;
        int const* tile_count_semaphore = nullptr;  // 未使用
    };

    // Device 端参数
    struct Params {
        int total_blocks;                       // 总 tile 数量
        cutlass::FastDivmod m_block_divmod;     // M 维度的除法器
        cutlass::FastDivmod head_divmod;        // Head 维度的除法器
    };

    // 转换参数
    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks_m),
                cutlass::FastDivmod(args.num_head)};
    }

    // 获取 grid 维度 (只用一维，大小为 SM 数量)
    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    /**
     * Work Tile Info
     *
     * 只存储线性 tile 索引，通过 divmod 计算 (M, H, B) 坐标
     */
    struct WorkTileInfo {
        int tile_idx;  // 线性 tile 索引

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        /**
         * 从线性索引计算 (M, H, B) 坐标
         *
         * tile_idx = m_block + M_blocks * (head + H * batch)
         *
         * 通过两次 divmod 计算：
         * 1. tile_idx / M_blocks = (m_block, head + H * batch)
         * 2. (head + H * batch) / H = (head, batch)
         */
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
            return {m_block, bidh, bidb};
        }

    };

    // 获取初始工作 tile (使用 blockIdx.x)
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    // 获取下一个工作 tile (跳过 gridDim.x 个 tiles)
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

///////////////////////////////////////////////////////////////////////////////

/**
 * Dynamic Persistent Tile Scheduler
 *
 * 动态持久化调度器：使用原子操作动态分配 tiles
 * 适用于负载不均衡的场景
 *
 * 适用场景：
 * - Causal attention (不同 tile 的计算量差异大)
 * - 不规则的序列长度
 *
 * 优点：
 * - 更好的负载均衡
 * - 适应不同的计算量
 *
 * 缺点：
 * - 需要原子操作 (轻微开销)
 * - 负载分配不可预测
 *
 * Grid 维度：(num_sm)
 * 使用 tile_count_semaphore 原子计数器来分配工作
 *
 * 注意：当前实现复用了 StaticPersistentTileScheduler 的 WorkTileInfo
 * 实际的动态调度逻辑需要在 kernel 中使用 atomicAdd 来实现
 */
class DynamicPersistentTileScheduler {

public:

    // Host 端参数
    struct Arguments {
        int const num_blocks_m;
        int const num_head;
        int const num_batch;
        int const* tile_count_semaphore;  // 原子计数器指针
    };

    // Device 端参数
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod;
        cutlass::FastDivmod const head_divmod;
        int const* tile_count_semaphore;
    };

    // 转换参数
    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.num_blocks_m * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks_m),
                cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    // 获取 grid 维度
    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    // 复用 Static scheduler 的 WorkTileInfo
    using WorkTileInfo = StaticPersistentTileScheduler::WorkTileInfo;

    // 获取初始工作 tile
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    // 获取下一个工作 tile
    // 注意：实际动态调度需要在这里使用 atomicAdd(tile_count_semaphore)
    // 当前实现退化为静态调度
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

///////////////////////////////////////////////////////////////////////////////

/**
 * (已废弃) StaticPersistentTileSchedulerOld
 *
 * 旧版本的静态调度器，保留用于兼容性
 * 使用 FastDivmod 成员变量而不是 Params
 *
 * 不推荐使用，请使用 StaticPersistentTileScheduler
 */
class StaticPersistentTileSchedulerOld {

private:
  int current_work_linear_idx_;
  cutlass::FastDivmod const &m_block_divmod, &head_divmod;
  int const total_blocks;

public:
  struct WorkTileInfo {
    int M_idx = 0;
    int H_idx = 0;
    int B_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }

  };

public:

  CUTLASS_DEVICE explicit StaticPersistentTileSchedulerOld(
      cutlass::FastDivmod const &m_block_divmod_,
      cutlass::FastDivmod const &head_divmod_,
      int const total_blocks_) :
    m_block_divmod(m_block_divmod_), head_divmod(head_divmod_), total_blocks(total_blocks_) {

#if defined(__CUDA_ARCH__)
    current_work_linear_idx_ = blockIdx.x;
#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(int linear_idx) const {
    if (linear_idx >= total_blocks) {
      return WorkTileInfo::invalid_work_tile();
    }

    int M_idx, H_idx, B_idx;
    int quotient = m_block_divmod.divmod(M_idx, linear_idx);
    B_idx = head_divmod.divmod(H_idx, quotient);
    return {M_idx, H_idx, B_idx, true};
  }

  CUTLASS_DEVICE
  void
  advance_to_next_work() {
    current_work_linear_idx_ += int(gridDim.x);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  fetch_next_work() {
    WorkTileInfo new_work_tile_info;
    advance_to_next_work();
    new_work_tile_info = get_current_work();
    return new_work_tile_info;
  }

};

} // namespace sage
