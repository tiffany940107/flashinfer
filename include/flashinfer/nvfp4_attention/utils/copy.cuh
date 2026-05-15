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
#include "cute/tensor.hpp"

namespace sage {

using namespace cute;

/**
 * 带边界检查的 Copy 操作
 *
 * 用于将数据从源 tensor 复制到目标 tensor，支持：
 * - MN 维度的边界检查（行方向）
 * - K 维度的边界检查（列方向）
 * - Out-of-bounds 区域的清零
 *
 * @tparam Is_even_MN 是否 MN 维度对齐（无需检查）
 * @tparam Is_even_K 是否 K 维度对齐（无需检查）
 * @tparam Clear_OOB_MN 是否清零 out-of-bounds 的 MN 元素
 * @tparam Clear_OOB_K 是否清零 out-of-bounds 的 K 元素
 *
 * @param tiled_copy TiledCopy 对象
 * @param S 源 tensor (rank-3: MMA, MMA_M, MMA_K)
 * @param D 目标 tensor (rank-3: MMA, MMA_M, MMA_K)
 * @param identity_MN Identity tensor 用于获取实际坐标
 * @param predicate_K K 维度的 predicate（是否在范围内）
 * @param max_MN MN 维度的最大有效值
 */
template <
    bool Is_even_MN = true,
    bool Is_even_K = true,
    bool Clear_OOB_MN = false,
    bool Clear_OOB_K = true,
    typename TiledCopy,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3
>
CUTLASS_DEVICE void copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &S,
    Tensor<Engine1, Layout1> &D,
    Tensor<Engine2, Layout2> const &identity_MN,
    Tensor<Engine3, Layout3> const &predicate_K,
    const int max_MN = 0
) {
    // 约束检查
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K

    // 逻辑约束：如果要清零 MN 则必须清零 K
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K),
                  "Cannot clear OOB_MN without clearing OOB_K");

    // 遍历 M 维度
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // 检查 M 维度是否在范围内
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            // 遍历 K 维度
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                // 检查 K 维度是否在范围内
                if (Is_even_K || predicate_K(k)) {
                    // 在范围内，执行复制
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    // K 维度越界且需要清零
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            // M 维度越界且需要清零整行
            cute::clear(D(_, m, _));
        }
    }
}

}  // namespace sage
