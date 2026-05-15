/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cute/tensor.hpp"

namespace sage {

using namespace cute;

/**
 * Layout 转换工具
 *
 * 提供 MMA layout 到其他 layout 的转换
 */

/**
 * 转换 MMA layout 为 reduction layout
 *
 * MMA layout: (MmaAtom, MmaM, MmaN)
 * Reduction layout: ((AtomM, MmaM), (AtomN, MmaN))
 *
 * 用于 softmax 的 row-wise reduction
 */
template<class Layout>
CUTLASS_DEVICE constexpr auto convert_to_reduction_layout(Layout mma_layout) {
    static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
    static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

    return make_layout(
        make_layout(get<0,1>(mma_layout), get<1>(mma_layout)),
        make_layout(get<0,0>(mma_layout), get<2>(mma_layout))
    );
}

/**
 * 转换 MMA layout 为 conversion layout
 *
 * 用于量化时的数据重排
 */
template<class Layout>
CUTLASS_DEVICE constexpr auto convert_to_conversion_layout(Layout mma_layout) {
    static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
    static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

    constexpr int MmaAtomN = size<0, 0>(mma_layout);
    constexpr int MmaAtomM = size<0, 1>(mma_layout);
    constexpr int MmaM = size<1>(mma_layout);
    constexpr int MmaN = size<2>(mma_layout);

    static_assert(MmaAtomN % 8 == 0, "MmaAtomN should be multiple of 8.");
    static_assert(MmaAtomM == 2, "MmaAtomM should be 2.");
    static_assert(MmaN % 2 == 0, "MmaN should be multiple of 2.");

    auto mma_n_division = zipped_divide(
        layout<2>(mma_layout), make_tile(_2{})
    );
    return make_layout(
        make_layout(layout<0,0>(mma_layout), make_layout(layout<0,1>(mma_layout), layout<0>(mma_n_division))),
        layout<1>(mma_layout), layout<1>(mma_n_division)
    );
}

}  // namespace sage
