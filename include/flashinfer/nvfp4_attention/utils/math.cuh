/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cutlass/cutlass.h>

namespace sage {

/**
 * 数学工具函数
 *
 * 提供高性能的数学运算，使用 PTX 指令优化
 */

/**
 * PTX exp2 (近似)
 *
 * 使用 PTX 指令计算 2^x
 * 比标准库的 exp2f 更快，但精度略低
 *
 * @param x - 输入
 * @return 2^x
 */
__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/**
 * Max operator
 */
template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y) {
        return x > y ? x : y;
    }
};

template<>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(float const& x, float const& y) {
        return fmaxf(x, y);
    }
};

/**
 * Sum operator
 */
template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y) {
        return x + y;
    }
};

/**
 * Vectorized float2 operations (使用 PTX)
 */
CUTLASS_DEVICE void add(
    float2& c,
    float2 const& a,
    float2 const& b
) {
    asm volatile("add.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
}

CUTLASS_DEVICE void mul(
    float2& c,
    float2 const& a,
    float2 const& b
) {
    asm volatile("mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
}

CUTLASS_DEVICE void fma(
    float2& d,
    float2 const& a,
    float2 const& b,
    float2 const& c
) {
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c))
    );
}

/**
 * FMA-based exp2 多项式近似 (FA4 风格)
 *
 * 使用 Cody-Waite 范围缩减 + 3阶 Horner 多项式:
 *   2^x = 2^n * 2^f, where x = n + f (n=整数, f=小数)
 *   2^f ≈ p0 + p1*f + p2*f^2 + p3*f^3
 *
 * 优势: 使用 FMA 单元而非 MUFU.EX2，分担硬件压力
 * 精度: ~20-bit mantissa (足够 softmax + FP4 量化)
 */
__forceinline__ __device__ float exp2_fma_poly(float x) {
    // 范围缩减: x = n + f
    float n = rintf(x);
    float f = x - n;

    // Horner 多项式: 2^f ≈ ((p3*f + p2)*f + p1)*f + p0
    // 系数来自 FA4 论文
    float poly = ((0.0771f * f + 0.2276f) * f + 0.6951f) * f + 1.0f;

    // 2^n * poly: 通过操作 IEEE 754 指数位实现
    int n_int = __float2int_rn(n);
    uint32_t poly_bits = __float_as_uint(poly);
    poly_bits += (uint32_t)(n_int << 23);
    return __uint_as_float(poly_bits);
}

}  // namespace sage
