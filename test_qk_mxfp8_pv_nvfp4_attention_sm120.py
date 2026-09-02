#!/usr/bin/env python3
"""
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Python usage and smoke test for the SM120 QK-MXFP8/PV-NVFP4 kernel.

This source-only branch contains the CUDA headers. Run this script in a
Python environment whose FlashInfer installation exposes
``flashinfer.sageattn_qk_fp8_pv_fp4``.

Examples:

  python test_qk_mxfp8_pv_nvfp4_attention_sm120.py
  python test_qk_mxfp8_pv_nvfp4_attention_sm120.py --layout NHD --causal --return-lse
  python test_qk_mxfp8_pv_nvfp4_attention_sm120.py --q-len 4096 --kv-len 4096 \
      --skip-reference --warmup 20 --repeat 100
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the FlashInfer Sage-style QK-MXFP8/PV-NVFP4 API on SM120."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-qo-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--q-len", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layout", choices=("HND", "NHD"), default="HND")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--no-smooth-k", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError(
            f"SM120 or SM121 is required, got SM{capability[0]}{capability[1]}"
        )

    try:
        import flashinfer
    except ImportError as error:
        raise RuntimeError(
            "Install FlashInfer and expose it through the active Python environment."
        ) from error
    if not hasattr(flashinfer, "sageattn_qk_fp8_pv_fp4"):
        raise RuntimeError(
            "The installed FlashInfer does not expose sageattn_qk_fp8_pv_fp4. "
            "Use a FlashInfer checkout that integrates this kernel."
        )
    return flashinfer


def make_inputs(args: argparse.Namespace):
    if args.head_dim != 128:
        raise ValueError("This kernel requires head_dim=128")
    if args.num_qo_heads % args.num_kv_heads:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if (
        min(
            args.batch_size,
            args.num_qo_heads,
            args.num_kv_heads,
            args.q_len,
            args.kv_len,
            args.warmup + 1,
            args.repeat,
        )
        <= 0
    ):
        raise ValueError("shape dimensions and repeat counts must be positive")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q_hnd = torch.randn(
        args.batch_size,
        args.num_qo_heads,
        args.q_len,
        args.head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    k_hnd = torch.randn(
        args.batch_size,
        args.num_kv_heads,
        args.kv_len,
        args.head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    v_hnd = torch.randn(
        args.batch_size,
        args.num_kv_heads,
        args.kv_len,
        args.head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    if args.layout == "HND":
        return q_hnd, k_hnd, v_hnd, q_hnd, k_hnd, v_hnd
    return (
        q_hnd.transpose(1, 2).contiguous(),
        k_hnd.transpose(1, 2).contiguous(),
        v_hnd.transpose(1, 2).contiguous(),
        q_hnd,
        k_hnd,
        v_hnd,
    )


def reference_attention(q, k, v, sm_scale: float, causal: bool):
    if q.shape[1] != k.shape[1]:
        group_size = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        q_index = torch.arange(q.shape[2], device=q.device)[:, None]
        kv_index = torch.arange(k.shape[2], device=q.device)[None, :]
        scores.masked_fill_(kv_index > q_index + k.shape[2] - q.shape[2], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float()), torch.logsumexp(scores, dim=-1)


def measure_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    flashinfer = require_environment()
    q, k, v, q_hnd, k_hnd, v_hnd = make_inputs(args)
    sm_scale = 1.0 / math.sqrt(args.head_dim)

    def run_kernel():
        return flashinfer.sageattn_qk_fp8_pv_fp4(
            q,
            k,
            v,
            tensor_layout=args.layout,
            is_causal=args.causal,
            sm_scale=sm_scale,
            smooth_k=not args.no_smooth_k,
            return_lse=args.return_lse,
        )

    result = run_kernel()
    if args.return_lse:
        output, lse = result
    else:
        output, lse = result, None

    if output.shape != q.shape or output.dtype != q.dtype:
        raise AssertionError(
            f"unexpected output: shape={output.shape}, dtype={output.dtype}"
        )
    if not torch.isfinite(output).all():
        raise AssertionError("kernel output contains NaN or Inf")
    if lse is not None:
        expected_lse_shape = (args.batch_size, args.num_qo_heads, args.q_len)
        if lse.shape != expected_lse_shape or lse.dtype != torch.float32:
            raise AssertionError(
                f"unexpected LSE: shape={lse.shape}, dtype={lse.dtype}"
            )
        if not torch.isfinite(lse).all():
            raise AssertionError("LSE contains NaN or Inf")

    print(
        f"device={torch.cuda.get_device_name()} shape=B{args.batch_size},"
        f"Hq{args.num_qo_heads},Hkv{args.num_kv_heads},Sq{args.q_len},"
        f"Sk{args.kv_len},D{args.head_dim} layout={args.layout} "
        f"causal={args.causal} return_lse={args.return_lse}"
    )
    print(f"output: shape={tuple(output.shape)} dtype={output.dtype}")

    if not args.skip_reference:
        reference, reference_lse = reference_attention(
            q_hnd, k_hnd, v_hnd, sm_scale, args.causal
        )
        output_hnd = output if args.layout == "HND" else output.transpose(1, 2)
        difference = output_hnd.float() - reference
        cosine = F.cosine_similarity(
            output_hnd.float().flatten(), reference.flatten(), dim=0
        ).item()
        relative_l2 = (
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
        ).item()
        mean_absolute_error = difference.abs().mean().item()
        print(
            f"accuracy: cosine={cosine:.6f} relative_l2={relative_l2:.6f} "
            f"mean_abs={mean_absolute_error:.6f}"
        )
        if cosine < 0.985 or relative_l2 > 0.20 or mean_absolute_error > 0.07:
            raise AssertionError("output accuracy is outside the expected range")
        if lse is not None:
            lse_max_error = (lse - reference_lse).abs().max().item()
            print(f"lse_max_error={lse_max_error:.6f}")
            if lse_max_error > 0.15:
                raise AssertionError("LSE accuracy is outside the expected range")

    latency_ms = measure_ms(run_kernel, args.warmup, args.repeat)
    if args.causal:
        valid_pairs = sum(
            max(0, min(args.kv_len, row + args.kv_len - args.q_len + 1))
            for row in range(args.q_len)
        )
    else:
        valid_pairs = args.q_len * args.kv_len
    flops = 4 * args.batch_size * args.num_qo_heads * valid_pairs * args.head_dim
    tflops = flops / latency_ms / 1.0e9
    print(f"inclusive latency={latency_ms:.6f} ms, attention={tflops:.1f} TFLOP/s")
    print("PASS")


if __name__ == "__main__":
    main()
