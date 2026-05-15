"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import flashinfer


_LONG_TEST_ENV = "FLASHINFER_NVFP4_ATTENTION_LONG_TEST"


def _long_tests_enabled():
    return os.environ.get(_LONG_TEST_ENV, "0").lower() in ("1", "true", "yes", "on")


def _require_sm120():
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip("SM120 GPU is required")


def _pad_seqlen_to_128(x):
    pad_len = (-x.shape[2]) % 128
    if pad_len == 0:
        return x.contiguous()
    return torch.nn.functional.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def _preprocess_qkv_ref(q, k, v):
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(_pad_seqlen_to_128, (q, k, v))
    batch, heads, seqlen, head_dim = q.shape
    q_grouped = q.reshape(batch, heads, seqlen // 128, 128, head_dim)
    qm = q_grouped.mean(dim=3)
    q = (q_grouped - qm.unsqueeze(3)).reshape(batch, heads, seqlen, head_dim).contiguous()
    qk_correction = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, qk_correction


def _reference_attention(q, k, v, causal):
    q, k, v, qk_correction = _preprocess_qkv_ref(q, k, v)
    sm_scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    scores = scores + qk_correction.repeat_interleave(128, dim=2) * sm_scale
    if causal:
        seqlen = q.shape[2]
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def _run_nvfp4_attention_accuracy_case(
    batch,
    heads,
    seqlen,
    head_dim,
    causal,
    cos_threshold,
    mean_abs_err_threshold,
):
    _require_sm120()

    torch.manual_seed(42)
    q = torch.randn((batch, heads, seqlen, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction = (
        flashinfer.nvfp4_attention_quantize_qkv(q, k, v)
    )

    out, lse = flashinfer.nvfp4_attention_fwd(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        sm_scale=head_dim**-0.5,
        causal=causal,
    )

    torch.cuda.synchronize()
    ref = _reference_attention(q, k, v, causal)[:, :, :seqlen, :]
    out = out[:, :, :seqlen, :]
    lse = lse[:, :, :seqlen]

    assert out.shape == ref.shape
    assert lse.shape == (batch, heads, seqlen)
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert not torch.isnan(lse).any()
    assert not torch.isinf(lse).any()

    mean_abs_err = (out.float() - ref.float()).abs().mean().item()
    cos_sim = F.cosine_similarity(out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
    assert mean_abs_err <= mean_abs_err_threshold
    assert cos_sim >= cos_threshold


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    (
        "batch",
        "heads",
        "seqlen",
        "head_dim",
        "causal",
        "cos_threshold",
        "mean_abs_err_threshold",
    ),
    [
        pytest.param(1, 4, 128, 64, False, 0.55, 0.20, id="s128-d64-noncausal"),
        pytest.param(1, 4, 256, 128, False, 0.70, 0.20, id="s256-d128-noncausal"),
        pytest.param(1, 4, 256, 128, True, 0.60, 0.20, id="s256-d128-causal"),
    ],
)
@torch.inference_mode()
def test_nvfp4_attention_sm120_accuracy(
    batch,
    heads,
    seqlen,
    head_dim,
    causal,
    cos_threshold,
    mean_abs_err_threshold,
):
    _run_nvfp4_attention_accuracy_case(
        batch,
        heads,
        seqlen,
        head_dim,
        causal,
        cos_threshold,
        mean_abs_err_threshold,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not _long_tests_enabled(),
    reason=f"set {_LONG_TEST_ENV}=1 to run long NVFP4 attention accuracy cases",
)
@pytest.mark.parametrize(
    (
        "batch",
        "heads",
        "seqlen",
        "head_dim",
        "causal",
        "cos_threshold",
        "mean_abs_err_threshold",
    ),
    [
        pytest.param(1, 1, 4096, 64, False, 0.55, 0.22, id="s4096-d64-noncausal"),
        pytest.param(1, 1, 4096, 128, True, 0.55, 0.22, id="s4096-d128-causal"),
        pytest.param(1, 1, 8192, 64, False, 0.50, 0.25, id="s8192-d64-noncausal"),
    ],
)
@torch.inference_mode()
def test_nvfp4_attention_sm120_long_accuracy(
    batch,
    heads,
    seqlen,
    head_dim,
    causal,
    cos_threshold,
    mean_abs_err_threshold,
):
    _run_nvfp4_attention_accuracy_case(
        batch,
        heads,
        seqlen,
        head_dim,
        causal,
        cos_threshold,
        mean_abs_err_threshold,
    )
