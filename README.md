# SM120 QK-MXFP8 / PV-NVFP4 Attention Kernel

This source-only branch contains the CUDA headers for the SM120/SM121
QK-MXFP8 / PV-NVFP4 attention kernel and a standalone Python usage and smoke
test:

```text
qk_mxfp8_pv_nvfp4_attention_sm120/
test_qk_mxfp8_pv_nvfp4_attention_sm120.py
```

## Requirements

- An NVIDIA SM120 or SM121 GPU
- PyTorch with CUDA support
- A FlashInfer build that integrates this kernel and exposes
  `flashinfer.sageattn_qk_fp8_pv_fp4`
- FP16 or BF16 Q, K, and V tensors
- Head dimension 128

The kernel-only branch does not contain the FlashInfer Python binding or build
system. Run the test in an environment containing the corresponding integrated
FlashInfer implementation.

## Python API

The one-call API follows the SageAttention-style interface:

```python
import torch
import flashinfer

batch_size = 1
num_qo_heads = 8
num_kv_heads = 2
seq_len = 256
head_dim = 128

q = torch.randn(
    batch_size,
    num_qo_heads,
    seq_len,
    head_dim,
    device="cuda",
    dtype=torch.bfloat16,
)
k = torch.randn(
    batch_size,
    num_kv_heads,
    seq_len,
    head_dim,
    device="cuda",
    dtype=torch.bfloat16,
)
v = torch.randn_like(k)

output = flashinfer.sageattn_qk_fp8_pv_fp4(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    smooth_k=True,
    return_lse=False,
)
```

Set `return_lse=True` when log-sum-exp is required:

```python
output, lse = flashinfer.sageattn_qk_fp8_pv_fp4(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=True,
    smooth_k=True,
    return_lse=True,
)
```

Supported layouts are:

- `HND`: `[batch, heads, sequence, head_dim]`
- `NHD`: `[batch, sequence, heads, head_dim]`

Grouped-query attention is supported when `num_qo_heads` is divisible by
`num_kv_heads`.

## Run the Smoke Test

The default command runs an HND, noncausal, output-only BF16 test. It compares
the output against an FP32 attention reference and reports inclusive latency.

```bash
python test_qk_mxfp8_pv_nvfp4_attention_sm120.py
```

Test NHD layout, causal masking, and LSE output:

```bash
python test_qk_mxfp8_pv_nvfp4_attention_sm120.py \
  --layout NHD \
  --causal \
  --return-lse
```

Run a larger performance-oriented shape:

```bash
python test_qk_mxfp8_pv_nvfp4_attention_sm120.py \
  --batch-size 4 \
  --num-qo-heads 8 \
  --num-kv-heads 8 \
  --q-len 4096 \
  --kv-len 4096 \
  --skip-reference \
  --warmup 20 \
  --repeat 100
```

Use `--skip-reference` for long sequences to avoid constructing the quadratic
FP32 score matrix. It does not change the timed kernel path.

Other useful options include:

```text
--dtype {bfloat16,float16}
--no-smooth-k
--seed SEED
--help
```

## Test Output

The test checks and reports:

- Output shape, dtype, and finite values
- Cosine similarity, relative L2 error, and mean absolute error against FP32
- Maximum LSE error when `--return-lse` is enabled
- Inclusive latency and theoretical attention TFLOP/s

Inclusive latency measures the complete Sage-style call, including Q/K/V
preprocessing and quantization. It is not an attention-kernel-only measurement.
The script prints `PASS` after all enabled checks succeed.
