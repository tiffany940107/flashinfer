"""SM100-specific output-alignment tests for CUTLASS NVFP4 GEMM."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, autotune, mm_fp4, nvfp4_quantize
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM100 GPU")
    if get_compute_capability(torch.device("cuda")) != (10, 0):
        pytest.skip("Requires an SM100 GPU")


def _make_guarded_output(
    m: int, n: int, dtype: torch.dtype, address_mod_32: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    element_size = torch.empty((), dtype=dtype).element_size()
    alignment_elements = 32 // element_size
    guard_elements = 2 * alignment_elements
    sentinel = -1234.0
    storage = torch.full(
        (m * n + 2 * guard_elements + alignment_elements,),
        sentinel,
        device="cuda",
        dtype=dtype,
    )

    offset = ((address_mod_32 - storage.data_ptr() % 32) % 32) // element_size
    while offset < guard_elements:
        offset += alignment_elements
    output = storage[offset : offset + m * n].view(m, n)

    assert output.is_contiguous()
    assert output.data_ptr() % 32 == address_mod_32
    assert output.stride(0) * output.element_size() % 32 == 0
    assert storage.numel() - offset - output.numel() >= guard_elements
    return storage, output, offset


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_sm100_store256_alignment_dispatch(out_dtype: torch.dtype) -> None:
    """Aligned Store256 and 16-byte-offset TMA fallback must be bit exact."""

    _skip_if_not_sm100()

    m, n, k = 256, 512, 512
    torch.manual_seed(100256)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(
        a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = torch.reciprocal(a_global_sf * b_global_sf).float()
    reference = torch.mm(a, b.T).float()

    aligned_storage, aligned, aligned_offset = _make_guarded_output(
        m, n, out_dtype, address_mod_32=0
    )
    fallback_storage, fallback, fallback_offset = _make_guarded_output(
        m, n, out_dtype, address_mod_32=16
    )

    AutoTuner.get().clear_cache()

    def invoke(output: torch.Tensor) -> torch.Tensor:
        with autotune(False):
            result = mm_fp4(
                a_fp4,
                b_fp4.T,
                a_sf,
                b_sf.T,
                alpha=alpha,
                out_dtype=out_dtype,
                out=output,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass",
                use_nvfp4=True,
            )
        assert result.data_ptr() == output.data_ptr()
        return result

    # Warm up JIT compilation, module loading, and workspace allocation before
    # capture. Both alignment paths are then captured and replayed together.
    invoke(aligned)
    invoke(fallback)
    torch.cuda.synchronize()
    aligned.zero_()
    fallback.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        aligned_result = invoke(aligned)
        fallback_result = invoke(fallback)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(aligned_result, fallback_result)
    cosine = F.cosine_similarity(
        reference.reshape(-1), aligned_result.float().reshape(-1), dim=0
    )
    assert cosine > 0.98

    for storage, output, offset in (
        (aligned_storage, aligned, aligned_offset),
        (fallback_storage, fallback, fallback_offset),
    ):
        sentinel = torch.tensor(-1234.0, device=storage.device, dtype=storage.dtype)
        assert torch.all(storage[:offset] == sentinel)
        assert torch.all(storage[offset + output.numel() :] == sentinel)
