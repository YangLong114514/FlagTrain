# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP6-weight / FP16-activation linear, ported from DeepSpeed's cuda_wf6af16_linear.

The operator computes ``output = hidden_states @ (W * scales)^T`` where ``W``
is a weight matrix quantized to FP6-E3M2 (1 sign, 3 exponent, 2 mantissa bits,
exponent bias 3, no inf/NaN encodings) with one fp16 scale per output channel.

Weight storage follows the same 2+4 split as DeepSpeed (the FP6-LLM scheme):
the top 2 bits of each 6-bit code (sign + highest exponent bit) live in
``weights_2bit``, the low 4 bits in ``weights_4bit``. The byte-level layout
here is FlagTrain's own -- DeepSpeed interleaves bits across 32 threads to
feed MMA fragments, which buys nothing in a Triton kernel -- so packed tensors
must *not* come from DeepSpeed's ``preprocess_weight``. Packing order is
row-major and dense:

* ``weights_2bit[m, j]``: codes of columns ``4j..4j+3``, top-2 bits of column
  ``4j+i`` at bit positions ``6-2i``.
* ``weights_4bit[m, j]``: codes of columns ``2j..2j+1``, low-4 bits of column
  ``2j+i`` at bit positions ``4-4i``.

This module only carries the compute kernel; quantizing weights and packing
them into this layout is input preparation, and reference helpers for it live
with the tests (``tests/deepspeed/test_wf6af16_linear.py``).

Dequantization mirrors the CUDA kernel bit-for-bit at the value level:
``code -> sign * 2^(e-3) * (1 + m/4)`` (with the ``e == 0`` subnormal branch
``sign * m * 2^-4``), multiplied by the per-channel scale.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_train
from flag_train import runtime
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def _gemm_configs():
    """Autotune candidates for the GEMM kernel.

    Small-``BLOCK_N`` entries serve the decode shapes (few tokens, where the
    kernel is bandwidth-bound on the packed weights); the large tiles with
    deeper pipelining serve the compute-bound prefill shapes.
    """
    configs = []
    for block_m, block_n, block_k, warps, stages in [
        (64, 16, 64, 4, 4),
        (64, 32, 64, 4, 4),
        (64, 64, 64, 4, 4),
        (128, 64, 64, 4, 3),
        (64, 128, 64, 4, 4),
        (128, 128, 32, 8, 3),
        (128, 128, 64, 8, 3),
        (128, 256, 32, 8, 3),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return configs


def _gemm_do_bench():
    """Autotune bench function that skips configs exceeding platform limits.

    MetaX's driver limits per-thread private memory (4 KB by default) and
    rejects over-limit kernels at launch time with a plain ``RuntimeError``,
    which the stock autotuner does not treat as "skip this config", so one
    over-limit config would abort the whole tuning run. Report such configs
    as infinitely slow instead. Other vendors keep the default benchmarker.
    """
    if runtime.device.vendor_name != "metax":
        return None
    bench = triton.runtime.driver.active.get_benchmarker()

    def do_bench(kernel_call, quantiles):
        try:
            return bench(kernel_call, quantiles=quantiles)
        except RuntimeError as e:
            logger.debug("wf6af16_linear autotune: skipping config: %s", e)
            return [float("inf"), float("inf"), float("inf")]

    return do_bench


@libentry()
@triton.autotune(
    configs=_gemm_configs(),
    key=["M", "N", "K", "split_k"],
    do_bench=_gemm_do_bench(),
)
@triton.jit
def wf6af16_linear_kernel(
    out_ptr,
    a_ptr,
    w2_ptr,
    w4_ptr,
    scale_ptr,
    M,
    N,
    K,
    split_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K_MODE: tl.constexpr,
):
    """One tile of ``C = (W * scales) @ A^T`` with FP6 weights unpacked inline.

    ``W`` is the packed FP6 weight ``[M, K]`` (2-bit + 4-bit slices), ``A`` the
    fp16 activation ``[N, K]`` and ``C`` the fp16 output ``[N, M]``. With
    ``SPLIT_K_MODE`` each program covers a ``K / split_k`` slice and writes its
    fp32 partial to ``out_ptr`` viewed as workspace ``[split_k, N, M]``; a
    follow-up reduction kernel produces the fp16 result.
    """
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_k = pid % split_k
    pid_mn = pid // split_k
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    k_per_split = K // split_k  # the wrapper guarantees split_k divides K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end

        a = tl.load(
            a_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Unpack the 6-bit codes with compact byte loads: the 2-bit slice is
        # read BLOCK_K//4 bytes per row (top-2 bits of 4 consecutive codes per
        # byte, at bit offsets 6, 4, 2, 0) and the 4-bit slice BLOCK_K//2
        # bytes per row (low-4 bits of 2 consecutive codes, at offsets 4, 0);
        # tl.interleave then restores the column order. Garbage codes from
        # tail bytes beyond k_end are harmless: the activation is zero there,
        # so the dot product cancels them.
        offs_b2 = k0 // 4 + tl.arange(0, BLOCK_K // 4)
        b2 = tl.load(
            w2_ptr + offs_m[:, None] * (K // 4) + offs_b2[None, :],
            mask=m_mask[:, None] & (offs_b2[None, :] * 4 < k_end),
            other=0,
        ).to(tl.int32)
        top2 = tl.interleave(
            tl.interleave((b2 >> 6) & 0x3, (b2 >> 2) & 0x3),
            tl.interleave((b2 >> 4) & 0x3, b2 & 0x3),
        )

        offs_b4 = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        b4 = tl.load(
            w4_ptr + offs_m[:, None] * (K // 2) + offs_b4[None, :],
            mask=m_mask[:, None] & (offs_b4[None, :] * 2 < k_end),
            other=0,
        ).to(tl.int32)
        low4 = tl.interleave((b4 >> 4) & 0xF, b4 & 0xF)

        code = (top2 << 4) | low4

        # FP6-E3M2 dequantization, bias 3, exponent 7 kept as a normal value:
        # value = sign * 2^(e-3) * (1 + m/4), subnormal branch m * 2^-4. Every
        # FP6 value is exactly representable in fp16, so the fp16 cast below
        # is lossless and the fp16 multiply by the scale rounds exactly the
        # way the CUDA kernel's __hmul chain does.
        e = (code >> 2) & 0x7
        mant = (code & 0x3).to(tl.float32)
        mag = tl.where(
            e == 0,
            mant * 0.0625,
            tl.exp2(e.to(tl.float32) - 3.0) * (1.0 + mant * 0.25),
        )
        w = tl.where((code >> 5) & 1 == 1, -mag, mag).to(tl.float16)

        scale = tl.load(scale_ptr + offs_m, mask=m_mask, other=0.0)
        w = w * scale[:, None]

        acc = tl.dot(a, tl.trans(w), acc)

    if SPLIT_K_MODE:
        tl.store(
            out_ptr + pid_k * N * M + offs_n[:, None] * M + offs_m[None, :],
            acc,
            mask=n_mask[:, None] & m_mask[None, :],
        )
    else:
        tl.store(
            out_ptr + offs_n[:, None] * M + offs_m[None, :],
            acc.to(tl.float16),
            mask=n_mask[:, None] & m_mask[None, :],
        )


@libentry()
@triton.jit
def wf6af16_split_k_reduce_kernel(
    ws_ptr,
    out_ptr,
    total,
    split_k,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum the ``split_k`` fp32 partials of ``[split_k, N, M]`` into fp16 output."""
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for s in range(split_k):
        acc += tl.load(ws_ptr + s * total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.float16), mask=mask)


# Profiled split-K choices from DeepSpeed's CUDAWf6Af16Linear (A100-80G),
# keyed by token chunk then output channel. Kept so the same shapes get the
# same partitioning the CUDA kernel was tuned with; values that do not divide
# K are dropped by the caller.
_SPLIT_K_MAP = [
    {
        3072: 18,
        4096: 13,
        5120: 10,
        6144: 9,
        8192: 6,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 7,
    },
    {
        3072: 9,
        4096: 6,
        5120: 5,
        6144: 9,
        8192: 3,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 6,
    },
    {
        3072: 6,
        4096: 4,
        5120: 7,
        6144: 3,
        8192: 2,
        10240: 5,
        14336: 5,
        28672: 5,
        57344: 4,
    },
    {
        3072: 9,
        4096: 3,
        5120: 5,
        6144: 2,
        8192: 5,
        10240: 4,
        14336: 8,
        28672: 6,
        57344: 4,
    },
    {
        3072: 7,
        4096: 5,
        5120: 2,
        6144: 5,
        8192: 4,
        10240: 1,
        14336: 3,
        28672: 3,
        57344: 4,
    },
    {
        3072: 3,
        4096: 2,
        5120: 5,
        6144: 3,
        8192: 1,
        10240: 8,
        14336: 3,
        28672: 4,
        57344: 3,
    },
    {
        3072: 5,
        4096: 7,
        5120: 3,
        6144: 5,
        8192: 7,
        10240: 3,
        14336: 1,
        28672: 1,
        57344: 3,
    },
    {
        3072: 2,
        4096: 5,
        5120: 4,
        6144: 1,
        8192: 5,
        10240: 2,
        14336: 6,
        28672: 4,
        57344: 1,
    },
    {
        3072: 2,
        4096: 3,
        5120: 1,
        6144: 1,
        8192: 3,
        10240: 3,
        14336: 3,
        28672: 1,
        57344: 1,
    },
    {
        3072: 5,
        4096: 4,
        5120: 1,
        6144: 4,
        8192: 2,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1,
    },
    {
        3072: 3,
        4096: 1,
        5120: 2,
        6144: 2,
        8192: 1,
        10240: 2,
        14336: 1,
        28672: 1,
        57344: 1,
    },
    {
        3072: 3,
        4096: 1,
        5120: 3,
        6144: 2,
        8192: 1,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1,
    },
]


def _pick_split_k(tokens, out_channels, in_channels):
    """Choose split-K the way DeepSpeed does, falling back to 1.

    The profiled map is consulted for small token counts; a candidate whose
    slices are not 4-column aligned is rejected because the kernel unpacks
    four codes per byte and each slice must start on a byte boundary.
    """
    split_k = -1
    if tokens <= 768:
        split_k = _SPLIT_K_MAP[(tokens - 1) // 64].get(out_channels, -1)
    if split_k <= 1 or in_channels % (split_k * 4) != 0:
        return 1
    return split_k


def wf6af16_linear(
    output,
    hidden_states,
    weights_2bit,
    weights_4bit,
    scale,
    out_channels=None,
    tokens=None,
    in_channels=None,
    split_k=None,
):
    """FP6-weight FP16-activation linear: ``output = hidden @ (W * scale)^T``.

    Port of DeepSpeed's ``cuda_wf6af16_linear`` operator. No bias, no
    activation fusion, no batched matmul -- same scope as the CUDA kernel.

    Args:
        output (Tensor): fp16 output, shape ``[tokens, out_channels]``.
        hidden_states (Tensor): fp16 activation, shape ``[tokens, in_channels]``.
        weights_2bit (Tensor): uint8 2-bit slice in the layout documented in
            the module docstring, shape ``[out_channels, in_channels // 4]``.
        weights_4bit (Tensor): uint8 4-bit slice in the layout documented in
            the module docstring, shape ``[out_channels, in_channels // 2]``.
        scale (Tensor): fp16 per-output-channel scales, shape ``[out_channels]``.
        out_channels (int, optional): inferred from ``scale`` when omitted.
        tokens (int, optional): inferred from ``hidden_states`` when omitted.
        in_channels (int, optional): inferred from ``hidden_states`` when omitted.
        split_k (int, optional): split-K factor; the profiled heuristic is
            used when omitted.

    Returns:
        Tensor: ``output``.
    """
    logger.debug("TRAIN WF6AF16_LINEAR")

    assert hidden_states.dtype == torch.float16, "activations must be fp16"
    assert output.dtype == torch.float16, "output must be fp16"
    assert scale.dtype == torch.float16, "scales must be fp16"
    assert weights_2bit.dtype == torch.uint8 and weights_4bit.dtype == torch.uint8
    assert (
        hidden_states.device.type == flag_train.device
    ), f"wf6af16_linear only supports {flag_train.device} tensors"

    tokens = tokens if tokens is not None else hidden_states.shape[0]
    in_channels = in_channels if in_channels is not None else hidden_states.shape[1]
    out_channels = out_channels if out_channels is not None else scale.shape[0]

    assert hidden_states.shape == (tokens, in_channels)
    assert output.shape == (tokens, out_channels)
    assert in_channels % 4 == 0, "in_channels must be a multiple of 4"
    assert weights_2bit.shape == (out_channels, in_channels // 4)
    assert weights_4bit.shape == (out_channels, in_channels // 2)

    if split_k is None:
        split_k = _pick_split_k(tokens, out_channels, in_channels)
    elif in_channels % (split_k * 4) != 0:
        # Each slice must start on a byte boundary (4 codes per byte).
        logger.debug(
            "wf6af16_linear: split_k=%d misaligns K=%d slices, using split_k=1",
            split_k,
            in_channels,
        )
        split_k = 1

    def grid(meta):
        return (
            triton.cdiv(out_channels, meta["BLOCK_M"])
            * triton.cdiv(tokens, meta["BLOCK_N"])
            * split_k,
        )

    with torch_device_fn.device(hidden_states.device):
        if split_k == 1:
            wf6af16_linear_kernel[grid](
                output,
                hidden_states,
                weights_2bit,
                weights_4bit,
                scale,
                out_channels,
                tokens,
                in_channels,
                split_k,
                SPLIT_K_MODE=False,
            )
        else:
            workspace = torch.empty(
                (split_k, tokens, out_channels),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            wf6af16_linear_kernel[grid](
                workspace,
                hidden_states,
                weights_2bit,
                weights_4bit,
                scale,
                out_channels,
                tokens,
                in_channels,
                split_k,
                SPLIT_K_MODE=True,
            )
            total = tokens * out_channels
            wf6af16_split_k_reduce_kernel[(triton.cdiv(total, 1024),)](
                workspace,
                output,
                total,
                split_k,
                BLOCK_SIZE=1024,
            )

    return output
