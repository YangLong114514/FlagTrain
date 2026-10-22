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
"""MetaX-tuned WF6-AF16 linear.

The generic implementation is ``flag_train.deepspeed.wf6af16_linear`` and is
what every backend without an override runs. This module carries its own copy
of the kernel with three changes that are worth 1.5-4.5x on MetaX's C550,
plus its own tuning around it:

* the activation tile is masked against ``K`` (a plain kernel argument) rather
  than ``k_end`` (a computed scalar). The two masks accept exactly the same
  lanes -- with ``split_k == 1`` they are identical, and with split-K the
  overshoot into a neighbour slice's columns stays in-bounds and is cancelled
  by zeroing the tail codes (see the kernel comment) -- but the MACA backend
  emits far better code for the argument-bound mask: measured on the benchmark
  shapes, (512, 4096, 4096) with the same tile goes from 3270 us to 812 us,
  and (128, 4096, 14336) from ~2900 us to 532 us. As a side effect the load
  keeps a statically analyzable predicate, which also avoids the MACA
  pipeliner's miscompilation of loop-bound-dependent load masks (the reason
  this kernel family must run with ``num_stages=1`` here: with deeper
  pipelining the same binary intermittently produces wrong tiles).
* a MetaX split-K choice for the decode shapes. The generic heuristic replays
  DeepSpeed's A100-profiled map, whose odd factors (13, 18, ...) mostly fail
  this kernel's byte-alignment constraint and fall back to 1, leaving a
  single token on a handful of CTAs. Powers of two chosen by token count
  roughly halve the decode latency on the C550 (264 us -> 123 us at one
  token, K = M = 4096).
* word-wide weight loads (``WIDE_LOAD``) on the decode path: the packed
  slices are read as uint32 and the bytes extracted in registers, worth ~5%
  on the decode shapes. It is engaged only on the decode token range (on the
  wide prefill tiles it measures neutral to clearly worse) and only when
  every row and every split-K slice starts on a 4-byte boundary of both
  packed slices (K % 16 == 0 and the slice length a multiple of 16 columns);
  the byte path is the fallback.

The autotune space is the measured winners of a tile sweep on the benchmark
shapes rather than the generic list: small weight tiles with wide token tiles
((32, 128) blocks) for prefill, (64, 16) with split-K for decode. Every
candidate keeps ``num_stages=1``: deeper pipelining never won the sweep, so
there is nothing to gain by approaching the pipeliner bug.

A ``do_bench`` wrapper reports a launch-time ``RuntimeError`` as infinitely
slow: MetaX's driver caps per-thread private memory (4 KB by default) and
rejects over-limit kernels at launch with a plain ``RuntimeError``, which the
stock autotuner does not treat as "skip this config".

The reduction kernel is imported from the generic module; it has no MetaX
divergence.

Registration is by name. ``_metax/ops/deepspeed/__init__.py`` re-exports this
module's ``wf6af16_linear``, and ``SpecOpRegistrar`` writes it over the
same-named generic function in ``flag_train.deepspeed`` when that package is
imported -- the namespace the tests and the benchmark read the operator from.
"""

import logging

import torch
import triton
import triton.language as tl

import flag_train
from flag_train.deepspeed.wf6af16_linear import wf6af16_split_k_reduce_kernel
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def _gemm_configs():
    """Autotune candidates for the GEMM kernel on MetaX.

    Winners of a tile sweep over the benchmark shapes on a C550 (see the
    module docstring): (64, 16) tiles with split-K serve the decode shapes,
    (32, 128) tiles serve prefill. ``num_stages`` stays 1 everywhere --
    deeper pipelining neither won the sweep nor is safe from the MACA
    pipeliner's load-mask miscompilation.
    """
    configs = []
    for block_m, block_n, block_k, warps in [
        (64, 16, 64, 4),
        (16, 16, 64, 2),
        (32, 128, 64, 4),
        (32, 128, 64, 8),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                num_warps=warps,
                num_stages=1,
            )
        )
    return configs


def _gemm_do_bench():
    """Autotune bench function that skips configs the driver rejects.

    See the module docstring: over-limit kernels fail at launch with a plain
    ``RuntimeError``, which is reported as infinitely slow so tuning continues
    with the remaining candidates.
    """
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
    WIDE_LOAD: tl.constexpr,
):
    """MetaX variant of the generic kernel; see the module docstring.

    Identical to ``flag_train.deepspeed.wf6af16_linear._wf6af16_linear_gemm``
    except for the activation mask (``offs_k < K`` instead of
    ``offs_k < k_end``) and the explicit zeroing of tail codes that replaces
    the cancelled-by-zero-activation trick.
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

    k_per_split = K // split_k  # the launcher guarantees split_k divides K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Masked against K, not k_end: the only lanes this accepts beyond
        # k_end belong to the next slice (still in-bounds, finite values), and
        # their contribution is cancelled by the zeroed codes below. A mask
        # bound to a plain kernel argument compiles far better on MACA than
        # one bound to the computed k_end -- and keeps the load off the
        # pipeliner path that miscompiles loop-bound-dependent predicates.
        a = tl.load(
            a_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )

        # Unpack the 6-bit codes. With WIDE_LOAD the packed slices are read as
        # uint32 words (one word per 16 / 8 columns) and the bytes are
        # extracted in registers -- fewer, wider load instructions, worth
        # 5-7% on the decode shapes. The byte path below is the fallback for
        # K % 16 != 0 or split-K slices whose start is not word-aligned: a
        # word must never straddle two rows or two slices.
        if WIDE_LOAD:
            offs_w2 = k0 // 16 + tl.arange(0, BLOCK_K // 16)
            w2w = tl.load(
                tl.cast(w2_ptr, tl.pointer_type(tl.uint32))
                + offs_m[:, None] * (K // 16)
                + offs_w2[None, :],
                mask=m_mask[:, None] & (offs_w2[None, :] * 16 < k_end),
                other=0,
            ).to(tl.int32, bitcast=True)
            # byte b of word w covers columns 16w+4b .. 16w+4b+3; the nested
            # interleave restores that column order.
            b2 = tl.interleave(
                tl.interleave(w2w & 0xFF, (w2w >> 16) & 0xFF),
                tl.interleave((w2w >> 8) & 0xFF, (w2w >> 24) & 0xFF),
            )
        else:
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

        if WIDE_LOAD:
            offs_w4 = k0 // 8 + tl.arange(0, BLOCK_K // 8)
            w4w = tl.load(
                tl.cast(w4_ptr, tl.pointer_type(tl.uint32))
                + offs_m[:, None] * (K // 8)
                + offs_w4[None, :],
                mask=m_mask[:, None] & (offs_w4[None, :] * 8 < k_end),
                other=0,
            ).to(tl.int32, bitcast=True)
            b4 = tl.interleave(
                tl.interleave(w4w & 0xFF, (w4w >> 16) & 0xFF),
                tl.interleave((w4w >> 8) & 0xFF, (w4w >> 24) & 0xFF),
            )
        else:
            offs_b4 = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            b4 = tl.load(
                w4_ptr + offs_m[:, None] * (K // 2) + offs_b4[None, :],
                mask=m_mask[:, None] & (offs_b4[None, :] * 2 < k_end),
                other=0,
            ).to(tl.int32)
        low4 = tl.interleave((b4 >> 4) & 0xF, b4 & 0xF)

        # Codes past k_end (accepted activation lanes past the slice bound, or
        # a byte holding a mix of in-range and tail columns) are forced to
        # code 0, which dequantizes to exactly +0.0, so tail lanes add nothing
        # to the dot.
        code = tl.where(offs_k[None, :] < k_end, (top2 << 4) | low4, 0)

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


def _pick_split_k(tokens, out_channels, in_channels):
    """MetaX split-K choice for the decode shapes, 1 for anything larger.

    A one-token decode over K = M = 4096 measures fastest at split-K 8 on the
    C550 (123 us against 264 us unsplit); 64 tokens wants 4. Powers of two
    keep every slice byte-aligned by construction; the constraint below is the
    same one the generic launcher documents (four codes per byte, so a slice
    must start on a multiple of 4 columns) plus a floor on slice length so the
    follow-up reduction does not eat the win.
    """
    if tokens > 64:
        return 1
    split_k = 8 if tokens <= 16 else 4
    while split_k > 1 and (
        in_channels % (split_k * 4) != 0 or in_channels // split_k < 64
    ):
        split_k //= 2
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

    Same contract as the generic ``flag_train.deepspeed.wf6af16_linear``; only
    the kernel variant, the tuning space, and the split-K choice differ (see
    the module docstring).
    """
    logger.debug("TRAIN_METAX WF6AF16_LINEAR")

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
    # The kernel indexes raw pointers with row-major strides.
    for t in (output, hidden_states, weights_2bit, weights_4bit, scale):
        assert t.is_contiguous(), "wf6af16_linear requires contiguous tensors"

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

    # The word-wide weight loads need every row and every split-K slice to
    # start on a 4-byte boundary of both packed slices: K % 16 == 0 keeps the
    # rows aligned, and k_per_split % 16 == 0 keeps the slices aligned. They
    # only pay on the decode path's narrow (64, 16) tiles; on the wide prefill
    # tiles they measure neutral to clearly worse (e.g. 1024 -> 1211 us at
    # (256, 8192, 8192)), so the gate also requires the decode token range.
    k_per_split = in_channels // split_k
    wide_load = (
        tokens <= 64
        and in_channels % 16 == 0
        and k_per_split % 16 == 0
        and weights_2bit.data_ptr() % 4 == 0
        and weights_4bit.data_ptr() % 4 == 0
    )

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
                WIDE_LOAD=wide_load,
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
                WIDE_LOAD=wide_load,
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
