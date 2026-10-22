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
"""Ascend WF6-AF16 linear.

The generic implementation is ``flag_train.deepspeed.wf6af16_linear`` and is
what every backend without an override runs. None of it survives this
backend's compiler (BiShengHIR, CANN 9.0, measured on a 910B4):

* the WIDE_LOAD path casts the uint8 packed-weight pointers to
  ``tl.pointer_type(tl.uint32)``; BiShengHIR rejects the bitwidth-widening
  pointer cast outright (``Casting pointers with unmatched bitwidth!``), and
  uint32 is not a hardware dtype on A2/A3 at all. Worse, the generic wrapper's
  heuristic turns WIDE_LOAD on for exactly the decode token range, so the
  generic path crashes at *compile* time precisely on the shapes it was tuned
  to accelerate.
* the byte path compiles only at BLOCK_K=32 (anything larger blows the 192 KB
  unified buffer: the dequant chain's many (BLOCK_M, BLOCK_K) intermediates
  are each materialized, and multi-buffering doubles them) and then
  *miscomputes*: ``code = (top2 << 4) | low4`` combines two tensors that carry
  different ``tl.interleave`` layouts, and the layout conversion BiShengHIR
  inserts for the ``|`` (or ``+``) scrambles the values -- each operand stored
  on its own is correct, the combination is garbage.

This module therefore carries its own kernel with a restructured unpack that
avoids interleave-layout conversions entirely: the four columns packed into
each 2-bit-slice byte are treated as four independent K-streams end to end.
The 4-bit slice is read with two stride-2 byte loads (even/odd bytes), each
stream's 6-bit code is assembled and dequantized elementwise from tensors
that all descend from same-pattern strided loads, the activation tile is
read as four stride-4 streams to match, and the tile's dot is the sum of the
four per-stream dots -- no ``tl.interleave`` anywhere, so BiShengHIR never
has to convert between interleave layouts (which it gets wrong, and which
would also materialize the widest intermediates in the kernel). The split-K
plumbing and the reduction kernel are the generic code's.

The autotune space is what measured correct and fast on a 910B4 (per-config
sweep with autotune bypassed -- autotune scores speed, not correctness, and a
miscompiling config would surface only on whichever shape selects it). The
split-K choice is Ascend's own: powers of two keep every slice byte-aligned
by construction, which neither the DeepSpeed A100 profile map's odd factors
nor a misaligned user choice guarantee; the constraint below (``split_k * 4``
dividing K, plus a floor on slice length) is the generic launcher's.

Registration is by name. ``_ascend/ops/deepspeed/__init__.py`` re-exports this
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
    """Autotune candidates for the GEMM kernel on Ascend.

    Winners of a correctness-and-latency sweep over the benchmark shapes on a
    910B4 (see the module docstring): with split-K, (128, 16..64) weight tiles
    serve the decode shapes (1 token, K = M = 4096: 2.2 us at split-K 16),
    (128, 128) tiles serve prefill (61 us at 1024 tokens, 4096x4096). The
    (16, 16) entry keeps tiny shapes off the big tiles' masked-overhead path.
    Every candidate keeps BLOCK_K=64: the stream dots reduce over
    BLOCK_K // 4 = 16 columns, the cube unit's minimum, and larger BLOCK_K
    only grows the unified-buffer footprint without winning a shape.
    """
    configs = []
    for block_m, block_n, block_k, warps, stages in [
        (128, 16, 64, 4, 1),
        (128, 32, 64, 4, 1),
        (128, 64, 64, 4, 1),
        (128, 128, 64, 4, 1),
        (64, 64, 64, 4, 2),
        (16, 16, 64, 4, 1),
    ]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return configs


@triton.jit
def _dequant_fp6(code):
    """FP6-E3M2 codes to fp16: sign * 2^(e-3) * (1 + m/4), subnormals m*2^-4.

    Every FP6 value is exactly representable in fp16, so the cast is lossless
    and the later fp16 multiply by the scale rounds exactly the way the CUDA
    kernel's __hmul chain does.
    """
    e = (code >> 2) & 0x7
    mant = (code & 0x3).to(tl.float32)
    mag = tl.where(
        e == 0,
        mant * 0.0625,
        tl.exp2(e.to(tl.float32) - 3.0) * (1.0 + mant * 0.25),
    )
    return tl.where((code >> 5) & 1 == 1, -mag, mag).to(tl.float16)


@triton.jit
def _wf6af16_linear_gemm_ascend(
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

    Same contract as the generic
    ``flag_train.deepspeed.wf6af16_linear._wf6af16_linear_gemm`` minus the
    WIDE_LOAD path; the unpack is restructured into four interleave-free
    K-streams (see the module docstring). ``W`` is the packed FP6 weight
    ``[M, K]`` (2-bit + 4-bit slices), ``A`` the fp16 activation ``[N, K]``
    and ``C`` the fp16 output ``[N, M]``. With ``SPLIT_K_MODE`` each program
    covers a ``K / split_k`` slice and writes its fp32 partial to ``out_ptr``
    viewed as workspace ``[split_k, N, M]``; a follow-up reduction kernel
    produces the fp16 result.
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

    scale = tl.load(scale_ptr + offs_m, mask=m_mask, other=0.0)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        # The four columns packed into each 2-bit-slice byte are one K-stream
        # each; the loop walks them a quarter-tile (BLOCK_K // 4 columns) at a
        # time. offs_q is the first column of each 4-column group.
        offs_q = k0 + 4 * tl.arange(0, BLOCK_K // 4)

        # Unpack stream i's 6-bit codes (see the module docstring). The 2-bit
        # slice is one byte per 4 columns; the 4-bit slice's even bytes hold
        # the low nibbles of the first two of those columns and the odd bytes
        # the last two. Garbage codes from tail bytes beyond k_end are
        # harmless: the matching activation lanes are masked to zero, so the
        # dot cancels them.
        offs_b2 = k0 // 4 + tl.arange(0, BLOCK_K // 4)
        b2 = tl.load(
            w2_ptr + offs_m[:, None] * (K // 4) + offs_b2[None, :],
            mask=m_mask[:, None] & (offs_b2[None, :] * 4 < k_end),
            other=0,
        ).to(tl.int32)
        offs_b4 = k0 // 2 + 2 * tl.arange(0, BLOCK_K // 4)
        b4e = tl.load(
            w4_ptr + offs_m[:, None] * (K // 2) + offs_b4[None, :],
            mask=m_mask[:, None] & (offs_b4[None, :] * 2 < k_end),
            other=0,
        ).to(tl.int32)
        b4o = tl.load(
            w4_ptr + offs_m[:, None] * (K // 2) + offs_b4[None, :] + 1,
            mask=m_mask[:, None] & (offs_b4[None, :] * 2 + 2 < k_end),
            other=0,
        ).to(tl.int32)

        w0 = _dequant_fp6((((b2 >> 6) & 0x3) << 4) | ((b4e >> 4) & 0xF))
        w1 = _dequant_fp6((((b2 >> 4) & 0x3) << 4) | (b4e & 0xF))
        w2 = _dequant_fp6((((b2 >> 2) & 0x3) << 4) | ((b4o >> 4) & 0xF))
        w3 = _dequant_fp6(((b2 & 0x3) << 4) | (b4o & 0xF))
        w0 = w0 * scale[:, None]
        w1 = w1 * scale[:, None]
        w2 = w2 * scale[:, None]
        w3 = w3 * scale[:, None]

        a0 = tl.load(
            a_ptr + offs_n[:, None] * K + offs_q[None, :],
            mask=n_mask[:, None] & (offs_q[None, :] < k_end),
            other=0.0,
        )
        a1 = tl.load(
            a_ptr + offs_n[:, None] * K + offs_q[None, :] + 1,
            mask=n_mask[:, None] & (offs_q[None, :] + 1 < k_end),
            other=0.0,
        )
        a2 = tl.load(
            a_ptr + offs_n[:, None] * K + offs_q[None, :] + 2,
            mask=n_mask[:, None] & (offs_q[None, :] + 2 < k_end),
            other=0.0,
        )
        a3 = tl.load(
            a_ptr + offs_n[:, None] * K + offs_q[None, :] + 3,
            mask=n_mask[:, None] & (offs_q[None, :] + 3 < k_end),
            other=0.0,
        )

        acc = tl.dot(a0, tl.trans(w0), acc)
        acc = tl.dot(a1, tl.trans(w1), acc)
        acc = tl.dot(a2, tl.trans(w2), acc)
        acc = tl.dot(a3, tl.trans(w3), acc)

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


# The bare JIT function above is wrapped with the Ascend-verified autotune
# space; the test suite reads the configs back off this object.
wf6af16_linear_kernel = libentry()(
    triton.autotune(
        configs=_gemm_configs(),
        key=["M", "N", "K", "split_k"],
    )(_wf6af16_linear_gemm_ascend)
)


def _pick_split_k(tokens, out_channels, in_channels):
    """Ascend split-K choice for the decode shapes, 1 for anything larger.

    Measured on a 910B4: one-token decode over K = M = 4096 is fastest at
    split-K 16 (2.2 us against ~35 us unsplit), and the 32/64-token shapes at
    split-K 8. Powers of two keep every slice byte-aligned by construction;
    the constraint below is the same one the generic launcher documents (four
    codes per byte, so a slice must start on a multiple of 4 columns) plus a
    floor on slice length so the follow-up reduction does not eat the win.
    """
    if tokens > 64:
        return 1
    split_k = 16 if tokens <= 16 else 8
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

    Same contract as the generic ``flag_train.deepspeed.wf6af16_linear``;
    only the kernel variant, the tuning space, and the split-K choice differ
    (see the module docstring).
    """
    logger.debug("TRAIN_ASCEND WF6AF16_LINEAR")

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
