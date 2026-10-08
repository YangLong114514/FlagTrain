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
"""Correctness tests for the WF6-AF16 linear operator.

Two oracles, because they fail differently:

* ``wf6af16_ref`` -- a plain-torch composition of the same contract (unpack
  the codes, dequantize through the FP6-E3M2 value table, matmul in fp32). It
  is the primary check and runs on any device, so the operator is testable
  off NVIDIA.
* DeepSpeed's ``cuda_wf6af16_linear`` -- the operator this implementation
  ports. It is an independent implementation and its version is recorded in
  ``_DEEPSPEED_VERSION``, as tests/deepspeed/README.md asks for, but it needs
  the ``deepspeed`` package with its InferenceCore CUDA extension, and on a
  backend in ``_DEEPSPEED_BASELINE_VENDORS`` below it is *required* -- if it
  will not load there the module raises rather than losing the check quietly.
  On other backends it is an additional check that is skipped.

Checking both is not redundant: ``wf6af16_ref`` is the same arithmetic written
from the same reading of the kernel, so a shared misreading of the contract
would survive it. DeepSpeed's operator is the only oracle that can disagree
with that reading. Note the packed byte layouts differ (DeepSpeed interleaves
bits across threads for its MMA pipeline, FlagTrain packs densely row-major),
so the comparison feeds both sides the *same* FP6 codes, packed by each
side's own packer.
"""

import pytest
import torch

import flag_train
from flag_train.deepspeed import wf6af16_linear
from flag_train.deepspeed.wf6af16_linear import (
    _fp6_e3m2_table,
    pack_weights_fp6,
    quantize_weights_fp6,
)

from .. import accuracy_utils as utils

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def unpack_weights_fp6(weights_2bit, weights_4bit):
    """Inverse of ``pack_weights_fp6``, composed from plain torch ops."""
    out_channels, packed_k = weights_2bit.shape
    in_channels = packed_k * 4

    b2 = weights_2bit.to(torch.int32)
    top2 = torch.stack([(b2 >> s) & 0x3 for s in (6, 4, 2, 0)], dim=-1)
    top2 = top2.reshape(out_channels, in_channels)

    b4 = weights_4bit.to(torch.int32)
    low4 = torch.stack([(b4 >> s) & 0xF for s in (4, 0)], dim=-1)
    low4 = low4.reshape(out_channels, in_channels)

    return ((top2 << 4) | low4).to(torch.uint8)


def wf6af16_ref(hidden_states, weights_2bit, weights_4bit, scales):
    """Reference for the operator, composed from plain torch ops.

    Mirrors the kernel's arithmetic: FP6 values are exact in fp16, the scale
    multiply rounds to fp16 once (as the CUDA kernel's __hmul chain does), and
    the matmul accumulates in fp32.
    """
    codes = unpack_weights_fp6(weights_2bit, weights_4bit)
    table = _fp6_e3m2_table(device=codes.device)
    w = (table[codes.long()].to(torch.float16) * scales[:, None]).to(torch.float32)
    return (hidden_states.to(torch.float32) @ w.T).to(torch.float16)


_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's cuda_wf6af16_linear reference is unavailable; install the "
    "deepspeed package on a CUDA host to run this check."
)
_DEEPSPEED_AMPERE_ONLY_MSG = (
    "DeepSpeed's cuda_wf6af16_linear is Ampere-only (kernel_matmul.cuh asserts "
    "__CUDA_ARCH__ == 800); this device's compute capability is not sm_80, so "
    "the reference cannot execute here."
)


# Backends whose reference is DeepSpeed. cuda_wf6af16_linear ships in the
# InferenceCore CUDA extension, so only a backend that can compile and execute
# one can host it. On these the reference is not optional -- a missing one is
# an environment fault, and skipping quietly would thin the suite without
# saying so.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_wf6af16():
    """``(op, version)`` for DeepSpeed's cuda_wf6af16_linear, or ``(None, None)``.

    ``InferenceCoreBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under
    ``torch_extensions``. The returned callable takes
    ``(hidden_states, codes, scales)`` and returns the fp16 output, matching
    ``wf6af16_ref``'s signature so tests can swap oracles freely.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    # The CUDA kernel asserts __CUDA_ARCH__ == 800 at runtime, so on any other
    # architecture it cannot serve as an oracle no matter how well it builds.
    if flag_train.vendor_name == "nvidia":
        capability = torch.cuda.get_device_capability()
        if capability != (8, 0):
            return None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import InferenceCoreBuilder

        inf_module = InferenceCoreBuilder().load()
        inf_module.create_handle()
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's "
            f"cuda_wf6af16_linear as its reference, but it could not be "
            f"loaded: {exc!r}. Build deepspeed, or drop the backend from "
            f"_DEEPSPEED_BASELINE_VENDORS."
        ) from exc

    def deepspeed_wf6af16(hidden_states, weights_2bit, weights_4bit, scales):
        out_channels = scales.shape[0]
        tokens, in_channels = hidden_states.shape

        # DeepSpeed's preprocess_weight wants the fake-quantized fp16 weights
        # on CPU and re-packs them into its own interleaved layout. Feeding it
        # the exact FP6 values from the table keeps that cast lossless.
        codes = unpack_weights_fp6(weights_2bit.cpu(), weights_4bit.cpu())
        table = _fp6_e3m2_table()
        fake_fp16 = table[codes.long()].to(torch.float16)
        ds_w2, ds_w4 = inf_module.preprocess_weight(fake_fp16)

        device = hidden_states.device
        output = torch.empty((tokens, out_channels), dtype=torch.float16, device=device)
        # DeepSpeed picks split-K from a profiled map; 1 is always valid and
        # keeps the oracle on the simplest code path.
        workspace = torch.empty(
            (1, out_channels, tokens), dtype=torch.float32, device=device
        )
        inf_module.cuda_wf6af16_linear(
            output,
            hidden_states,
            ds_w2.to(device),
            ds_w4.to(device),
            scales.to(device),
            workspace,
            out_channels,
            tokens,
            in_channels,
            1,
        )
        return output

    return deepspeed_wf6af16, deepspeed.__version__


# Resolved once, at module import time.
_deepspeed_wf6af16, _DEEPSPEED_VERSION = _load_deepspeed_wf6af16()

# The torch reference is always available, so only the DeepSpeed checks skip.
if (
    _deepspeed_wf6af16 is None
    and flag_train.vendor_name == "nvidia"
    and torch.cuda.get_device_capability() != (8, 0)
):
    _ORACLE_SKIP_REASON = _DEEPSPEED_AMPERE_ONLY_MSG
else:
    _ORACLE_SKIP_REASON = _DEEPSPEED_UNAVAILABLE_MSG
requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_wf6af16 is None, reason=_ORACLE_SKIP_REASON
)


def _make_case(tokens, in_channels, out_channels, seed=0):
    """Random FP6 codes/scales/activations plus both packings of the weights."""
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 64, (out_channels, in_channels), generator=gen).to(
        torch.uint8
    )
    scales = (torch.rand(out_channels, generator=gen) + 0.5).to(torch.float16)
    hidden = torch.randn(tokens, in_channels, generator=gen).to(torch.float16)

    device = flag_train.device
    hidden = hidden.to(device)
    scales = scales.to(device)
    weights_2bit, weights_4bit = pack_weights_fp6(codes.to(device))
    return hidden, weights_2bit, weights_4bit, scales


def _run_train(hidden, weights_2bit, weights_4bit, scales, split_k=None):
    tokens, in_channels = hidden.shape
    out_channels = scales.shape[0]
    output = torch.empty(
        (tokens, out_channels), dtype=torch.float16, device=hidden.device
    )
    return wf6af16_linear(
        output,
        hidden,
        weights_2bit,
        weights_4bit,
        scales,
        out_channels,
        tokens,
        in_channels,
        split_k=split_k,
    )


# Shapes are (tokens, in_channels, out_channels). Deliberately not aligned to
# the CUDA kernel's 256/64 tiling: the Triton kernel masks its edges, and the
# torch reference has no alignment constraint at all. in_channels stays a
# multiple of 4 because the packing is defined in units of 4 codes.
_SHAPES = [
    (1, 64, 64),
    (3, 128, 48),
    (16, 256, 256),
    (33, 512, 384),
    (100, 1024, 512),
]


@pytest.mark.wf6af16_linear
@pytest.mark.parametrize("shape", _SHAPES)
def test_wf6af16_linear(shape):
    """The Triton kernel must match the torch reference on unaligned shapes."""
    tokens, in_channels, out_channels = shape
    hidden, weights_2bit, weights_4bit, scales = _make_case(*shape)

    train_out = _run_train(hidden, weights_2bit, weights_4bit, scales)
    ref_out = wf6af16_ref(hidden, weights_2bit, weights_4bit, scales)

    # atol scales with the reduction length, as in any fp16 GEMM comparison.
    utils.train_assert_close(
        utils.to_reference(train_out),
        utils.to_reference(ref_out),
        torch.float16,
        reduce_dim=in_channels,
    )


@pytest.mark.wf6af16_linear
@pytest.mark.parametrize("split_k", [1, 2, 4, 8])
def test_wf6af16_linear_split_k(split_k):
    """Split-K partials plus the reduction must agree with the single-pass run."""
    hidden, weights_2bit, weights_4bit, scales = _make_case(7, 512, 256)

    train_out = _run_train(hidden, weights_2bit, weights_4bit, scales, split_k=split_k)
    ref_out = wf6af16_ref(hidden, weights_2bit, weights_4bit, scales)

    utils.train_assert_close(
        utils.to_reference(train_out),
        utils.to_reference(ref_out),
        torch.float16,
        reduce_dim=512,
    )


@pytest.mark.wf6af16_linear
def test_pack_roundtrip():
    """Packing then unpacking must return the original codes."""
    codes = torch.randint(0, 64, (128, 512)).to(torch.uint8)
    weights_2bit, weights_4bit = pack_weights_fp6(codes)
    utils.train_assert_equal(unpack_weights_fp6(weights_2bit, weights_4bit), codes)


@pytest.mark.wf6af16_linear
def test_quantize_weights_fp6():
    """Quantization error must stay within half of the coarsest FP6 quantum.

    The coarsest spacing of FP6-E3M2 is 4 (between 24 and 28), so with the
    scale pinned to ``amax / 28`` no element may be off by more than ``2 *
    scale``; values quantized from the middle of a binade do strictly better.
    """
    weight = torch.randn(64, 256, dtype=torch.float32)
    codes, scales = quantize_weights_fp6(weight)

    table = _fp6_e3m2_table()
    reconstructed = table[codes.long()] * scales.to(torch.float32)[:, None]

    assert codes.dtype == torch.uint8
    assert int(codes.max()) < 64
    assert scales.dtype == torch.float16
    assert (reconstructed - weight).abs().max() <= 2.0 * scales.max() + 1e-6


@pytest.mark.wf6af16_linear
@requires_deepspeed_reference
@pytest.mark.parametrize("tokens", [1, 7, 64, 65, 200])
def test_matches_deepspeed_oracle(tokens):
    """Pin the DeepSpeed oracle explicitly, so a run that quietly stopped
    reaching it (deepspeed missing) is visible rather than silently thinner.

    DeepSpeed's CUDA kernel requires out_channels % 256 == 0 and in_channels %
    64 == 0, hence the fixed weight shape here.
    """
    hidden, weights_2bit, weights_4bit, scales = _make_case(tokens, 512, 256)

    train_out = _run_train(hidden, weights_2bit, weights_4bit, scales)
    ds_out = _deepspeed_wf6af16(hidden, weights_2bit, weights_4bit, scales)

    utils.train_assert_close(
        utils.to_reference(train_out),
        utils.to_reference(ds_out),
        torch.float16,
        reduce_dim=512,
    )
