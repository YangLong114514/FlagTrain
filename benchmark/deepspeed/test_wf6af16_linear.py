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
"""Performance benchmark for the WF6-AF16 linear operator.

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below:

* on a listed backend with an Ampere GPU, DeepSpeed's ``cuda_wf6af16_linear``
  **is** the baseline. The CUDA kernel asserts ``__CUDA_ARCH__ == 800``, so on
  any other architecture it cannot run -- the baseline falls back to the torch
  reference there rather than measuring nothing;
* everywhere else the baseline is ``wf6af16_ref``, an unfused torch
  composition (unpack, dequantize to an fp16 weight matrix, cuBLAS GEMM). It
  is not a competitor, and the speedup there says how far the kernel is from
  *a* correct implementation rather than from the best one.

The weight packings differ between the two implementations (DeepSpeed
interleaves bits across threads for its MMA pipeline, FlagTrain packs densely
row-major), so the inputs carry both packings; packing itself is setup, not
part of the timed region.
"""

import pytest
import torch

import flag_train
from flag_train.deepspeed import wf6af16_linear
from flag_train.deepspeed.wf6af16_linear import _fp6_e3m2_table, pack_weights_fp6

from .. import base

# Inference-style shapes: (tokens, in_channels, out_channels). The token count
# stays in the decode range the FP6 kernel targets; the channel pairs mirror
# the sizes DeepSpeed's split-K map is profiled for.
_WF6AF16_SHAPES = [
    (1, 4096, 4096),
    (16, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 14336),
    (256, 8192, 8192),
    (512, 4096, 4096),
]

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _unpack_weights_fp6(weights_2bit, weights_4bit):
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
    """Unfused torch baseline: dequantize, then a plain fp16 GEMM."""
    codes = _unpack_weights_fp6(weights_2bit, weights_4bit)
    table = _fp6_e3m2_table(device=codes.device)
    w = table[codes.long()].to(torch.float16) * scales[:, None]
    return hidden_states @ w.T


# Backends whose baseline is DeepSpeed when the extension and GPU allow it.
# cuda_wf6af16_linear ships in the InferenceCore CUDA extension, so only a
# backend that can compile and execute one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_wf6af16():
    """DeepSpeed's cuda_wf6af16_linear, or ``None`` when it cannot run here.

    Besides the extension building at all, the kernel hard-codes
    ``__CUDA_ARCH__ == 800``, so on non-Ampere NVIDIA GPUs it is treated as
    unavailable and the torch reference becomes the baseline.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None
    if flag_train.vendor_name == "nvidia" and torch.cuda.get_device_capability() != (
        8,
        0,
    ):
        return None

    try:
        from deepspeed.ops.op_builder import InferenceCoreBuilder

        inf_module = InferenceCoreBuilder().load()
        inf_module.create_handle()
    except Exception:
        # The baseline is advisory, not contractual: a missing extension just
        # means the torch reference is timed instead.
        return None

    def deepspeed_wf6af16(hidden_states, weights_2bit, weights_4bit, scales):
        out_channels = scales.shape[0]
        tokens, in_channels = hidden_states.shape
        # weights_2bit/weights_4bit here are DeepSpeed's own packing, produced
        # in the input setup; only the kernel call is timed.
        output = torch.empty(
            (tokens, out_channels), dtype=torch.float16, device=hidden_states.device
        )
        workspace = torch.empty(
            (1, out_channels, tokens), dtype=torch.float32, device=hidden_states.device
        )
        inf_module.cuda_wf6af16_linear(
            output,
            hidden_states,
            weights_2bit,
            weights_4bit,
            scales,
            workspace,
            out_channels,
            tokens,
            in_channels,
            1,
        )
        return output

    # The DeepSpeed packing needs the host-side preprocess_weight; expose it
    # so the input setup can pre-pack.
    deepspeed_wf6af16.preprocess_weight = inf_module.preprocess_weight
    return deepspeed_wf6af16


# Resolved once, so the first-use JIT compile is not counted in the measurement.
_deepspeed_wf6af16 = _load_deepspeed_wf6af16()

_BASELINE = (
    "deepspeed cuda_wf6af16_linear"
    if _deepspeed_wf6af16 is not None
    else "wf6af16_ref (torch)"
)


class Wf6Af16LinearBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _WF6AF16_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_WF6AF16_SHAPES)

    def set_more_shapes(self):
        return []


def wf6af16_input_fn(shape, dtype, device):
    tokens, in_channels, out_channels = shape
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 64, (out_channels, in_channels), generator=gen).to(
        torch.uint8
    )
    scales = (torch.rand(out_channels, generator=gen) + 0.5).to(torch.float16)
    hidden = torch.randn(tokens, in_channels, generator=gen).to(dtype)

    weights_2bit, weights_4bit = pack_weights_fp6(codes.to(device))

    if _deepspeed_wf6af16 is not None:
        # DeepSpeed's packer wants the fake-quantized fp16 weights on CPU; the
        # table values are exact in fp16, so the repack is lossless.
        table = _fp6_e3m2_table()
        fake_fp16 = table[codes.long()].to(torch.float16)
        ds_w2, ds_w4 = _deepspeed_wf6af16.preprocess_weight(fake_fp16)
        ds_w2 = ds_w2.to(device)
        ds_w4 = ds_w4.to(device)
    else:
        ds_w2 = ds_w4 = None

    yield (
        hidden.to(device),
        weights_2bit,
        weights_4bit,
        scales.to(device),
        ds_w2,
        ds_w4,
    )


def torch_op(hidden, weights_2bit, weights_4bit, scales, ds_w2, ds_w4):
    """Baseline, chosen by platform. See the module docstring."""
    if _deepspeed_wf6af16 is not None:
        return _deepspeed_wf6af16(hidden, ds_w2, ds_w4, scales)
    return wf6af16_ref(hidden, weights_2bit, weights_4bit, scales)


def train_op(hidden, weights_2bit, weights_4bit, scales, ds_w2, ds_w4):
    """The operator under test."""
    out_channels = scales.shape[0]
    tokens, in_channels = hidden.shape
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
    )


@pytest.mark.wf6af16_linear
def test_wf6af16_linear_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = Wf6Af16LinearBenchmark(
        input_fn=wf6af16_input_fn,
        op_name="wf6af16_linear",
        torch_op=torch_op,
        dtypes=[torch.float16],
    )
    bench.set_train(train_op)
    bench.run()
