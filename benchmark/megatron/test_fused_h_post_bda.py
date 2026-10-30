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
"""Benchmark fused_h_post_bda against Megatron's Triton kernels when usable.

Megatron is optional: a small real forward/backward probe selects its kernels,
otherwise both benchmarks warn and use torch.compile baselines that mirror
Megatron's native implementation. Backward receives grad_output and the original
inputs directly, so is_backward stays False; enabling it would make
GenericBenchmark run an extra autograd forward.
"""

import warnings

import pytest
import torch

import flag_train
from flag_train.megatron import fused_h_post_bda_backward as backward
from flag_train.megatron import fused_h_post_bda_forward as forward

from .. import base

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
_CORE_SHAPES = [
    (2, 4, 4, 1024, 0),
    (2, 4, 4, 1024, 1),
    (64, 8, 4, 4096, 0),
    (64, 8, 4, 4096, 1),
    (128, 1, 8, 7168, 0),
    (128, 1, 8, 7168, 1),
]
_COMPREHENSIVE_SHAPES = [
    (1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1),
    (1, 2, 2, 65, 0),
    (1, 2, 2, 65, 1),
    (2, 3, 4, 130, 0),
    (2, 3, 4, 130, 1),
    (512, 4, 4, 4096, 0),
    (512, 4, 4, 4096, 1),
    (8192, 4, 4, 7168, 0),
    (8192, 4, 4, 7168, 1),
]


@torch.compile
def _torch_forward(h_res, original_residual, h_post, x, bias=None):
    s, b, n, channels = original_residual.shape
    h_res_batched = h_res.view(s * b, n, n)
    residual_batched = original_residual.view(s * b, n, channels)
    mixed = torch.bmm(h_res_batched.transpose(1, 2), residual_batched).view(
        s, b, n, channels
    )
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)
    if bias is not None:
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, channels)
        return x_expanded + bias_expanded + mixed
    return x_expanded + mixed


@torch.compile
def _torch_backward(grad_output, h_res, original_residual, h_post, x, bias=None):
    s, b, n, channels = original_residual.shape
    sb = s * b
    go = grad_output.reshape(sb, n, channels)
    hr = h_res.reshape(sb, n, n)
    orig = original_residual.reshape(sb, n, channels)
    hp = h_post.reshape(sb, n)
    x_flat = x.reshape(sb, channels)

    g_hr = torch.bmm(orig, go.transpose(1, 2)).view(s, b, n, n)
    g_res = torch.bmm(hr, go).view(s, b, n, channels)
    g_x = torch.sum(go * hp.unsqueeze(-1), dim=1).view(s, b, channels)
    xb = x_flat if bias is None else x_flat + bias.view(1, channels)
    g_hp = torch.sum(go * xb.unsqueeze(1), dim=2).view(s, b, n)
    g_bias = (
        g_x.reshape(sb, channels).sum(dim=0).to(dtype=bias.dtype)
        if bias is not None
        else None
    )
    return (
        g_hr.to(dtype=h_res.dtype),
        g_res.to(dtype=original_residual.dtype),
        g_hp.to(dtype=h_post.dtype),
        g_x.to(dtype=x.dtype),
        g_bias,
    )


def _select_baseline():
    try:
        from megatron.core.fusions.fused_mhc_kernels import (
            _triton_h_post_bda_bwd,
            _triton_h_post_bda_fwd,
        )

        hr, orig, hp, x, bias = next(
            forward_input_fn((1, 1, 2, 8, 1), torch.bfloat16, flag_train.device)
        )
        with base.torch_device_fn.device(hr.device):
            output = _triton_h_post_bda_fwd(hr, orig, hp, x, bias)
            grad_output = torch.randn_like(output)
            _triton_h_post_bda_bwd(grad_output, hr, orig, hp, x, bias)
            base.torch_device_fn.synchronize()
    except Exception as exc:
        warnings.warn(
            "Megatron Triton fused_h_post_bda baseline is unavailable "
            f"({exc!r}); falling back to the Megatron-native torch.compile baseline.",
            stacklevel=2,
        )
        return (
            _torch_forward,
            _torch_backward,
            "Megatron-native torch.compile (fallback)",
        )
    return (
        _triton_h_post_bda_fwd,
        _triton_h_post_bda_bwd,
        "megatron _triton_h_post_bda_fwd/_bwd",
    )


def _make_inputs(shape, dtype, device):
    s, b, n, channels, with_bias = shape
    h_res = torch.randn((s, b, n, n), dtype=dtype, device=device)
    original_residual = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    h_post = torch.randn((s, b, n), dtype=dtype, device=device)
    x = torch.randn((s, b, channels), dtype=dtype, device=device)
    bias = torch.randn((channels,), dtype=dtype, device=device) if with_bias else None
    return h_res, original_residual, h_post, x, bias


def forward_input_fn(shape, dtype, device):
    yield _make_inputs(shape, dtype, device)


def backward_input_fn(shape, dtype, device):
    s, b, n, channels, _ = shape
    h_res, original_residual, h_post, x, bias = _make_inputs(shape, dtype, device)
    grad_output = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    yield grad_output, h_res, original_residual, h_post, x, bias


# Resolve and compile the optional baseline outside the timed regions.
_baseline_forward, _baseline_backward, _BASELINE = _select_baseline()


class FusedHPostBDABenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES
    DEFAULT_SHAPE_DESC = "S, B, N, C, WITH_BIAS"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)
        if (
            base.Config.bench_level == base.consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(_COMPREHENSIVE_SHAPES)


def torch_forward(h_res, original_residual, h_post, x, bias=None):
    return _baseline_forward(h_res, original_residual, h_post, x, bias)


def torch_backward(grad_output, h_res, original_residual, h_post, x, bias=None):
    return _baseline_backward(grad_output, h_res, original_residual, h_post, x, bias)


@pytest.mark.fused_h_post_bda_forward
def test_fused_h_post_bda_forward_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = FusedHPostBDABenchmark(
        input_fn=forward_input_fn,
        op_name="fused_h_post_bda_forward",
        torch_op=torch_forward,
        dtypes=_DTYPES,
        is_backward=False,
    )
    bench.set_train(forward)
    bench.run()


@pytest.mark.fused_h_post_bda_backward
def test_fused_h_post_bda_backward_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = FusedHPostBDABenchmark(
        input_fn=backward_input_fn,
        op_name="fused_h_post_bda_backward",
        torch_op=torch_backward,
        dtypes=_DTYPES,
        # The public backward API is timed directly, not through autograd.
        is_backward=False,
    )
    bench.set_train(backward)
    bench.run()
