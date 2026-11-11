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
"""Performance benchmark for the fused mHC H_post BDA forward operator.

The baseline is Megatron-LM's ``_triton_h_post_bda_fwd`` when the installed
megatron-core provides a kernel that can run here -- the kernel this port is a
port *of*. When that kernel cannot run, the baseline is Megatron's own
``native_h_post_bda`` torch composition, compiled the way Megatron compiles it
(eager if ``torch.compile`` fails on this platform). ``_BASELINE`` records which
one was measured. On Ascend, no local baseline is probed or executed;
``torch_op=None`` makes the NVIDIA ``--base-data`` record the only comparison.

Both take the same ``(h_res, original_residual, h_post, x, bias)`` tuple, so
there is nothing to pre-compute outside the timed region. The functional suite
(``tests/megatron/test_fused_h_post_bda_forward.py``) holds the operator to the
torch composition, so the timed baseline is not the only check.
"""

import warnings

import pytest
import torch

import flag_train

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


def _eager_forward(h_res, original_residual, h_post, x, bias=None):
    """Megatron-LM's ``native_h_post_bda`` body, transcribed."""
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


_torch_forward = torch.compile(_eager_forward)

# The compiled baseline needs one graph per (shape, dtype); with dynamo's
# default budget of 8 recompiles the tail of the sweep would be served eager.
_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    """Keep ``@torch.compile`` compiling across the whole sweep."""
    config = torch._dynamo.config
    names = [
        name
        for name in ("recompile_limit", "cache_size_limit")
        if hasattr(config, name)
    ]
    saved = {name: getattr(config, name) for name in names}
    for name in names:
        setattr(config, name, max(saved[name], _COMPILE_BUDGET))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def _load_baseline():
    """Megatron's Triton forward kernel, else its torch composition."""
    try:
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_fwd

        inputs = next(
            forward_input_fn((1, 1, 2, 8, 1), torch.bfloat16, flag_train.device)
        )
        with base.torch_device_fn.device(inputs[0].device):
            _triton_h_post_bda_fwd(*inputs)
            base.torch_device_fn.synchronize()
        return _triton_h_post_bda_fwd, "megatron _triton_h_post_bda_fwd"
    except Exception as exc:
        warnings.warn(
            "Megatron-LM's Triton fused_h_post_bda_forward kernel is "
            f"unavailable ({exc!r}); timing its native_h_post_bda torch "
            "composition instead.",
            stacklevel=2,
        )

    try:
        inputs = next(
            forward_input_fn((1, 1, 2, 8, 1), torch.bfloat16, flag_train.device)
        )
        with base.torch_device_fn.device(inputs[0].device):
            _torch_forward(*inputs)
            base.torch_device_fn.synchronize()
        return (
            _torch_forward,
            "megatron native_h_post_bda torch composition (@torch.compile)",
        )
    except Exception as compile_exc:
        warnings.warn(
            f"torch.compile baseline is unavailable ({compile_exc!r}); "
            "falling back to eager execution.",
            stacklevel=2,
        )
        return (
            _eager_forward,
            "megatron native_h_post_bda torch composition (eager)",
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


if flag_train.vendor_name == "ascend":
    torch_op, _BASELINE = None, "NVIDIA base-data only (no local baseline)"
else:
    torch_op, _BASELINE = _load_baseline()


class FusedHPostBDAForwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES
    DEFAULT_SHAPE_DESC = "S, B, N, C, WITH_BIAS"

    @staticmethod
    def _event_latency(op, *args, **kwargs):
        for _ in range(5):
            op(*args, **kwargs)
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            op(*args, **kwargs)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 30

    def get_latency(self, op, *args, _reference_candidate=False, **kwargs):
        if (
            base.vendor_name == "ascend"
            and base.Config.mode == base.consts.BenchMode.KERNEL
        ):
            return self._event_latency(op, *args, **kwargs)
        return super().get_latency(
            op, *args, _reference_candidate=_reference_candidate, **kwargs
        )

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)
        if (
            base.Config.bench_level == base.consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(_COMPREHENSIVE_SHAPES)


@pytest.mark.fused_h_post_bda_forward
def test_fused_h_post_bda_forward_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = FusedHPostBDAForwardBenchmark(
        input_fn=forward_input_fn,
        op_name="fused_h_post_bda_forward",
        # Megatron's Triton kernel, or its native_h_post_bda composition when
        # that kernel cannot run here.
        torch_op=torch_op,
        dtypes=_DTYPES,
        is_backward=False,
    )
    bench.set_train(flag_train.megatron.fused_h_post_bda_forward)
    bench.run()
