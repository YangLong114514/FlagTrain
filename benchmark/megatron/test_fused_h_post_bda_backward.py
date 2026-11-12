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
"""Performance benchmark for the fused mHC H_post BDA backward operator.

The local baseline is Megatron-LM's ``_triton_h_post_bda_bwd`` when installed.
Otherwise ``torch_op`` is ``None`` and the NVIDIA ``--base-data`` record is the
only comparison. Ascend always uses the NVIDIA record without probing Megatron.
No PyTorch composition is timed as a benchmark baseline.

The operator returns the gradients directly, so ``is_backward`` stays ``False``.
"""

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


def torch_op():
    try:
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_bwd

        return _triton_h_post_bda_bwd
    except Exception:
        return None


def _make_inputs(shape, dtype, device):
    s, b, n, channels, with_bias = shape
    h_res = torch.randn((s, b, n, n), dtype=dtype, device=device)
    original_residual = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    h_post = torch.randn((s, b, n), dtype=dtype, device=device)
    x = torch.randn((s, b, channels), dtype=dtype, device=device)
    bias = torch.randn((channels,), dtype=dtype, device=device) if with_bias else None
    grad_output = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    return grad_output, h_res, original_residual, h_post, x, bias


def backward_input_fn(shape, dtype, device):
    yield _make_inputs(shape, dtype, device)


if flag_train.vendor_name == "ascend":
    baseline_op = None
else:
    baseline_op = torch_op()
_BASELINE = (
    "megatron _triton_h_post_bda_bwd"
    if baseline_op is not None
    else "NVIDIA base-data only (no local baseline)"
)


class FusedHPostBDABackwardBenchmark(base.GenericBenchmark):
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


@pytest.mark.fused_h_post_bda_backward
def test_fused_h_post_bda_backward_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = FusedHPostBDABackwardBenchmark(
        input_fn=backward_input_fn,
        op_name="fused_h_post_bda_backward",
        torch_op=baseline_op,
        dtypes=_DTYPES,
        is_backward=False,
    )
    bench.set_train(flag_train.megatron.fused_h_post_bda_backward)
    bench.run()
