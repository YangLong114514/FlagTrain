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

"""NVIDIA baseline for fused_h_post_bda_forward.

Self-contained module; depends only on torch and Megatron-LM. It measures
``_triton_h_post_bda_fwd`` directly and raises when that implementation is
unavailable instead of substituting a torch composition.
"""

import torch

OP_NAME = "fused_h_post_bda_forward"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False

# (s, b, n, C, with_bias), copied from the benchmark's core and extra shapes.
_SHAPES = [
    (2, 4, 4, 1024, 0),
    (2, 4, 4, 1024, 1),
    (64, 8, 4, 4096, 0),
    (64, 8, 4, 4096, 1),
    (128, 1, 8, 7168, 0),
    (128, 1, 8, 7168, 1),
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

_MEGATRON_FORWARD = None


def _megatron_forward():
    global _MEGATRON_FORWARD
    if _MEGATRON_FORWARD is None:
        try:
            from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_fwd
        except Exception as exc:
            raise RuntimeError(
                "fused_h_post_bda_forward baseline requires Megatron-LM's "
                "_triton_h_post_bda_fwd; install Megatron-LM from a version "
                f"that provides it: {exc!r}"
            ) from exc
        _MEGATRON_FORWARD = _triton_h_post_bda_fwd
    return _MEGATRON_FORWARD


def torch_op(h_res, original_residual, h_post, x, bias=None):
    return _megatron_forward()(h_res, original_residual, h_post, x, bias)


def native():
    _megatron_forward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def _make_operands(s, b, n, channels, with_bias, dtype, device):
    h_res = torch.randn((s, b, n, n), dtype=dtype, device=device)
    original_residual = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    h_post = torch.randn((s, b, n), dtype=dtype, device=device)
    x = torch.randn((s, b, channels), dtype=dtype, device=device)
    bias = torch.randn((channels,), dtype=dtype, device=device) if with_bias else None
    return h_res, original_residual, h_post, x, bias


def build_inputs(binding, dtype, device):
    return _make_operands(*binding["shape"], dtype, device), {}


def key_shape(binding):
    *shape, with_bias = binding["shape"]
    suffix = "bias" if with_bias else "no_bias"
    return "x".join(str(dim) for dim in shape) + "_" + suffix


def config(binding, dtype):
    s, b, n, channels, with_bias = binding["shape"]
    inputs = {
        "h_res": {"shape": [s, b, n, n], "dtype": str(dtype)},
        "original_residual": {
            "shape": [s, b, n, channels],
            "dtype": str(dtype),
        },
        "h_post": {"shape": [s, b, n], "dtype": str(dtype)},
        "x": {"shape": [s, b, channels], "dtype": str(dtype)},
    }
    if with_bias:
        inputs["bias"] = {"shape": [channels], "dtype": str(dtype)}
    return {
        "inputs": inputs,
        "dims": {"s": s, "b": b, "n": n, "C": channels, "with_bias": with_bias},
    }
