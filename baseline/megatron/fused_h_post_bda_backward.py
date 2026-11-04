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

"""NVIDIA baseline for fused_h_post_bda_backward.

Self-contained module; depends only on torch and Megatron-LM. It measures
``_triton_h_post_bda_bwd`` directly and raises when that implementation is
unavailable instead of substituting a torch composition or autograd pass.
"""

import torch

OP_NAME = "fused_h_post_bda_backward"
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

_MEGATRON_BACKWARD = None


def _megatron_backward():
    global _MEGATRON_BACKWARD
    if _MEGATRON_BACKWARD is None:
        try:
            from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_bwd
        except Exception as exc:
            raise RuntimeError(
                "fused_h_post_bda_backward baseline requires Megatron-LM's "
                "_triton_h_post_bda_bwd; install Megatron-LM from a version "
                f"that provides it: {exc!r}"
            ) from exc
        _MEGATRON_BACKWARD = _triton_h_post_bda_bwd
    return _MEGATRON_BACKWARD


def torch_op(grad_output, h_res, original_residual, h_post, x, bias=None):
    return _megatron_backward()(grad_output, h_res, original_residual, h_post, x, bias)


def native():
    _megatron_backward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    s, b, n, channels, with_bias = binding["shape"]
    h_res = torch.randn((s, b, n, n), dtype=dtype, device=device)
    original_residual = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    h_post = torch.randn((s, b, n), dtype=dtype, device=device)
    x = torch.randn((s, b, channels), dtype=dtype, device=device)
    bias = torch.randn((channels,), dtype=dtype, device=device) if with_bias else None
    grad_output = torch.randn((s, b, n, channels), dtype=dtype, device=device)
    return (grad_output, h_res, original_residual, h_post, x, bias), {}


def key_shape(binding):
    *shape, with_bias = binding["shape"]
    suffix = "bias" if with_bias else "no_bias"
    return "x".join(str(dim) for dim in shape) + "_" + suffix


def config(binding, dtype):
    s, b, n, channels, with_bias = binding["shape"]
    inputs = {
        "grad_output": {
            "shape": [s, b, n, channels],
            "dtype": str(dtype),
        },
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
