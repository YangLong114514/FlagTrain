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
"""Correctness tests for fused mHC H_post BDA forward."""

import contextlib

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

_DTYPE_SHAPE_CASES = [
    (torch.bfloat16, 1, 1, 1, 1),
    (torch.bfloat16, 1, 2, 2, 65),
    (torch.bfloat16, 2, 1, 4, 130),
    (torch.bfloat16, 2, 3, 4, 130),
    (torch.bfloat16, 1, 1, 8, 257),
    (torch.bfloat16, 2, 5, 8, 257),
    (torch.bfloat16, 513, 1, 4, 130),
    (torch.float16, 1, 2, 2, 65),
    (torch.float16, 2, 1, 4, 130),
    (torch.float16, 1, 1, 8, 257),
    (torch.float16, 513, 1, 4, 257),
    (torch.float16, 2, 5, 8, 257),
    (torch.float32, 1, 2, 2, 65),
    (torch.float32, 2, 1, 4, 130),
    (torch.float32, 1, 1, 8, 257),
    (torch.float32, 513, 1, 4, 257),
    (torch.float32, 2, 5, 8, 257),
]


@contextlib.contextmanager
def _strict_fp32_matmul():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _tolerances(dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-5, 1e-5


def _rand(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=flag_train.device).uniform_(-0.1, 0.1)


def _make_inputs(s, b, n, channels, dtype, with_bias):
    return (
        _rand((s, b, n, n), dtype),
        _rand((s, b, n, channels), dtype),
        _rand((s, b, n), dtype),
        _rand((s, b, channels), dtype),
        _rand((channels,), dtype) if with_bias else None,
    )


def _reference_copy(tensor):
    if tensor is None:
        return None
    return utils.to_reference(tensor).clone()


def _ref_h_post_bda(h_res, original_residual, h_post, x, bias=None):
    s, b, n, channels = original_residual.shape
    with _strict_fp32_matmul():
        mixed = torch.bmm(
            h_res.reshape(s * b, n, n).transpose(1, 2),
            original_residual.reshape(s * b, n, channels),
        ).reshape(s, b, n, channels)
    output = h_post.unsqueeze(-1) * x.unsqueeze(2) + mixed
    if bias is not None:
        output = output + h_post.unsqueeze(-1) * bias.reshape(1, 1, 1, channels)
    return output


def _assert_close(actual, expected, dtype, name):
    atol, _ = _tolerances(dtype)
    utils.train_assert_close(utils.to_reference(actual), expected, dtype, atol=atol)


def _non_contiguous(tensor):
    padded = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] * 2),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result = padded[..., ::2]
    result.copy_(tensor)
    assert not result.is_contiguous()
    return result


@pytest.mark.fused_h_post_bda_forward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype,s,b,n,channels", _DTYPE_SHAPE_CASES)
def test_forward_matches_torch_reference(dtype, s, b, n, channels, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _ref_h_post_bda(*(_reference_copy(tensor) for tensor in inputs))
    _assert_close(actual, expected, dtype, "output")


@pytest.mark.fused_h_post_bda_forward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_forward_medium_shapes_match_torch_reference(dtype, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(64, 8, 4, 4099, dtype, with_bias)
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(_ref_h_post_bda(*inputs))
    _assert_close(actual, expected, dtype, "medium-shape output")


@pytest.mark.fused_h_post_bda_forward
def test_forward_uses_h_res_transpose():
    h_res = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32, device=flag_train.device
    )
    original_residual = torch.tensor(
        [[[[10.0, 100.0], [1.0, 2.0]]]], dtype=torch.float32, device=flag_train.device
    )
    h_post = torch.zeros(1, 1, 2, dtype=torch.float32, device=flag_train.device)
    x = torch.zeros(1, 1, 2, dtype=torch.float32, device=flag_train.device)
    expected = torch.tensor(
        [[[[13.0, 106.0], [24.0, 208.0]]]],
        dtype=torch.float32,
        device=flag_train.device,
    )

    actual = flag_train.megatron.fused_h_post_bda_forward(
        h_res, original_residual, h_post, x
    )
    utils.train_assert_equal(actual, expected)


@pytest.mark.fused_h_post_bda_forward
def test_forward_accepts_non_contiguous_inputs():
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(2, 1, 4, 130, torch.bfloat16, with_bias=True)
    )
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _ref_h_post_bda(*(_reference_copy(tensor) for tensor in inputs))
    _assert_close(actual, expected, torch.bfloat16, "non-contiguous output")


@pytest.mark.fused_h_post_bda_forward
def test_forward_medium_shape_accepts_non_contiguous_inputs():
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(64, 8, 4, 4096, torch.bfloat16, with_bias=True)
    )
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(_ref_h_post_bda(*inputs))
    _assert_close(actual, expected, torch.bfloat16, "medium non-contiguous output")
