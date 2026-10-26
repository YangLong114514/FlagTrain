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
"""Correctness tests for fused mHC H_post BDA forward and backward."""

import pytest
import torch

import flag_train
from flag_train.megatron import fused_h_post_bda_backward, fused_h_post_bda_forward

from .. import accuracy_utils as utils

_DTYPE_SHAPE_CASES = [
    (torch.bfloat16, 1, 2, 2, 65),
    (torch.bfloat16, 2, 1, 4, 130),
    (torch.bfloat16, 1, 1, 8, 257),
    (torch.float16, 2, 1, 4, 130),
    (torch.float32, 2, 1, 4, 130),
]
_GRADIENT_NAMES = ("h_res", "original_residual", "h_post", "x", "bias")


def _tolerances(dtype, backward=False):
    if dtype == torch.bfloat16:
        return (5e-2, 5e-2) if backward else (2e-2, 2e-2)
    if dtype == torch.float16:
        return (5e-3, 5e-3) if backward else (1e-3, 1e-3)
    return (1e-5, 1e-5)


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
    mixed = torch.bmm(
        h_res.reshape(s * b, n, n).transpose(1, 2),
        original_residual.reshape(s * b, n, channels),
    ).reshape(s, b, n, channels)
    output = h_post.unsqueeze(-1) * x.unsqueeze(2) + mixed
    if bias is not None:
        output = output + h_post.unsqueeze(-1) * bias.reshape(1, 1, 1, channels)
    return output


def _ref_h_post_bda_backward(
    grad_output, h_res, original_residual, h_post, x, bias=None
):
    h_res = h_res.requires_grad_(True)
    original_residual = original_residual.requires_grad_(True)
    h_post = h_post.requires_grad_(True)
    x = x.requires_grad_(True)
    bias = bias.requires_grad_(True) if bias is not None else None
    output = _ref_h_post_bda(h_res, original_residual, h_post, x, bias)
    inputs = (
        (h_res, original_residual, h_post, x)
        if bias is None
        else (
            h_res,
            original_residual,
            h_post,
            x,
            bias,
        )
    )
    gradients = torch.autograd.grad(output, inputs, grad_output)
    return gradients if bias is not None else (*gradients, None)


def _assert_close(actual, expected, dtype, name, backward=False):
    atol, rtol = _tolerances(dtype, backward)
    torch.testing.assert_close(
        utils.to_reference(actual),
        expected,
        atol=atol,
        rtol=rtol,
        msg=f"{'backward' if backward else 'forward'} mismatch on {name}",
    )


def _non_contiguous(tensor):
    if tensor is None:
        return None
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
    actual = fused_h_post_bda_forward(*inputs)
    expected = _ref_h_post_bda(*(_reference_copy(tensor) for tensor in inputs))
    _assert_close(actual, expected, dtype, "output")


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype,s,b,n,channels", _DTYPE_SHAPE_CASES)
def test_backward_matches_torch_autograd(dtype, s, b, n, channels, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    assert len(actual) == len(expected) == len(_GRADIENT_NAMES)
    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        if name == "bias" and not with_bias:
            assert actual_grad is None
            assert expected_grad is None
        else:
            _assert_close(actual_grad, expected_grad, dtype, name, backward=True)


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

    actual = fused_h_post_bda_forward(h_res, original_residual, h_post, x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.fused_h_post_bda_forward
def test_forward_accepts_non_contiguous_inputs():
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(2, 1, 4, 130, torch.bfloat16, with_bias=True)
    )
    actual = fused_h_post_bda_forward(*inputs)
    expected = _ref_h_post_bda(*(_reference_copy(tensor) for tensor in inputs))
    _assert_close(actual, expected, torch.bfloat16, "non-contiguous output")


@pytest.mark.fused_h_post_bda_backward
def test_backward_accepts_non_contiguous_inputs():
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(2, 1, 4, 130, torch.bfloat16, with_bias=True)
    )
    grad_output = _non_contiguous(_rand((2, 1, 4, 130), torch.bfloat16))
    actual = fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(
            actual_grad,
            expected_grad,
            torch.bfloat16,
            f"non-contiguous {name}",
            backward=True,
        )
