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
"""Correctness tests for fused mHC H_post BDA backward."""

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
_GRADIENT_NAMES = ("h_res", "original_residual", "h_post", "x", "bias")


@contextlib.contextmanager
def _strict_fp32_matmul():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _tolerances(dtype):
    # Megatron's official BF16 backward test uses atol=rtol=5e-2. FP16 and FP32
    # are FlagTrain extensions. Long-C reductions and split-C partial merging
    # use a different accumulation order from Autograd, while FP32 accumulators
    # do not eliminate rounding when gradients are written in a narrow dtype.
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    if dtype == torch.float16:
        return 5e-3, 5e-3
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
        else (h_res, original_residual, h_post, x, bias)
    )
    with _strict_fp32_matmul():
        gradients = torch.autograd.grad(output, inputs, grad_output)
    return gradients if bias is not None else (*gradients, None)


def _assert_close(actual, expected, dtype, name):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    atol, _ = _tolerances(dtype)
    utils.train_assert_close(utils.to_reference(actual), expected, dtype, atol=atol)


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


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype,s,b,n,channels", _DTYPE_SHAPE_CASES)
def test_backward_matches_torch_autograd(dtype, s, b, n, channels, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    assert len(actual) == len(expected) == len(_GRADIENT_NAMES)
    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        if name == "bias" and not with_bias:
            assert actual_grad is None
            assert expected_grad is None
        else:
            _assert_close(actual_grad, expected_grad, dtype, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_backward_split_c_matches_torch_autograd(dtype, with_bias):
    torch.manual_seed(2026)
    s, b, n, channels = 128, 1, 8, 4096
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, dtype, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("s,b,n,channels", [(128, 1, 4, 512), (128, 1, 8, 2048)])
def test_backward_dot_threshold_matches_torch_autograd(
    dtype, s, b, n, channels, with_bias
):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, dtype, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_backward_multiple_launch_batches_match_torch_autograd(with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(1367, 1, 4, 4097, torch.bfloat16, with_bias)
    grad_output = _rand((1367, 1, 4, 4097), torch.bfloat16)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, torch.bfloat16, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("channels", [513, 768])
def test_backward_padded_b_channel_matches_torch_autograd(channels, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(128, 1, 4, channels, torch.bfloat16, with_bias)
    grad_output = _rand((128, 1, 4, channels), torch.bfloat16)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, torch.bfloat16, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.skipif(flag_train.vendor_name != "ascend", reason="Ascend grid limit")
def test_backward_scalar_b_respects_grid_limit():
    torch.manual_seed(2026)
    inputs = _make_inputs(65536, 1, 4, 65, torch.bfloat16, with_bias=True)
    grad_output = _rand((65536, 1, 4, 65), torch.bfloat16)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, torch.bfloat16, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "s,b,n,channels",
    [(1, 1, 4, 130), (1, 2, 2, 65), (1, 2, 2, 130), (1, 2, 2, 1024)],
)
def test_backward_small_fused_bias_is_independent(dtype, s, b, n, channels):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias=True)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, dtype, name)
    assert actual[3].data_ptr() != actual[4].data_ptr()
    expected_bias = actual[3].view(s * b, channels).to(torch.float32).sum(dim=0).to(dtype)
    utils.train_assert_equal(
        utils.to_reference(actual[4]), utils.to_reference(expected_bias)
    )


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels", [512, 2048, 4097, 7168])
def test_backward_dot_single_row_bias_is_independent(dtype, channels):
    torch.manual_seed(2026)
    inputs = _make_inputs(1, 1, 8, channels, dtype, with_bias=True)
    grad_output = _rand((1, 1, 8, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, dtype, name)
    assert actual[3].data_ptr() != actual[4].data_ptr()
    utils.train_assert_equal(
        utils.to_reference(actual[4]), utils.to_reference(actual[3].view(channels))
    )


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize(
    "s,b,n,channels",
    [(2, 1, 4, 130), (128, 1, 8, 512), (1, 1, 8, 513)],
    ids=["small_fused", "dot", "dot_bias1"],
)
def test_backward_accepts_non_contiguous_inputs(s, b, n, channels):
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(s, b, n, channels, torch.bfloat16, with_bias=True)
    )
    grad_output = _non_contiguous(_rand((s, b, n, channels), torch.bfloat16))
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = _ref_h_post_bda_backward(
        _reference_copy(grad_output), *(_reference_copy(tensor) for tensor in inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(
            actual_grad, expected_grad, torch.bfloat16, f"non-contiguous {name}"
        )
