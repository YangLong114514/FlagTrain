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


def torch_op():
    try:
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_bwd

        return _triton_h_post_bda_bwd
    except Exception:
        return _ref_h_post_bda_backward


if flag_train.vendor_name == "ascend":
    reference_op = _ref_h_post_bda_backward
else:
    reference_op = torch_op()

# Tests marked with this exercise a specific Ascend kernel path or launcher
# behavior. They are skipped on other vendors because running them there takes
# far too long -- every distinct shape re-triggers the generic implementation's
# autotune sweep -- so they run on Ascend only.
_ASCEND_ONLY = pytest.mark.skipif(
    flag_train.vendor_name != "ascend",
    reason="Exercises an Ascend-specific kernel path",
)


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


def _contiguous(inputs):
    # Megatron's native op requires bias as a positional argument and reads it
    # without calling .contiguous(), so the reference always gets dense copies.
    return tuple(
        tensor.contiguous() if tensor is not None else None for tensor in inputs
    )


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype,s,b,n,channels", _DTYPE_SHAPE_CASES)
def test_backward_matches_reference(dtype, s, b, n, channels, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = tuple(
        _reference_copy(grad) for grad in reference_op(grad_output, *inputs)
    )

    assert len(actual) == len(expected) == len(_GRADIENT_NAMES)
    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        if name == "bias" and not with_bias:
            assert actual_grad is None
            assert expected_grad is None
        else:
            _assert_close(actual_grad, expected_grad, dtype, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.skipif(
    flag_train.vendor_name != "nvidia" or reference_op is _ref_h_post_bda_backward,
    reason="Requires Megatron's native backward on NVIDIA",
)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_backward_megatron_native_matches_torch_composition(with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(2, 3, 4, 130, torch.bfloat16, with_bias)
    grad_output = _rand((2, 3, 4, 130), torch.bfloat16)
    actual = reference_op(grad_output, *inputs)
    expected = tuple(
        _reference_copy(grad) for grad in _ref_h_post_bda_backward(grad_output, *inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, torch.bfloat16, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_backward_split_c_matches_reference(dtype, with_bias):
    torch.manual_seed(2026)
    s, b, n, channels = 128, 1, 8, 4096
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = tuple(
        _reference_copy(grad) for grad in reference_op(grad_output, *inputs)
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, dtype, name)


@pytest.mark.fused_h_post_bda_backward
@pytest.mark.parametrize(
    "s,b,n,channels",
    [
        (2, 1, 4, 130),
        (128, 1, 8, 512),
        (1, 1, 8, 513),
        (128, 1, 4, 512),
        (128, 1, 2, 256),
        (128, 1, 4, 4097),
        (1, 1, 4, 4097),
        (2048, 4, 4, 65),
    ],
    ids=[
        "small_fused",
        "dot",
        "dot_bias1",
        "vector4_single",
        "vecn_single",
        "vector4",
        "vector4_bias1",
        "vector4_rows",
    ],
)
def test_backward_accepts_non_contiguous_inputs(s, b, n, channels):
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(s, b, n, channels, torch.bfloat16, with_bias=True)
    )
    grad_output = _non_contiguous(_rand((s, b, n, channels), torch.bfloat16))
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = tuple(
        _reference_copy(grad)
        for grad in reference_op(*_contiguous((grad_output, *inputs)))
    )

    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(
            actual_grad, expected_grad, torch.bfloat16, f"non-contiguous {name}"
        )


# Ascend kernel-path coverage and host-side contract checks, grouped into a few
# loop-driven cases to keep the suite small. Failure messages carry the
# offending case instead of a parametrized id.


def _assert_backward_matches(s, b, n, channels, dtype, with_bias, label):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    grad_output = _rand((s, b, n, channels), dtype)
    actual = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    expected = tuple(
        _reference_copy(grad) for grad in reference_op(grad_output, *inputs)
    )
    for name, actual_grad, expected_grad in zip(_GRADIENT_NAMES, actual, expected):
        _assert_close(actual_grad, expected_grad, dtype, f"{label} {name}")
    return actual


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_backward
def test_backward_ascend_kernel_paths_match_reference():
    cases = []
    # dot dispatch threshold boundaries
    for dtype in (torch.bfloat16, torch.float32):
        for with_bias in (False, True):
            for s, b, n, channels in [
                (128, 1, 4, 256),
                (128, 1, 4, 300),
                (128, 1, 8, 256),
                (128, 1, 4, 512),
                (128, 1, 2, 256),
                (128, 1, 2, 100),
            ]:
                cases.append((s, b, n, channels, dtype, with_bias))
    # vector4 rows
    for dtype in (torch.bfloat16, torch.float32):
        for with_bias in (False, True):
            cases.append((2048, 4, 4, 2049, dtype, with_bias))
    # large A kernel
    cases.append((8193, 1, 4, 4097, torch.bfloat16, False))
    cases.append((8193, 1, 4, 2049, torch.float32, False))
    # B kernel padded channel
    cases.append((128, 1, 4, 513, torch.bfloat16, True))
    cases.append((128, 1, 4, 768, torch.bfloat16, True))

    for s, b, n, channels, dtype, with_bias in cases:
        _assert_backward_matches(
            s,
            b,
            n,
            channels,
            dtype,
            with_bias,
            f"path ({s},{b},{n},{channels},{dtype},bias={with_bias})",
        )


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_backward
def test_backward_ascend_launch_batches_and_grid_limit():
    # Multiple launch batches for the dot and vector4-rows paths.
    for s, b, n, channels in [
        (1367, 1, 4, 4097),  # dot
        (40960, 1, 4, 65),  # vector4 rows
    ]:
        for with_bias in (False, True):
            _assert_backward_matches(
                s,
                b,
                n,
                channels,
                torch.bfloat16,
                with_bias,
                f"batches ({s},{b},{n},{channels},bias={with_bias})",
            )
    # The scalar-B fallback must respect the grid dimension limit.
    _assert_backward_matches(
        65536, 1, 4, 65, torch.bfloat16, True, "grid limit (65536,1,4,65)"
    )


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_backward
def test_backward_ascend_bias_gradient_semantics():
    # Small fused path: the bias gradient is an independent allocation reduced
    # from the rounded g_x in fp32.
    for dtype in (torch.bfloat16, torch.float32):
        for s, b, n, channels in [
            (1, 1, 4, 130),
            (1, 2, 2, 65),
            (1, 2, 2, 1024),
            (2, 4, 4, 1024),
        ]:
            actual = _assert_backward_matches(
                s,
                b,
                n,
                channels,
                dtype,
                True,
                f"small fused ({s},{b},{n},{channels},{dtype})",
            )
            assert actual[3].data_ptr() != actual[4].data_ptr()
            expected_bias = (
                actual[3].view(s * b, channels).to(torch.float32).sum(dim=0).to(dtype)
            )
            if s * b <= 2:
                utils.train_assert_equal(
                    utils.to_reference(actual[4]), utils.to_reference(expected_bias)
                )
            else:
                _assert_close(
                    actual[4], _reference_copy(expected_bias), dtype, "independent bias"
                )

    # Dot single-row path: the bias gradient equals the single g_x row exactly.
    for dtype in (torch.bfloat16, torch.float32):
        for n in (4, 8):
            for channels in (512, 4097):
                actual = _assert_backward_matches(
                    1,
                    1,
                    n,
                    channels,
                    dtype,
                    True,
                    f"single row ({n},{channels},{dtype})",
                )
                assert actual[3].data_ptr() != actual[4].data_ptr()
                utils.train_assert_equal(
                    utils.to_reference(actual[4]),
                    utils.to_reference(actual[3].view(channels)),
                )


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_backward
def test_backward_ascend_repeated_calls_and_validation():
    # Repeated calls must return independent, correct outputs.
    for dtype in (torch.bfloat16, torch.float32):
        for with_bias in (False, True):
            torch.manual_seed(2026)
            inputs = _make_inputs(1, 2, 2, 65, dtype, with_bias)
            grad_output = _rand((1, 2, 2, 65), dtype)
            first = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
            first_sums = tuple(
                grad.float().sum() if grad is not None else None for grad in first
            )
            second = flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
            expected = tuple(
                _reference_copy(grad) for grad in reference_op(grad_output, *inputs)
            )
            label = f"{dtype},bias={with_bias}"
            for index, (first_grad, second_grad, expected_grad) in enumerate(
                zip(first, second, expected)
            ):
                if expected_grad is None:
                    assert first_grad is None and second_grad is None
                    continue
                assert (
                    first_grad.untyped_storage().data_ptr()
                    != second_grad.untyped_storage().data_ptr()
                )
                utils.train_assert_equal(
                    utils.to_reference(first_sums[index]),
                    utils.to_reference(first_grad.float().sum()),
                )
                _assert_close(first_grad, expected_grad, dtype, f"first call {label}")
                _assert_close(second_grad, expected_grad, dtype, f"second call {label}")
            assert len(
                {
                    grad.untyped_storage().data_ptr()
                    for grad in second
                    if grad is not None
                }
            ) == sum(grad is not None for grad in expected)

    # Validation must still reject a mismatched grad_output.
    inputs = _make_inputs(1, 2, 2, 65, torch.bfloat16, with_bias=False)
    grad_output = _rand((1, 2, 2, 65), torch.bfloat16)
    for _ in range(2):
        flag_train.megatron.fused_h_post_bda_backward(grad_output, *inputs)
    with pytest.raises(ValueError, match="grad_output must have"):
        flag_train.megatron.fused_h_post_bda_backward(grad_output[:0], *inputs)
