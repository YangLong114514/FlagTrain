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
    # Megatron's official BF16 forward test uses atol=rtol=2e-2. FP16 and FP32
    # are FlagTrain extensions. FP32 accumulation still leaves output rounding
    # and reduction-order differences from the PyTorch reference.
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


def torch_op():
    try:
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_post_bda_fwd

        return _triton_h_post_bda_fwd
    except Exception:
        return _ref_h_post_bda


if flag_train.vendor_name == "ascend":
    reference_op = _ref_h_post_bda
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


def _contiguous(inputs):
    # Megatron's native op requires bias as a positional argument and reads it
    # without calling .contiguous(), so the reference always gets dense copies.
    return tuple(
        tensor.contiguous() if tensor is not None else None for tensor in inputs
    )


@pytest.mark.fused_h_post_bda_forward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype,s,b,n,channels", _DTYPE_SHAPE_CASES)
def test_forward_matches_reference(dtype, s, b, n, channels, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(reference_op(*inputs))
    _assert_close(actual, expected, dtype, "output")


@pytest.mark.fused_h_post_bda_forward
@pytest.mark.skipif(
    flag_train.vendor_name != "nvidia" or reference_op is _ref_h_post_bda,
    reason="Requires Megatron's native forward on NVIDIA",
)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_forward_megatron_native_matches_torch_composition(with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(2, 3, 4, 130, torch.bfloat16, with_bias)
    actual = reference_op(*inputs)
    expected = _reference_copy(_ref_h_post_bda(*inputs))
    _assert_close(actual, expected, torch.bfloat16, "native output")


@pytest.mark.fused_h_post_bda_forward
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_forward_medium_shapes_match_reference(dtype, with_bias):
    torch.manual_seed(2026)
    inputs = _make_inputs(64, 8, 4, 4099, dtype, with_bias)
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(reference_op(*inputs))
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
    utils.train_assert_equal(utils.to_reference(actual), utils.to_reference(expected))


@pytest.mark.fused_h_post_bda_forward
def test_forward_accepts_non_contiguous_inputs():
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(2, 1, 4, 130, torch.bfloat16, with_bias=True)
    )
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(reference_op(*_contiguous(inputs)))
    _assert_close(actual, expected, torch.bfloat16, "non-contiguous output")


@pytest.mark.fused_h_post_bda_forward
def test_forward_medium_shape_accepts_non_contiguous_inputs():
    torch.manual_seed(2026)
    inputs = tuple(
        _non_contiguous(tensor)
        for tensor in _make_inputs(64, 8, 4, 4096, torch.bfloat16, with_bias=True)
    )
    actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(reference_op(*_contiguous(inputs)))
    _assert_close(actual, expected, torch.bfloat16, "medium non-contiguous output")


# Ascend kernel-path coverage and host-side contract checks, grouped into a few
# loop-driven cases to keep the suite small. Failure messages carry the
# offending case instead of a parametrized id.
_ASCEND_FORWARD_PATH_CASES = [
    # vector4, single channel block
    (512, 1, 4, 65, torch.bfloat16, True),
    (512, 1, 4, 65, torch.float32, True),
    (512, 1, 4, 1024, torch.bfloat16, True),
    (512, 1, 4, 1024, torch.float32, True),
    # vector8, single channel block
    (256, 1, 8, 65, torch.bfloat16, True),
    (256, 1, 8, 65, torch.float32, True),
    (256, 1, 8, 1024, torch.bfloat16, True),
    (256, 1, 8, 1024, torch.float32, True),
    # vector4 medium rows (ROWS=4)
    (256, 4, 4, 1024, torch.bfloat16, True),
    (256, 4, 4, 1024, torch.float16, True),
    (256, 4, 4, 4097, torch.bfloat16, True),
    (256, 4, 4, 4097, torch.float16, True),
    # vector4 rows (ROWS=8)
    (2048, 4, 4, 1024, torch.bfloat16, False),
    (2048, 4, 4, 1024, torch.bfloat16, True),
    (2048, 4, 4, 4097, torch.bfloat16, False),
    (2048, 4, 4, 4097, torch.bfloat16, True),
    # row dot
    (128, 1, 8, 7168, torch.bfloat16, False),
    (128, 1, 8, 7168, torch.bfloat16, True),
    (128, 1, 8, 7168, torch.float32, False),
    (128, 1, 8, 7168, torch.float32, True),
    # grouped dot
    (8193, 1, 8, 4097, torch.bfloat16, False),
    (8193, 1, 8, 4097, torch.bfloat16, True),
    (8193, 1, 8, 2049, torch.float32, False),
    (8193, 1, 8, 2049, torch.float32, True),
]


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_forward
def test_forward_ascend_kernel_paths_match_reference():
    for s, b, n, channels, dtype, with_bias in _ASCEND_FORWARD_PATH_CASES:
        torch.manual_seed(2026)
        inputs = _make_inputs(s, b, n, channels, dtype, with_bias)
        actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
        expected = _reference_copy(reference_op(*inputs))
        _assert_close(
            actual,
            expected,
            dtype,
            f"path ({s},{b},{n},{channels},{dtype},bias={with_bias})",
        )


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_forward
def test_forward_ascend_launch_batches_match_reference():
    for s, b, n, channels in [
        (1367, 1, 4, 4097),  # dot, multiple launch batches
        (2048, 4, 8, 4096),  # row dot, multiple launch batches
        (8192, 1, 8, 65),  # vector8, multiple launch batches
        (20480, 2, 4, 2049),  # vector4 rows, multiple launch batches
    ]:
        for with_bias in (False, True):
            torch.manual_seed(2026)
            inputs = _make_inputs(s, b, n, channels, torch.bfloat16, with_bias)
            actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
            expected = _reference_copy(reference_op(*inputs))
            _assert_close(
                actual,
                expected,
                torch.bfloat16,
                f"batches ({s},{b},{n},{channels},bias={with_bias})",
            )


@_ASCEND_ONLY
@pytest.mark.fused_h_post_bda_forward
def test_forward_ascend_repeated_calls_and_edge_inputs():
    # Repeated calls must return independent, correct outputs.
    for dtype in (torch.bfloat16, torch.float32):
        for with_bias in (False, True):
            for non_contiguous in (False, True):
                torch.manual_seed(2026)
                inputs = _make_inputs(2, 3, 4, 130, dtype, with_bias)
                if non_contiguous:
                    inputs = tuple(
                        _non_contiguous(t) if t is not None else None for t in inputs
                    )
                first = flag_train.megatron.fused_h_post_bda_forward(*inputs)
                second = flag_train.megatron.fused_h_post_bda_forward(*inputs)
                first_sum = first.float().sum()
                expected = _reference_copy(reference_op(*_contiguous(inputs)))
                label = f"{dtype},bias={with_bias},non_contiguous={non_contiguous}"
                assert (
                    first.untyped_storage().data_ptr()
                    != second.untyped_storage().data_ptr()
                )
                utils.train_assert_equal(
                    utils.to_reference(first_sum),
                    utils.to_reference(first.float().sum()),
                )
                _assert_close(first, expected, dtype, f"first call {label}")
                _assert_close(second, expected, dtype, f"second call {label}")

    # Non-contiguous inputs on the vector dispatch paths.
    for s, b, n, channels, dtype in [
        (256, 1, 8, 130, torch.float16),
        (256, 4, 4, 130, torch.bfloat16),
    ]:
        torch.manual_seed(2026)
        inputs = tuple(
            _non_contiguous(t) for t in _make_inputs(s, b, n, channels, dtype, True)
        )
        actual = flag_train.megatron.fused_h_post_bda_forward(*inputs)
        expected = _reference_copy(reference_op(*_contiguous(inputs)))
        _assert_close(
            actual, expected, dtype, f"non-contiguous ({s},{b},{n},{channels})"
        )

    # Contiguous inputs whose storage offset breaks 16B alignment.
    def unaligned(tensor):
        flat = tensor.reshape(-1)
        storage = torch.empty(
            flat.numel() + 1, dtype=tensor.dtype, device=tensor.device
        )
        storage[1:].copy_(flat)
        result = storage[1:].view(tensor.shape)
        assert result.is_contiguous()
        assert result.data_ptr() % 16 != 0
        return result

    torch.manual_seed(2026)
    inputs = tuple(
        unaligned(t) for t in _make_inputs(2, 3, 4, 130, torch.bfloat16, True)
    )
    first = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    second = flag_train.megatron.fused_h_post_bda_forward(*inputs)
    expected = _reference_copy(reference_op(*inputs))
    assert first.data_ptr() != second.data_ptr()
    _assert_close(first, expected, torch.bfloat16, "first unaligned output")
    _assert_close(second, expected, torch.bfloat16, "second unaligned output")

    # Validation must still reject bad shapes after repeated calls.
    inputs = _make_inputs(2, 3, 4, 130, torch.bfloat16, with_bias=False)
    for _ in range(2):
        flag_train.megatron.fused_h_post_bda_forward(*inputs)
    with pytest.raises(ValueError, match="h_res must have shape"):
        flag_train.megatron.fused_h_post_bda_forward(inputs[0][:0], *inputs[1:])
