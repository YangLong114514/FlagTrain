# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused mHC H_post BDA forward operator.

Ported from Megatron-LM's ``fused_mhc_kernels.py``. The forward computes
``H_res.T @ original_residual + H_post * (x + bias)``.
"""

import logging
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@contextmanager
def _disable_aabs():
    # The flagtree Triton fork's AABS adjusts autotune configs in place, so one
    # tiny-C autotune run permanently cripples BLOCK_C for every later shape.
    autotuning = getattr(getattr(triton, "knobs", None), "autotuning", None)
    if autotuning is None or not hasattr(autotuning, "adjust_block_size"):
        yield
        return
    prev = autotuning.adjust_block_size
    autotuning.adjust_block_size = False
    try:
        yield
    finally:
        autotuning.adjust_block_size = prev


def _hpb_configs():
    return [
        triton.Config({"BLOCK_C": block_c, "BLOCK_S": block_s}, num_warps=num_warps)
        for block_c in (64, 128, 256, 512)
        for block_s in (1, 2, 4, 8)
        for num_warps in (2, 4, 8)
    ]


def _hpb_autotune(**autotune_kwargs):
    """Autotune while treating configs rejected by the driver as infinitely slow."""

    def decorator(fn):
        tuner = triton.autotune(**autotune_kwargs)(fn)
        bench = tuner._bench

        def _safe_bench(*args, **kwargs):
            try:
                return bench(*args, **kwargs)
            except RuntimeError:
                return [float("inf")] * 3

        tuner._bench = _safe_bench
        return tuner

    return decorator


def _prune_hpb_configs(configs, nargs, **kwargs):
    if not kwargs["REUSE_INPUT"]:
        return configs
    block_channels = (256, 512) if nargs["hr_ptr"].dtype == torch.float32 else (512,)
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_C"] in block_channels
        and config.kwargs["BLOCK_S"] in (1, 2)
        and config.num_warps == 2
    ]


def _validate_inputs(h_res, original_residual, h_post, x, bias):
    if original_residual.dim() != 4:
        raise ValueError("original_residual must have shape [s, b, n, C]")
    s, b, n, channels = original_residual.shape
    expected = (
        (h_res, (s, b, n, n), "h_res"),
        (h_post, (s, b, n), "h_post"),
        (x, (s, b, channels), "x"),
    )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.device != original_residual.device:
            raise ValueError("all inputs must be on the same device")
        if tensor.dtype != original_residual.dtype:
            raise ValueError("all inputs must have the same dtype")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != channels):
        raise ValueError(f"bias must have shape [{channels}]")
    if bias is not None and (
        bias.device != original_residual.device or bias.dtype != original_residual.dtype
    ):
        raise ValueError("bias must have the same device and dtype as the other inputs")
    if original_residual.device.type != runtime_device.name:
        raise ValueError(
            f"fused_h_post_bda only supports {runtime_device.name} tensors"
        )
    if s * b == 0 or channels == 0:
        raise ValueError("empty inputs are not supported")
    if n == 0 or n != triton.next_power_of_2(n):
        raise ValueError("n must be a positive power of two")
    return s, b, n, channels


@libentry()
@_hpb_autotune(
    configs=_hpb_configs(),
    key=["C", "N", "REUSE_INPUT"],
    prune_configs_by={"early_config_prune": _prune_hpb_configs},
)
@triton.jit
def _triton_hpb_fwd_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    sb,
    C: tl.constexpr,
    N: tl.constexpr,
    stride_hr_s,
    stride_hr_i,
    stride_hr_j,
    stride_orig_s,
    stride_orig_n,
    stride_orig_c,
    stride_out_s,
    stride_out_n,
    stride_out_c,
    HAS_BIAS: tl.constexpr,
    REUSE_INPUT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tle.program_id(0)
    pid_c = tle.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    x_tile = tl.load(
        x_ptr + offs_s[:, None] * C + offs_c[None, :], mask=mask_2d, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        x_tile += bias_tile[None, :]

    if REUSE_INPUT == 1:
        offs_n = tl.arange(0, N)
        hp = tl.load(
            hp_ptr + offs_s[None, :] * N + offs_n[:, None],
            mask=mask_s[None, :],
            other=0.0,
        ).to(tl.float32)
        outputs = hp[:, :, None] * x_tile[None, :, :]
        for j in tl.static_range(N):
            orig_j = tl.load(
                orig_ptr
                + offs_s[:, None] * stride_orig_s
                + j * stride_orig_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            hr = tl.load(
                hr_ptr
                + offs_s[None, :] * stride_hr_s
                + j * stride_hr_i
                + offs_n[:, None] * stride_hr_j,
                mask=mask_s[None, :],
                other=0.0,
            ).to(tl.float32)
            outputs += hr[:, :, None] * orig_j[None, :, :]
        tl.store(
            out_ptr
            + offs_s[None, :, None] * stride_out_s
            + offs_n[:, None, None] * stride_out_n
            + offs_c[None, None, :],
            outputs.to(out_ptr.dtype.element_ty),
            mask=mask_2d[None, :, :],
        )
    else:
        for i in tl.static_range(N):
            hp_i = tl.load(hp_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(
                tl.float32
            )
            out_i = hp_i[:, None] * x_tile
            for j in tl.static_range(N):
                hr_ji = tl.load(
                    hr_ptr + offs_s * stride_hr_s + j * stride_hr_i + i * stride_hr_j,
                    mask=mask_s,
                    other=0.0,
                ).to(tl.float32)
                orig_j = tl.load(
                    orig_ptr
                    + offs_s[:, None] * stride_orig_s
                    + j * stride_orig_n
                    + offs_c[None, :],
                    mask=mask_2d,
                    other=0.0,
                ).to(tl.float32)
                out_i += hr_ji[:, None] * orig_j

            tl.store(
                out_ptr
                + offs_s[:, None] * stride_out_s
                + i * stride_out_n
                + offs_c[None, :],
                out_i.to(out_ptr.dtype.element_ty),
                mask=mask_2d,
            )


def fused_h_post_bda_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Compute ``H_res.T @ original_residual + H_post * (x + bias)``."""
    logger.debug("TRAIN FUSED_H_POST_BDA_FORWARD")
    s, b, n, channels = _validate_inputs(h_res, original_residual, h_post, x, bias)
    sb = s * b
    out = torch.empty(sb, n, channels, dtype=h_res.dtype, device=h_res.device)
    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, channels)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, channels)

    grid = lambda meta: (
        triton.cdiv(sb, meta["BLOCK_S"]),
        triton.cdiv(channels, meta["BLOCK_C"]),
    )
    small_nvidia = (
        runtime_device.vendor_name == "nvidia" and sb <= 8 and channels <= 1024
    )
    reuse_input = (
        runtime_device.vendor_name == "nvidia"
        and not small_nvidia
        and (n >= 8 or sb > 512 or h_res.dtype == torch.float32)
    )
    if small_nvidia:
        fwd_kernel = _triton_hpb_fwd_kernel.fn.fn
        fixed_config = {
            "BLOCK_C": 64 if channels <= 256 else 128,
            "BLOCK_S": 2 if n <= 2 else 1,
            "num_warps": 4,
        }
    elif (
        runtime_device.vendor_name == "nvidia"
        and n == 4
        and sb == 512
        and channels == 4096
        and h_res.dtype in (torch.bfloat16, torch.float16)
    ):
        fwd_kernel = _triton_hpb_fwd_kernel.fn.fn
        fixed_config = {
            "BLOCK_C": 256 if h_res.dtype == torch.bfloat16 else 512,
            "BLOCK_S": 2,
            "num_warps": 2 if h_res.dtype == torch.bfloat16 else 4,
        }
    elif runtime_device.vendor_name == "nvidia":
        fwd_kernel = _triton_hpb_fwd_kernel.fn
        fixed_config = {}
    else:
        fwd_kernel = _triton_hpb_fwd_kernel
        fixed_config = {}
    device_context = (
        nullcontext()
        if runtime_device.vendor_name == "nvidia"
        else torch_device_fn.device(h_res.device)
    )
    with _disable_aabs(), device_context:
        fwd_kernel[grid](
            hr_flat,
            orig_flat,
            hp_flat,
            x_flat,
            bias.contiguous() if bias is not None else x_flat,
            out,
            sb,
            channels,
            n,
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),
            orig_flat.stride(0),
            orig_flat.stride(1),
            orig_flat.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            HAS_BIAS=(bias is not None),
            REUSE_INPUT=reuse_input,
            **fixed_config,
        )
    return out.view(s, b, n, channels)
