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
"""Fused mHC H_post BDA forward and backward operators.

Ported from Megatron-LM's ``fused_mhc_kernels.py`` at:
https://github.com/NVIDIA/Megatron-LM/blob/2052769557dcf7045c4091d1996f0bc59afeb80e/megatron/core/fusions/fused_mhc_kernels.py

The forward computes ``H_res.T @ residual + H_post * (x + bias)``. The
backward returns gradients in Megatron's order: ``h_res``, original residual,
``h_post``, ``x``, and optional bias.
"""

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle


def _hpb_configs():
    return [
        triton.Config({"BLOCK_C": block_c, "BLOCK_S": block_s}, num_warps=num_warps)
        for block_c in (64, 128, 256, 512)
        for block_s in (1, 2, 4, 8)
        for num_warps in (2, 4, 8)
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
@triton.autotune(configs=_hpb_configs(), key=["C", "N"])
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

    for i in tl.static_range(N):
        hp_i = tl.load(hp_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(tl.float32)
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


@libentry()
@triton.autotune(configs=_hpb_configs(), key=["C", "N"])
@triton.jit
def _triton_hpb_bwd_g_x_orig_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_orig_ptr,
    g_x_ptr,
    sb,
    C: tl.constexpr,
    N: tl.constexpr,
    stride_go_s,
    stride_go_n,
    stride_go_c,
    stride_hr_s,
    stride_hr_i,
    stride_hr_j,
    stride_orig_s,
    stride_orig_n,
    stride_orig_c,
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

    g_x_acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    for j in tl.static_range(N):
        go_j = tl.load(
            go_ptr + offs_s[:, None] * stride_go_s + j * stride_go_n + offs_c[None, :],
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        hp_j = tl.load(hp_ptr + offs_s * N + j, mask=mask_s, other=0.0).to(tl.float32)
        g_x_acc += hp_j[:, None] * go_j
    tl.store(
        g_x_ptr + offs_s[:, None] * C + offs_c[None, :],
        g_x_acc.to(g_x_ptr.dtype.element_ty),
        mask=mask_2d,
    )

    for i in tl.static_range(N):
        g_orig_i = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(N):
            go_j = tl.load(
                go_ptr
                + offs_s[:, None] * stride_go_s
                + j * stride_go_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            hr_ij = tl.load(
                hr_ptr + offs_s * stride_hr_s + i * stride_hr_i + j * stride_hr_j,
                mask=mask_s,
                other=0.0,
            ).to(tl.float32)
            g_orig_i += hr_ij[:, None] * go_j
        tl.store(
            g_orig_ptr
            + offs_s[:, None] * stride_orig_s
            + i * stride_orig_n
            + offs_c[None, :],
            g_orig_i.to(g_orig_ptr.dtype.element_ty),
            mask=mask_2d,
        )


@libentry()
@triton.autotune(configs=_hpb_configs(), key=["C", "N"])
@triton.jit
def _triton_hpb_bwd_g_hp_hr_kernel(
    go_ptr,
    orig_ptr,
    x_ptr,
    bias_ptr,
    g_hr_ptr,
    g_hp_ptr,
    sb,
    C: tl.constexpr,
    N: tl.constexpr,
    stride_go_s,
    stride_go_n,
    stride_go_c,
    stride_orig_s,
    stride_orig_n,
    stride_orig_c,
    stride_hr_s,
    stride_hr_i,
    stride_hr_j,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tle.program_id(0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < sb

    g_hp_acc = tl.zeros((BLOCK_S, N), dtype=tl.float32)
    g_hr_acc = tl.zeros((BLOCK_S, N * N), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]
        x_tile = tl.load(
            x_ptr + offs_s[:, None] * C + offs_c[None, :], mask=mask_2d, other=0.0
        ).to(tl.float32)
        if HAS_BIAS:
            bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            x_tile += bias_tile[None, :]

        for i in tl.static_range(N):
            go_i = tl.load(
                go_ptr
                + offs_s[:, None] * stride_go_s
                + i * stride_go_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            dot_hp = tl.sum(go_i * x_tile, axis=1)
            g_hp_acc += tl.where(
                tl.arange(0, N)[None, :] == i,
                dot_hp[:, None],
                tl.zeros((BLOCK_S, N), dtype=tl.float32),
            )
            for j in tl.static_range(N):
                orig_j = tl.load(
                    orig_ptr
                    + offs_s[:, None] * stride_orig_s
                    + j * stride_orig_n
                    + offs_c[None, :],
                    mask=mask_2d,
                    other=0.0,
                ).to(tl.float32)
                dot_hr = tl.sum(go_i * orig_j, axis=1)
                g_hr_acc += tl.where(
                    tl.arange(0, N * N)[None, :] == j * N + i,
                    dot_hr[:, None],
                    tl.zeros((BLOCK_S, N * N), dtype=tl.float32),
                )

    offs_n = tl.arange(0, N)
    tl.store(
        g_hp_ptr + offs_s[:, None] * N + offs_n[None, :],
        g_hp_acc.to(g_hp_ptr.dtype.element_ty),
        mask=mask_s[:, None],
    )

    nn_offs = tl.arange(0, N * N)
    for i in tl.static_range(N):
        for j in tl.static_range(N):
            col_mask = (nn_offs == (i * N + j)).to(tl.float32)
            val = tl.sum(g_hr_acc * col_mask[None, :], axis=1)
            tl.store(
                g_hr_ptr + offs_s * stride_hr_s + i * stride_hr_i + j * stride_hr_j,
                val.to(g_hr_ptr.dtype.element_ty),
                mask=mask_s,
            )


def fused_h_post_bda_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Compute ``H_res.T @ original_residual + H_post * (x + bias)``."""
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
    if small_nvidia:
        fwd_kernel = _triton_hpb_fwd_kernel.fn.fn
        fixed_config = {
            "BLOCK_C": 64 if channels <= 256 else 128,
            "BLOCK_S": 2 if n <= 2 else 1,
            "num_warps": 4,
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
    with device_context:
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
            **fixed_config,
        )
    return out.view(s, b, n, channels)


def fused_h_post_bda_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Return gradients for the five inputs of :func:`fused_h_post_bda_forward`."""
    s, b, n, channels = _validate_inputs(h_res, original_residual, h_post, x, bias)
    if tuple(grad_output.shape) != (s, b, n, channels):
        raise ValueError("grad_output must have the same shape as original_residual")
    if (
        grad_output.device != original_residual.device
        or grad_output.dtype != original_residual.dtype
    ):
        raise ValueError(
            "grad_output must have the same device and dtype as the other inputs"
        )

    sb = s * b
    g_hr = torch.empty(sb, n, n, dtype=h_res.dtype, device=h_res.device)
    g_res = torch.empty(
        sb, n, channels, dtype=original_residual.dtype, device=h_res.device
    )
    g_hp = torch.empty(sb, n, dtype=h_post.dtype, device=h_res.device)
    g_x = torch.empty(sb, channels, dtype=x.dtype, device=x.device)

    go_flat = grad_output.contiguous().view(sb, n, channels)
    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, channels)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, channels)

    grid_a = lambda meta: (
        triton.cdiv(sb, meta["BLOCK_S"]),
        triton.cdiv(channels, meta["BLOCK_C"]),
    )
    grid_b = lambda meta: (triton.cdiv(sb, meta["BLOCK_S"]),)
    small_nvidia = (
        runtime_device.vendor_name == "nvidia" and sb <= 8 and channels <= 1024
    )
    if small_nvidia:
        bwd_g_x_orig_kernel = _triton_hpb_bwd_g_x_orig_kernel.fn.fn
        bwd_g_hp_hr_kernel = _triton_hpb_bwd_g_hp_hr_kernel.fn.fn
        fixed_config = {
            "BLOCK_C": 64 if channels <= 256 else 128,
            "BLOCK_S": 2 if n <= 2 else 1,
            "num_warps": 4,
        }
        device_context = nullcontext()
    elif runtime_device.vendor_name == "nvidia":
        bwd_g_x_orig_kernel = _triton_hpb_bwd_g_x_orig_kernel.fn
        bwd_g_hp_hr_kernel = _triton_hpb_bwd_g_hp_hr_kernel.fn
        fixed_config = {}
        device_context = nullcontext()
    else:
        bwd_g_x_orig_kernel = _triton_hpb_bwd_g_x_orig_kernel
        bwd_g_hp_hr_kernel = _triton_hpb_bwd_g_hp_hr_kernel
        fixed_config = {}
        device_context = torch_device_fn.device(h_res.device)
    with device_context:
        bwd_g_x_orig_kernel[grid_a](
            go_flat,
            hr_flat,
            hp_flat,
            g_res,
            g_x,
            sb,
            channels,
            n,
            go_flat.stride(0),
            go_flat.stride(1),
            go_flat.stride(2),
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),
            g_res.stride(0),
            g_res.stride(1),
            g_res.stride(2),
            **fixed_config,
        )
        bwd_g_hp_hr_kernel[grid_b](
            go_flat,
            orig_flat,
            x_flat,
            bias.contiguous() if bias is not None else x_flat,
            g_hr,
            g_hp,
            sb,
            channels,
            n,
            go_flat.stride(0),
            go_flat.stride(1),
            go_flat.stride(2),
            orig_flat.stride(0),
            orig_flat.stride(1),
            orig_flat.stride(2),
            g_hr.stride(0),
            g_hr.stride(1),
            g_hr.stride(2),
            HAS_BIAS=(bias is not None),
            **fixed_config,
        )

    g_bias = g_x.sum(dim=0).to(dtype=bias.dtype) if bias is not None else None
    return (
        g_hr.view(s, b, n, n),
        g_res.view(s, b, n, channels),
        g_hp.view(s, b, n),
        g_x.view(s, b, channels),
        g_bias,
    )
