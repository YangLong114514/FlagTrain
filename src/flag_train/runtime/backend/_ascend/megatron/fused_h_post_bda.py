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
"""Ascend Triton implementation for fused mHC H_post BDA."""

import importlib
from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from torch import Tensor

from flag_train.runtime import torch_device_fn

_FIXED_CONFIG = {"BLOCK_C": 64, "BLOCK_S": 1, "num_warps": 4}
_B_CONFIG = {"BLOCK_C": 64, "BLOCK_S": 1, "num_warps": 4}
_MAX_CORE_DIM = 65535
_DOT_BLOCK_C = 2048
_DOT_B_BLOCK_C = 1024
_DOT_LAUNCH_TILES = 4096
_BIAS_SPLIT = 16
_BIAS_BLOCK_S = 16
_FIXED_LAUNCH_CACHE = {}


@lru_cache(maxsize=1)
def _generic():
    return importlib.import_module("flag_train.megatron.fused_h_post_bda")


def _launch_fixed(kernel, grid, *args, num_warps):
    specialization = []
    for param, arg in zip(kernel.params, args):
        if param.is_constexpr:
            specialization.append(arg)
        elif hasattr(arg, "data_ptr"):
            specialization.append((arg.dtype, arg.data_ptr() % 16))
        elif isinstance(arg, int):
            specialization.append((arg == 1, arg % 16 == 0))
        else:
            specialization.append((type(arg), arg))
    key = (
        kernel.cache_key,
        torch_device_fn.current_device(),
        grid,
        num_warps,
        tuple(specialization),
    )
    runner = _FIXED_LAUNCH_CACHE.get(key)
    if runner is None:
        compiled = kernel.warmup(*args, grid=grid, num_warps=num_warps)
        runner = compiled[(grid[0], grid[1] if len(grid) > 1 else 1, 1)]
        _FIXED_LAUNCH_CACHE[key] = runner
    runner(*args)


def _launch_fixed_tiles(kernel, num_tiles, *args, offset_index, num_warps):
    specialization = []
    for param, arg in zip(kernel.params, args):
        if param.is_constexpr:
            specialization.append(arg)
        elif hasattr(arg, "data_ptr"):
            specialization.append((arg.dtype, arg.data_ptr() % 16))
        elif isinstance(arg, int):
            specialization.append((arg == 1, arg % 16 == 0))
        else:
            specialization.append((type(arg), arg))
    specialization = tuple(specialization)
    device = torch_device_fn.current_device()
    for tile_offset in range(0, num_tiles, _DOT_LAUNCH_TILES):
        grid = (min(num_tiles - tile_offset, _DOT_LAUNCH_TILES),)
        key = (kernel.cache_key, device, grid, num_warps, specialization)
        runner = _FIXED_LAUNCH_CACHE.get(key)
        batch_args = (
            args[:offset_index] + (tile_offset,) + args[offset_index + 1 :]
        )
        if runner is None:
            compiled = kernel.warmup(*batch_args, grid=grid, num_warps=num_warps)
            runner = compiled[(grid[0], 1, 1)]
            _FIXED_LAUNCH_CACHE[key] = runner
        runner(*batch_args)


@triton.jit
def _triton_hpb_fwd_dot_kernel(
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    out_ptr,
    tile_offset,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_BLOCKS: tl.constexpr,
):
    tile = tl.program_id(0) + tile_offset
    pid_s = tile // C_BLOCKS
    pid_c = tile % C_BLOCKS
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask = mask_n[:, None] & mask_c[None, :]
    hr = tl.load(
        hr_ptr + pid_s * N * N + offs_n[None, :] * N + offs_n[:, None],
        mask=mask_n[:, None] & mask_n[None, :],
        other=0.0,
    )
    orig = tl.load(
        orig_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
        mask=mask,
        other=0.0,
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    x_tile = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
        tl.float32
    )
    if HAS_BIAS:
        x_tile += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    out = tl.dot(hr, orig, input_precision="ieee") + hp[:, None] * x_tile[None, :]
    tl.store(
        out_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _triton_hpb_bwd_a_dot_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_res_ptr,
    g_x_ptr,
    tile_offset,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_BLOCKS: tl.constexpr,
):
    tile = tl.program_id(0) + tile_offset
    pid_s = tile // C_BLOCKS
    pid_c = tile % C_BLOCKS
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask = mask_n[:, None] & mask_c[None, :]
    go = tl.load(
        go_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
        mask=mask,
        other=0.0,
    )
    hr = tl.load(
        hr_ptr + pid_s * N * N + offs_n[:, None] * N + offs_n[None, :],
        mask=mask_n[:, None] & mask_n[None, :],
        other=0.0,
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    g_res = tl.dot(hr, go, input_precision="ieee")
    g_x = tl.sum(go * hp[:, None], axis=0)
    tl.store(
        g_res_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
        g_res.to(g_res_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        g_x_ptr + pid_s * C + offs_c,
        g_x.to(g_x_ptr.dtype.element_ty),
        mask=mask_c,
    )


@triton.jit
def _triton_hpb_bwd_b_dot_kernel(
    go_ptr,
    orig_ptr,
    x_ptr,
    bias_ptr,
    g_hr_ptr,
    g_hp_ptr,
    sb,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    for pid_s in range(pid, sb, num_programs):
        g_hr = tl.zeros((NP, NP), dtype=tl.float32)
        g_hp = tl.zeros((NP,), dtype=tl.float32)
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            mask = mask_n[:, None] & mask_c[None, :]
            go = tl.load(
                go_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
                mask=mask,
                other=0.0,
            )
            orig = tl.load(
                orig_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
                mask=mask,
                other=0.0,
            )
            x_tile = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            if HAS_BIAS:
                x_tile += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(
                    tl.float32
                )
            al.compile_hint(orig, "dot_pad_only_k")
            al.compile_hint(go, "dot_pad_only_k")
            g_hr += tl.dot(orig, tl.trans(go), input_precision="ieee")
            g_hp += tl.sum(go * x_tile[None, :], axis=1)
        tl.store(
            g_hr_ptr + pid_s * N * N + offs_n[:, None] * N + offs_n[None, :],
            g_hr.to(g_hr_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_n[None, :],
        )
        tl.store(
            g_hp_ptr + pid_s * N + offs_n,
            g_hp.to(g_hp_ptr.dtype.element_ty),
            mask=mask_n,
        )


@triton.jit
def _triton_hpb_bias_single_kernel(
    g_x_ptr,
    bias_ptr,
    sb,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for s_start in range(0, sb, BLOCK_S):
        rows = s_start + offs_s
        tile = tl.load(
            g_x_ptr + rows[:, None] * C + offs_c[None, :],
            mask=(rows < sb)[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tile, axis=0)
    tl.store(bias_ptr + offs_c, acc.to(bias_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def _triton_hpb_bias_partial_kernel(
    g_x_ptr,
    partial_ptr,
    sb,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_part = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for s_start in range(pid_part * BLOCK_S, sb, SPLIT * BLOCK_S):
        rows = s_start + offs_s
        tile = tl.load(
            g_x_ptr + rows[:, None] * C + offs_c[None, :],
            mask=(rows < sb)[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tile, axis=0)
    tl.store(partial_ptr + pid_part * C + offs_c, acc, mask=mask_c)


@triton.jit
def _triton_hpb_bias_merge_kernel(
    partial_ptr,
    bias_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_part = tl.arange(0, SPLIT)
    mask_c = offs_c < C
    partial = tl.load(
        partial_ptr + offs_part[:, None] * C + offs_c[None, :],
        mask=mask_c[None, :],
        other=0.0,
    )
    bias = tl.sum(partial, axis=0)
    tl.store(bias_ptr + offs_c, bias.to(bias_ptr.dtype.element_ty), mask=mask_c)


def _dot_eligible(channels, n):
    return channels >= 512 and n <= 16


def _dot_grid(sb, channels):
    total = sb * triton.cdiv(channels, _DOT_BLOCK_C)
    return (min(total, _MAX_CORE_DIM),)


def _grid_a_config(sb, channels):
    config = _FIXED_CONFIG.copy()
    config["BLOCK_C"] = min(
        config["BLOCK_C"], max(64, triton.next_power_of_2(channels))
    )
    c_blocks = triton.cdiv(channels, config["BLOCK_C"])
    if c_blocks > _MAX_CORE_DIM:
        raise ValueError("channel grid exceeds the Ascend core dimension limit")
    grid = (triton.cdiv(sb, config["BLOCK_S"]), c_blocks)
    while grid[0] * grid[1] > _MAX_CORE_DIM:
        config["BLOCK_S"] *= 2
        grid = (triton.cdiv(sb, config["BLOCK_S"]), c_blocks)
    return grid, config


def _bias_gradient(g_x, sb, channels, bias):
    if bias is None:
        return None
    g_bias = torch.empty(channels, dtype=bias.dtype, device=bias.device)
    if sb <= 512:
        with torch_device_fn.device(g_x.device):
            _launch_fixed(
                _triton_hpb_bias_single_kernel,
                (triton.cdiv(channels, 512),),
                g_x,
                g_bias,
                sb,
                channels,
                512,
                _BIAS_BLOCK_S,
                num_warps=4,
            )
        return g_bias
    partial = torch.empty(
        _BIAS_SPLIT, channels, dtype=torch.float32, device=g_x.device
    )
    with torch_device_fn.device(g_x.device):
        _launch_fixed(
            _triton_hpb_bias_partial_kernel,
            (triton.cdiv(channels, 512), _BIAS_SPLIT),
            g_x,
            partial,
            sb,
            channels,
            512,
            _BIAS_BLOCK_S,
            _BIAS_SPLIT,
            num_warps=4,
        )
        _launch_fixed(
            _triton_hpb_bias_merge_kernel,
            (triton.cdiv(channels, 512),),
            partial,
            g_bias,
            channels,
            512,
            _BIAS_SPLIT,
            num_warps=4,
        )
    return g_bias


def fused_h_post_bda_forward(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    generic = _generic()
    s, b, n, channels = generic._validate_inputs(
        h_res, original_residual, h_post, x, bias
    )
    sb = s * b
    out = torch.empty(s, b, n, channels, dtype=h_res.dtype, device=h_res.device)
    if n <= 16:
        hr_flat = h_res.contiguous()
        orig_flat = original_residual.contiguous()
        hp_flat = h_post.contiguous()
        x_flat = x.contiguous()
        bias_flat = bias.contiguous() if bias is not None else x_flat
        max_block_c = 1024 if h_res.dtype == torch.float32 else _DOT_BLOCK_C
        dot_block_c = min(max_block_c, max(64, triton.next_power_of_2(channels)))
        c_blocks = triton.cdiv(channels, dot_block_c)
        num_tiles = sb * c_blocks
        with torch_device_fn.device(h_res.device):
            _launch_fixed_tiles(
                _triton_hpb_fwd_dot_kernel,
                num_tiles,
                hr_flat,
                orig_flat,
                hp_flat,
                x_flat,
                bias_flat,
                out,
                0,
                channels,
                n,
                max(4, n),
                bias is not None,
                dot_block_c,
                c_blocks,
                offset_index=6,
                num_warps=4,
            )
        return out

    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, channels)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, channels)
    bias_flat = bias.contiguous() if bias is not None else x_flat
    out_flat = out.view(sb, n, channels)
    grid, config = _grid_a_config(sb, channels)
    with generic._disable_aabs(), torch_device_fn.device(h_res.device):
        generic._triton_hpb_fwd_kernel.fn.fn[grid](
            hr_flat,
            orig_flat,
            hp_flat,
            x_flat,
            bias_flat,
            out_flat,
            sb,
            channels,
            n,
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),
            orig_flat.stride(0),
            orig_flat.stride(1),
            orig_flat.stride(2),
            out_flat.stride(0),
            out_flat.stride(1),
            out_flat.stride(2),
            HAS_BIAS=(bias is not None),
            REUSE_INPUT=False,
            **config,
        )
    return out


def fused_h_post_bda_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    generic = _generic()
    s, b, n, channels = generic._validate_inputs(
        h_res, original_residual, h_post, x, bias
    )
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
    g_hr = torch.empty(s, b, n, n, dtype=h_res.dtype, device=h_res.device)
    g_res = torch.empty(
        s, b, n, channels, dtype=original_residual.dtype, device=h_res.device
    )
    g_hp = torch.empty(s, b, n, dtype=h_post.dtype, device=h_res.device)
    g_x = torch.empty(s, b, channels, dtype=x.dtype, device=x.device)
    go_flat = grad_output.contiguous()
    hr_flat = h_res.contiguous()
    orig_flat = original_residual.contiguous()
    hp_flat = h_post.contiguous()
    x_flat = x.contiguous()
    bias_flat = bias.contiguous() if bias is not None else x_flat
    dot_block_c = 1024 if h_res.dtype == torch.float32 else _DOT_BLOCK_C
    c_blocks = triton.cdiv(channels, dot_block_c)
    num_tiles = sb * c_blocks
    dot_eligible = _dot_eligible(channels, n)
    small_fused = sb <= 8 and channels <= 1024 and n <= 4
    stride_c_s = n * channels
    stride_hr_s = n * n

    if small_fused:
        config = {
            "BLOCK_C": 64,
            "BLOCK_S": 2 if n <= 2 else 1,
            "num_warps": 4,
        }
        with generic._disable_aabs(), torch_device_fn.device(h_res.device):
            _launch_fixed(
                generic._triton_hpb_bwd_small_kernel,
                (triton.cdiv(sb, config["BLOCK_S"]),),
                go_flat,
                hr_flat,
                orig_flat,
                hp_flat,
                x_flat,
                bias_flat,
                g_hr,
                g_res,
                g_hp,
                g_x,
                sb,
                channels,
                n,
                stride_c_s,
                channels,
                1,
                stride_hr_s,
                n,
                1,
                stride_c_s,
                channels,
                1,
                bias is not None,
                config["BLOCK_C"],
                config["BLOCK_S"],
                num_warps=config["num_warps"],
            )
    elif dot_eligible:
        with torch_device_fn.device(h_res.device):
            _launch_fixed_tiles(
                _triton_hpb_bwd_a_dot_kernel,
                num_tiles,
                go_flat,
                hr_flat,
                hp_flat,
                g_res,
                g_x,
                0,
                channels,
                n,
                16,
                dot_block_c,
                c_blocks,
                offset_index=5,
                num_warps=4,
            )
            _launch_fixed(
                _triton_hpb_bwd_b_dot_kernel,
                (min(sb, _MAX_CORE_DIM),),
                go_flat,
                orig_flat,
                x_flat,
                bias_flat,
                g_hr,
                g_hp,
                sb,
                channels,
                n,
                16,
                bias is not None,
                min(
                    _DOT_B_BLOCK_C,
                    max(64, triton.next_power_of_2(channels)),
                ),
                num_warps=8,
            )
    else:
        grid_a, config_a = _grid_a_config(sb, channels)
        with generic._disable_aabs(), torch_device_fn.device(h_res.device):
            generic._triton_hpb_bwd_g_x_orig_kernel.fn.fn[grid_a](
                go_flat,
                hr_flat,
                hp_flat,
                g_res,
                g_x,
                sb,
                channels,
                n,
                stride_c_s,
                channels,
                1,
                stride_hr_s,
                n,
                1,
                stride_c_s,
                channels,
                1,
                REUSE_INPUT=False,
                **config_a,
            )
            generic._triton_hpb_bwd_g_hp_hr_kernel.fn.fn[(sb,)](
                go_flat,
                orig_flat,
                x_flat,
                bias_flat,
                g_hr,
                g_hp,
                sb,
                channels,
                n,
                stride_c_s,
                channels,
                1,
                stride_c_s,
                channels,
                1,
                stride_hr_s,
                n,
                1,
                HAS_BIAS=(bias is not None),
                **_B_CONFIG,
            )

    g_bias = _bias_gradient(g_x, sb, channels, bias)
    return g_hr, g_res, g_hp, g_x, g_bias
