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
"""Ascend Triton implementation for fused mHC H_post BDA backward."""

import logging
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from torch import Tensor
from triton.runtime.driver import driver

from flag_train.megatron.fused_h_post_bda_backward import (
    _disable_aabs,
    _triton_hpb_bwd_g_hp_hr_kernel,
    _triton_hpb_bwd_g_x_orig_kernel,
    _triton_hpb_bwd_small_kernel,
    _validate_inputs,
)
from flag_train.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _device_context(device):
    if torch_device_fn.current_device() == device.index:
        return nullcontext()
    return torch_device_fn.device(device)


_FIXED_CONFIG = {"BLOCK_C": 64, "BLOCK_S": 1, "num_warps": 4}
_B_CONFIG = {"BLOCK_C": 64, "BLOCK_S": 1, "num_warps": 4}
_MAX_CORE_DIM = 65535
_DOT_BLOCK_C = 2048
_DOT_B_BLOCK_C = 1024
_DOT_LAUNCH_TILES = 4096
_BIAS_SPLIT = 16
_BIAS_BLOCK_S = 16
_FIXED_LAUNCH_CACHE = {}
_BACKWARD_PLAN_CACHE = {}
_KERNEL_SIGNATURES = {}
_RAW_STREAM_NO_WAIT = getattr(
    __import__("torch_npu")._C, "_npu_getCurrentRawStreamNoWait", None
)


def _kernel_signature(kernel):
    signature = _KERNEL_SIGNATURES.get(kernel)
    if signature is None:
        signature = (
            kernel.cache_key,
            tuple(param.is_constexpr for param in kernel.params),
        )
        _KERNEL_SIGNATURES[kernel] = signature
    return signature


def _current_stream(device):
    if _RAW_STREAM_NO_WAIT is not None:
        return _RAW_STREAM_NO_WAIT(device)
    return driver.active.get_current_stream(device)


def _backward_plan_key(grad_output, h_res, original_residual, h_post, x, bias):
    tensors = (grad_output, h_res, original_residual, h_post, x)
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        return None
    if original_residual.dim() != 4:
        return None
    s, b, n, channels = original_residual.shape
    sb = s * b
    if sb <= 0 or sb > 8 or channels <= 0 or channels > 1024:
        return None
    if n <= 0 or n > 4 or n & (n - 1):
        return None
    dtype = original_residual.dtype
    device = original_residual.device
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    if device.type != "npu":
        return None
    if (
        grad_output.shape != (s, b, n, channels)
        or h_res.shape != (s, b, n, n)
        or h_post.shape != (s, b, n)
        or x.shape != (s, b, channels)
    ):
        return None
    if (
        grad_output.dtype != dtype
        or h_res.dtype != dtype
        or h_post.dtype != dtype
        or x.dtype != dtype
    ):
        return None
    if (
        grad_output.device != device
        or h_res.device != device
        or h_post.device != device
        or x.device != device
    ):
        return None
    if not (
        grad_output.is_contiguous()
        and h_res.is_contiguous()
        and original_residual.is_contiguous()
        and h_post.is_contiguous()
        and x.is_contiguous()
    ):
        return None
    if bias is not None:
        if not isinstance(bias, Tensor) or bias.dim() != 1:
            return None
        if bias.shape[0] != channels or bias.dtype != dtype or bias.device != device:
            return None
        if not bias.is_contiguous():
            return None
    if triton.knobs.runtime.launch_enter_hook.calls or (
        triton.knobs.runtime.launch_exit_hook.calls
    ):
        return None
    alignment = (
        grad_output.data_ptr() % 16,
        h_res.data_ptr() % 16,
        original_residual.data_ptr() % 16,
        h_post.data_ptr() % 16,
        x.data_ptr() % 16,
    )
    if bias is not None:
        alignment += (bias.data_ptr() % 16,)
    stream = _current_stream(device.index)
    return (
        tuple(original_residual.shape),
        dtype,
        device,
        bias is not None,
        alignment,
        stream,
    )


def _launch_backward_plan(
    plan, stream, grad_output, h_res, original_residual, h_post, x, bias
):
    (
        small_compiled,
        small_grid,
        small_constants,
        small_bias_fused,
        bias_compiled,
        bias_grid,
        bias_constants,
        expected_alignment,
    ) = plan
    g_hr = torch.empty(h_res.shape, dtype=h_res.dtype, device=h_res.device)
    g_res = torch.empty(
        original_residual.shape,
        dtype=original_residual.dtype,
        device=original_residual.device,
    )
    g_hp = torch.empty(h_post.shape, dtype=h_post.dtype, device=h_post.device)
    g_x = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    g_bias = (
        torch.empty(bias.shape, dtype=bias.dtype, device=bias.device)
        if bias is not None
        else None
    )
    alignment = (
        g_hr.data_ptr() % 16,
        g_res.data_ptr() % 16,
        g_hp.data_ptr() % 16,
        g_x.data_ptr() % 16,
        g_bias.data_ptr() % 16 if g_bias is not None else None,
    )
    if alignment != expected_alignment:
        return None
    bias_arg = bias if bias is not None else x
    with _device_context(original_residual.device):
        if small_bias_fused:
            args = (
                grad_output,
                h_res,
                original_residual,
                h_post,
                x,
                bias_arg,
                g_hr,
                g_res,
                g_hp,
                g_x,
                g_bias,
            ) + small_constants
        else:
            args = (
                grad_output,
                h_res,
                original_residual,
                h_post,
                x,
                bias_arg,
                g_hr,
                g_res,
                g_hp,
                g_x,
            ) + small_constants
        small_compiled.run(
            small_grid,
            1,
            1,
            stream,
            small_compiled.function,
            small_compiled.packed_metadata,
            None,
            None,
            None,
            *args,
        )
        if bias_compiled is not None:
            bias_args = (g_x, g_bias) + bias_constants
            bias_compiled.run(
                bias_grid,
                1,
                1,
                stream,
                bias_compiled.function,
                bias_compiled.packed_metadata,
                None,
                None,
                None,
                *bias_args,
            )
    return g_hr, g_res, g_hp, g_x, g_bias


def _register_backward_plan(
    key,
    grad_output,
    h_res,
    original_residual,
    h_post,
    x,
    bias,
    g_hr,
    g_res,
    g_hp,
    g_x,
    g_bias,
    sb,
    channels,
    n,
    small_bias_fused,
    block_c,
    block_s,
):
    if key in _BACKWARD_PLAN_CACHE:
        return
    stride_c_s = n * channels
    stride_hr_s = n * n
    bias_arg = bias if bias is not None else x
    if small_bias_fused:
        small_grid = 1
        small_constants = (
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
            block_c,
            block_s,
            sb > 1,
        )
        try:
            small_compiled = _triton_hpb_bwd_small_bias_kernel.warmup(
                grad_output,
                h_res,
                original_residual,
                h_post,
                x,
                bias_arg,
                g_hr,
                g_res,
                g_hp,
                g_x,
                g_bias,
                *small_constants,
                grid=(small_grid, 1, 1),
                num_warps=4,
            )
        except Exception:
            return
    else:
        small_grid = triton.cdiv(sb, block_s)
        small_constants = (
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
            block_c,
            block_s,
        )
        try:
            small_compiled = _triton_hpb_bwd_small_kernel.warmup(
                grad_output,
                h_res,
                original_residual,
                h_post,
                x,
                bias_arg,
                g_hr,
                g_res,
                g_hp,
                g_x,
                *small_constants,
                grid=(small_grid, 1, 1),
                num_warps=4,
            )
        except Exception:
            return
    bias_compiled = None
    bias_grid = 0
    bias_constants = ()
    if bias is not None and not small_bias_fused:
        bias_grid = triton.cdiv(channels, 512)
        bias_constants = (sb, channels, 512, _BIAS_BLOCK_S)
        try:
            bias_compiled = _triton_hpb_bias_single_kernel.warmup(
                g_x,
                g_bias,
                *bias_constants,
                grid=(bias_grid, 1, 1),
                num_warps=4,
            )
        except Exception:
            return
    alignment = (
        g_hr.data_ptr() % 16,
        g_res.data_ptr() % 16,
        g_hp.data_ptr() % 16,
        g_x.data_ptr() % 16,
        g_bias.data_ptr() % 16 if g_bias is not None else None,
    )
    _BACKWARD_PLAN_CACHE[key] = (
        small_compiled,
        small_grid,
        small_constants,
        small_bias_fused,
        bias_compiled,
        bias_grid,
        bias_constants,
        alignment,
    )


def _specialization(kernel, args):
    cache_key, constexpr = _kernel_signature(kernel)
    specialization = []
    for is_constexpr, arg in zip(constexpr, args):
        if is_constexpr:
            specialization.append(arg)
        elif hasattr(arg, "data_ptr"):
            specialization.append((arg.dtype, arg.data_ptr() % 16))
        elif isinstance(arg, int):
            specialization.append((arg == 1, arg % 16 == 0))
        else:
            specialization.append((type(arg), arg))
    return cache_key, tuple(specialization)


def _run_fixed_entry(entry, grid, stream, fast_launch, args):
    compiled, runner = entry
    if fast_launch:
        compiled.run(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            1,
            stream,
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *args,
        )
    else:
        runner(*args, stream=stream)


def _launch_fixed(kernel, grid, *args, num_warps):
    cache_key, specialization = _specialization(kernel, args)
    device = torch_device_fn.current_device()
    stream = _current_stream(device)
    fast_launch = not triton.knobs.runtime.launch_enter_hook.calls and not (
        triton.knobs.runtime.launch_exit_hook.calls
    )
    key = (cache_key, device, grid, num_warps, specialization)
    entry = _FIXED_LAUNCH_CACHE.get(key)
    if entry is None:
        compiled = kernel.warmup(*args, grid=grid, num_warps=num_warps)
        entry = (compiled, compiled[(grid[0], grid[1] if len(grid) > 1 else 1, 1)])
        _FIXED_LAUNCH_CACHE[key] = entry
    _run_fixed_entry(entry, grid, stream, fast_launch, args)


def _launch_fixed_tiles(kernel, num_tiles, *args, offset_index, num_warps):
    cache_key, specialization = _specialization(kernel, args)
    device = torch_device_fn.current_device()
    stream = _current_stream(device)
    fast_launch = not triton.knobs.runtime.launch_enter_hook.calls and not (
        triton.knobs.runtime.launch_exit_hook.calls
    )
    for tile_offset in range(0, num_tiles, _DOT_LAUNCH_TILES):
        grid = (min(num_tiles - tile_offset, _DOT_LAUNCH_TILES),)
        key = (cache_key, device, grid, num_warps, specialization)
        entry = _FIXED_LAUNCH_CACHE.get(key)
        batch_args = args[:offset_index] + (tile_offset,) + args[offset_index + 1 :]
        if entry is None:
            compiled = kernel.warmup(*batch_args, grid=grid, num_warps=num_warps)
            entry = (compiled, compiled[(grid[0], 1, 1)])
            _FIXED_LAUNCH_CACHE[key] = entry
        _run_fixed_entry(entry, grid, stream, fast_launch, batch_args)


@triton.jit
def _triton_hpb_bwd_a_dot_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_res_ptr,
    g_x_ptr,
    g_bias_ptr,
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
    g_x_out = g_x.to(g_x_ptr.dtype.element_ty)
    tl.store(g_x_ptr + pid_s * C + offs_c, g_x_out, mask=mask_c)
    if HAS_BIAS:
        tl.store(
            g_bias_ptr + pid_s * C + offs_c,
            g_x_out.to(g_bias_ptr.dtype.element_ty),
            mask=mask_c,
        )


@triton.jit
def _triton_hpb_bwd_a_grouped_dot_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_res_ptr,
    g_x_ptr,
    g_bias_ptr,
    tile_offset,
    C: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_BLOCKS: tl.constexpr,
    C_GROUP: tl.constexpr,
    C_GROUPS: tl.constexpr,
):
    tile = tl.program_id(0) + tile_offset
    pid_s = tile // C_GROUPS
    pid_group = tile % C_GROUPS
    offs_n = tl.arange(0, NP)
    mask_n = offs_n < N
    hr = tl.load(
        hr_ptr + pid_s * N * N + offs_n[:, None] * N + offs_n[None, :],
        mask=mask_n[:, None] & mask_n[None, :],
        other=0.0,
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    for c_offset in range(0, C_GROUP):
        pid_c = pid_group * C_GROUP + c_offset
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask = mask_n[:, None] & mask_c[None, :]
        go = tl.load(
            go_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            mask=mask,
            other=0.0,
        )
        g_res = tl.dot(hr, go, input_precision="ieee")
        g_x = tl.sum(go * hp[:, None], axis=0)
        tl.store(
            g_res_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            g_res.to(g_res_ptr.dtype.element_ty),
            mask=mask,
        )
        g_x_out = g_x.to(g_x_ptr.dtype.element_ty)
        tl.store(g_x_ptr + pid_s * C + offs_c, g_x_out, mask=mask_c)
        if HAS_BIAS:
            tl.store(
                g_bias_ptr + pid_s * C + offs_c,
                g_x_out.to(g_bias_ptr.dtype.element_ty),
                mask=mask_c,
            )


@triton.jit
def _triton_hpb_bwd_a_vector4_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_res_ptr,
    g_x_ptr,
    g_bias_ptr,
    tile_offset,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0) + tile_offset
    hr_base = hr_ptr + pid_s * 16
    h00 = tl.load(hr_base + 0)
    h01 = tl.load(hr_base + 1)
    h02 = tl.load(hr_base + 2)
    h03 = tl.load(hr_base + 3)
    h10 = tl.load(hr_base + 4)
    h11 = tl.load(hr_base + 5)
    h12 = tl.load(hr_base + 6)
    h13 = tl.load(hr_base + 7)
    h20 = tl.load(hr_base + 8)
    h21 = tl.load(hr_base + 9)
    h22 = tl.load(hr_base + 10)
    h23 = tl.load(hr_base + 11)
    h30 = tl.load(hr_base + 12)
    h31 = tl.load(hr_base + 13)
    h32 = tl.load(hr_base + 14)
    h33 = tl.load(hr_base + 15)
    hp_base = hp_ptr + pid_s * 4
    hp0 = tl.load(hp_base + 0).to(tl.float32)
    hp1 = tl.load(hp_base + 1).to(tl.float32)
    hp2 = tl.load(hp_base + 2).to(tl.float32)
    hp3 = tl.load(hp_base + 3).to(tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        go_base = go_ptr + pid_s * 4 * C
        go0 = tl.load(go_base + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        go1 = tl.load(go_base + C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        go2 = tl.load(go_base + 2 * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        go3 = tl.load(go_base + 3 * C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        g_res0 = h00 * go0 + h01 * go1 + h02 * go2 + h03 * go3
        g_res1 = h10 * go0 + h11 * go1 + h12 * go2 + h13 * go3
        g_res2 = h20 * go0 + h21 * go1 + h22 * go2 + h23 * go3
        g_res3 = h30 * go0 + h31 * go1 + h32 * go2 + h33 * go3
        g_x = hp0 * go0 + hp1 * go1 + hp2 * go2 + hp3 * go3
        g_res_base = g_res_ptr + pid_s * 4 * C + offs_c
        tl.store(
            g_res_base,
            g_res0.to(g_res_ptr.dtype.element_ty),
            mask=mask_c,
        )
        tl.store(
            g_res_base + C,
            g_res1.to(g_res_ptr.dtype.element_ty),
            mask=mask_c,
        )
        tl.store(
            g_res_base + 2 * C,
            g_res2.to(g_res_ptr.dtype.element_ty),
            mask=mask_c,
        )
        tl.store(
            g_res_base + 3 * C,
            g_res3.to(g_res_ptr.dtype.element_ty),
            mask=mask_c,
        )
        g_x_out = g_x.to(g_x_ptr.dtype.element_ty)
        tl.store(g_x_ptr + pid_s * C + offs_c, g_x_out, mask=mask_c)
        if HAS_BIAS:
            tl.store(g_bias_ptr + offs_c, g_x_out, mask=mask_c)


@triton.jit
def _triton_hpb_bwd_a_vector4_rows_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_res_ptr,
    g_x_ptr,
    g_bias_ptr,
    tile_offset,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0) + tile_offset
    for row in range(ROWS):
        pid_s = pid * ROWS + row
        hr_base = hr_ptr + pid_s * 16
        h00 = tl.load(hr_base + 0)
        h01 = tl.load(hr_base + 1)
        h02 = tl.load(hr_base + 2)
        h03 = tl.load(hr_base + 3)
        h10 = tl.load(hr_base + 4)
        h11 = tl.load(hr_base + 5)
        h12 = tl.load(hr_base + 6)
        h13 = tl.load(hr_base + 7)
        h20 = tl.load(hr_base + 8)
        h21 = tl.load(hr_base + 9)
        h22 = tl.load(hr_base + 10)
        h23 = tl.load(hr_base + 11)
        h30 = tl.load(hr_base + 12)
        h31 = tl.load(hr_base + 13)
        h32 = tl.load(hr_base + 14)
        h33 = tl.load(hr_base + 15)
        hp_base = hp_ptr + pid_s * 4
        hp0 = tl.load(hp_base + 0).to(tl.float32)
        hp1 = tl.load(hp_base + 1).to(tl.float32)
        hp2 = tl.load(hp_base + 2).to(tl.float32)
        hp3 = tl.load(hp_base + 3).to(tl.float32)
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            go_base = go_ptr + pid_s * 4 * C
            go0 = tl.load(go_base + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            go1 = tl.load(go_base + C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            go2 = tl.load(go_base + 2 * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            go3 = tl.load(go_base + 3 * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            g_res0 = h00 * go0 + h01 * go1 + h02 * go2 + h03 * go3
            g_res1 = h10 * go0 + h11 * go1 + h12 * go2 + h13 * go3
            g_res2 = h20 * go0 + h21 * go1 + h22 * go2 + h23 * go3
            g_res3 = h30 * go0 + h31 * go1 + h32 * go2 + h33 * go3
            g_x = hp0 * go0 + hp1 * go1 + hp2 * go2 + hp3 * go3
            g_res_base = g_res_ptr + pid_s * 4 * C + offs_c
            tl.store(
                g_res_base,
                g_res0.to(g_res_ptr.dtype.element_ty),
                mask=mask_c,
            )
            tl.store(
                g_res_base + C,
                g_res1.to(g_res_ptr.dtype.element_ty),
                mask=mask_c,
            )
            tl.store(
                g_res_base + 2 * C,
                g_res2.to(g_res_ptr.dtype.element_ty),
                mask=mask_c,
            )
            tl.store(
                g_res_base + 3 * C,
                g_res3.to(g_res_ptr.dtype.element_ty),
                mask=mask_c,
            )
            g_x_out = g_x.to(g_x_ptr.dtype.element_ty)
            tl.store(g_x_ptr + pid_s * C + offs_c, g_x_out, mask=mask_c)
            if HAS_BIAS:
                tl.store(g_bias_ptr + offs_c, g_x_out, mask=mask_c)


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
def _triton_hpb_bwd_b_vector4_kernel(
    go_ptr,
    orig_ptr,
    x_ptr,
    bias_ptr,
    g_hr_ptr,
    g_hp_ptr,
    sb,
    C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_n = tl.arange(0, 4)
    for pid_s in range(pid, sb, num_programs):
        go_base = go_ptr + pid_s * 4 * C
        orig_base = orig_ptr + pid_s * 4 * C
        g_hr = tl.zeros((4, 4), dtype=tl.float32)
        g_hp = tl.zeros((4,), dtype=tl.float32)
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            z = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            if HAS_BIAS:
                z += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            go = tl.load(
                go_base + offs_n[:, None] * C + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            orig = tl.load(
                orig_base + offs_n[:, None] * C + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            g_hr += tl.sum(orig[:, None, :] * go[None, :, :], axis=2)
            g_hp += tl.sum(go * z[None, :], axis=1)
        tl.store(
            g_hr_ptr + pid_s * 16 + offs_n[:, None] * 4 + offs_n[None, :],
            g_hr.to(g_hr_ptr.dtype.element_ty),
        )
        tl.store(
            g_hp_ptr + pid_s * 4 + offs_n,
            g_hp.to(g_hp_ptr.dtype.element_ty),
        )


@triton.jit
def _triton_hpb_bwd_a_vecn_kernel(
    go_ptr,
    hr_ptr,
    hp_ptr,
    g_res_ptr,
    g_x_ptr,
    g_bias_ptr,
    tile_offset,
    C: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0) + tile_offset
    offs_n = tl.arange(0, N)
    hr = tl.load(hr_ptr + pid_s * N * N + offs_n[:, None] * N + offs_n[None, :]).to(
        tl.float32
    )
    hp = tl.load(hp_ptr + pid_s * N + offs_n).to(tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        go = tl.load(
            go_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            mask=mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        g_res = tl.sum(hr[:, :, None] * go[None, :, :], axis=1)
        g_x = tl.sum(go * hp[:, None], axis=0)
        tl.store(
            g_res_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
            g_res.to(g_res_ptr.dtype.element_ty),
            mask=mask_c[None, :],
        )
        g_x_out = g_x.to(g_x_ptr.dtype.element_ty)
        tl.store(g_x_ptr + pid_s * C + offs_c, g_x_out, mask=mask_c)
        if HAS_BIAS:
            tl.store(
                g_bias_ptr + pid_s * C + offs_c,
                g_x_out.to(g_bias_ptr.dtype.element_ty),
                mask=mask_c,
            )


@triton.jit
def _triton_hpb_bwd_b_vecn_kernel(
    go_ptr,
    orig_ptr,
    x_ptr,
    bias_ptr,
    g_hr_ptr,
    g_hp_ptr,
    sb,
    C: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_n = tl.arange(0, N)
    for pid_s in range(pid, sb, num_programs):
        g_hr = tl.zeros((N, N), dtype=tl.float32)
        g_hp = tl.zeros((N,), dtype=tl.float32)
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            go = tl.load(
                go_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            orig = tl.load(
                orig_ptr + pid_s * N * C + offs_n[:, None] * C + offs_c[None, :],
                mask=mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            g_hr += tl.sum(orig[:, None, :] * go[None, :, :], axis=2)
            x_tile = tl.load(x_ptr + pid_s * C + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            if HAS_BIAS:
                x_tile += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(
                    tl.float32
                )
            g_hp += tl.sum(go * x_tile[None, :], axis=1)
        tl.store(
            g_hr_ptr + pid_s * N * N + offs_n[:, None] * N + offs_n[None, :],
            g_hr.to(g_hr_ptr.dtype.element_ty),
        )
        tl.store(
            g_hp_ptr + pid_s * N + offs_n,
            g_hp.to(g_hp_ptr.dtype.element_ty),
        )


@triton.jit
def _triton_hpb_bwd_small_bias_kernel(
    go_ptr,
    hr_ptr,
    orig_ptr,
    hp_ptr,
    x_ptr,
    bias_ptr,
    g_hr_ptr,
    g_res_ptr,
    g_hp_ptr,
    g_x_ptr,
    g_bias_ptr,
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
    SUM_BIAS: tl.constexpr,
):
    pid_s = tl.program_id(0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_n = tl.arange(0, N)
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
        bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        x_tile += bias_tile[None, :]
        g_x_acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        g_res_acc = tl.zeros((N, BLOCK_S, BLOCK_C), dtype=tl.float32)
        for i in tl.static_range(N):
            go_i = tl.load(
                go_ptr
                + offs_s[:, None] * stride_go_s
                + i * stride_go_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            hp_i = tl.load(hp_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(
                tl.float32
            )
            hr_i = tl.load(
                hr_ptr
                + offs_s[None, :] * stride_hr_s
                + offs_n[:, None] * stride_hr_i
                + i * stride_hr_j,
                mask=mask_s[None, :],
                other=0.0,
            ).to(tl.float32)
            g_x_acc += hp_i[:, None] * go_i
            g_res_acc += hr_i[:, :, None] * go_i[None, :, :]
            dot_hp = tl.sum(go_i * x_tile, axis=1)
            g_hp_acc += tl.where(
                offs_n[None, :] == i,
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
        tl.store(
            g_res_ptr
            + offs_s[None, :, None] * stride_orig_s
            + offs_n[:, None, None] * stride_orig_n
            + offs_c[None, None, :],
            g_res_acc.to(g_res_ptr.dtype.element_ty),
            mask=mask_2d[None, :, :],
        )
        tl.store(
            g_x_ptr + offs_s[:, None] * C + offs_c[None, :],
            g_x_acc.to(g_x_ptr.dtype.element_ty),
            mask=mask_2d,
        )
        if SUM_BIAS:
            g_bias = tl.sum(g_x_acc.to(g_x_ptr.dtype.element_ty).to(tl.float32), axis=0)
            tl.store(
                g_bias_ptr + offs_c,
                g_bias.to(g_bias_ptr.dtype.element_ty),
                mask=mask_c,
            )
        else:
            tl.store(
                g_bias_ptr + offs_s[:, None] * 0 + offs_c[None, :],
                g_x_acc.to(g_bias_ptr.dtype.element_ty),
                mask=(offs_s == 0)[:, None] & mask_c[None, :],
            )
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
    return n <= 16 and (channels >= 256 or (n in (1, 2, 4) and channels >= 64))


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
        with _device_context(g_x.device):
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
    partial = torch.empty(_BIAS_SPLIT, channels, dtype=torch.float32, device=g_x.device)
    with _device_context(g_x.device):
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


def fused_h_post_bda_backward(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Return gradients for the five inputs of the H_post BDA forward operator."""
    logger.debug("TRAIN_ASCEND FUSED_H_POST_BDA_BACKWARD")
    plan_key = _backward_plan_key(
        grad_output, h_res, original_residual, h_post, x, bias
    )
    if plan_key is not None:
        plan = _BACKWARD_PLAN_CACHE.get(plan_key)
        if plan is not None:
            result = _launch_backward_plan(
                plan,
                plan_key[-1],
                grad_output,
                h_res,
                original_residual,
                h_post,
                x,
                bias,
            )
            if result is not None:
                return result
    s, b, n, channels = _validate_inputs(h_res, original_residual, h_post, x, bias)
    if tuple(grad_output.shape) != (s, b, n, channels):
        raise ValueError("grad_output must have shape [s, b, n, C]")
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

    small_bias_fused = (
        small_fused and (sb == 1 or (sb == 2 and n <= 2)) and bias is not None
    )
    dot_bias1_fused = dot_eligible and sb == 1 and bias is not None
    bias_fused = small_bias_fused or dot_bias1_fused
    g_bias = (
        torch.empty(channels, dtype=bias.dtype, device=bias.device)
        if bias_fused
        else None
    )
    if small_fused:
        config = {
            "BLOCK_C": 64,
            "BLOCK_S": (2 if n <= 2 else (sb if small_bias_fused and n == 4 else 1)),
            "num_warps": 4,
        }
        with _disable_aabs(), _device_context(h_res.device):
            if small_bias_fused:
                _launch_fixed(
                    _triton_hpb_bwd_small_bias_kernel,
                    (1,),
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
                    g_bias,
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
                    config["BLOCK_C"],
                    config["BLOCK_S"],
                    sb > 1,
                    num_warps=config["num_warps"],
                )
            else:
                _launch_fixed(
                    _triton_hpb_bwd_small_kernel,
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
        with _device_context(h_res.device):
            if n == 4:
                vector_block_c = (
                    1024 if sb > 8192 and h_res.dtype != torch.float32 else 2048
                )
                if c_blocks == 1:
                    vector_block_c = min(
                        vector_block_c, max(64, triton.next_power_of_2(channels))
                    )
                if sb > 4096 and sb % 8 == 0:
                    _launch_fixed_tiles(
                        _triton_hpb_bwd_a_vector4_rows_kernel,
                        sb // 8,
                        go_flat,
                        hr_flat,
                        hp_flat,
                        g_res,
                        g_x,
                        g_x,
                        0,
                        channels,
                        False,
                        vector_block_c,
                        8,
                        offset_index=6,
                        num_warps=4,
                    )
                else:
                    _launch_fixed_tiles(
                        _triton_hpb_bwd_a_vector4_kernel,
                        sb,
                        go_flat,
                        hr_flat,
                        hp_flat,
                        g_res,
                        g_x,
                        g_bias if dot_bias1_fused else g_x,
                        0,
                        channels,
                        dot_bias1_fused,
                        vector_block_c,
                        offset_index=6,
                        num_warps=4,
                    )
            elif n <= 2 and channels <= 2048:
                _launch_fixed_tiles(
                    _triton_hpb_bwd_a_vecn_kernel,
                    sb,
                    go_flat,
                    hr_flat,
                    hp_flat,
                    g_res,
                    g_x,
                    g_bias if dot_bias1_fused else g_x,
                    0,
                    channels,
                    n,
                    dot_bias1_fused,
                    min(2048, max(64, triton.next_power_of_2(channels))),
                    offset_index=6,
                    num_warps=4,
                )
            elif c_blocks > 1:
                if sb > 8192:
                    default_c_group = 8 if h_res.dtype == torch.float32 else 4
                else:
                    default_c_group = 2
                c_groups = triton.cdiv(c_blocks, default_c_group)
                c_group = triton.cdiv(c_blocks, c_groups)
                _launch_fixed_tiles(
                    _triton_hpb_bwd_a_grouped_dot_kernel,
                    sb * c_groups,
                    go_flat,
                    hr_flat,
                    hp_flat,
                    g_res,
                    g_x,
                    g_bias if dot_bias1_fused else g_x,
                    0,
                    channels,
                    n,
                    16,
                    dot_bias1_fused,
                    dot_block_c,
                    c_blocks,
                    c_group,
                    c_groups,
                    offset_index=6,
                    num_warps=4,
                )
            else:
                _launch_fixed_tiles(
                    _triton_hpb_bwd_a_dot_kernel,
                    num_tiles,
                    go_flat,
                    hr_flat,
                    hp_flat,
                    g_res,
                    g_x,
                    g_bias if dot_bias1_fused else g_x,
                    0,
                    channels,
                    n,
                    16,
                    dot_bias1_fused,
                    dot_block_c,
                    c_blocks,
                    offset_index=6,
                    num_warps=4,
                )
            if n == 4:
                _launch_fixed(
                    _triton_hpb_bwd_b_vector4_kernel,
                    (min(sb, _MAX_CORE_DIM),),
                    go_flat,
                    orig_flat,
                    x_flat,
                    bias_flat,
                    g_hr,
                    g_hp,
                    sb,
                    channels,
                    bias is not None,
                    min(
                        1024 if channels >= 4096 else 512,
                        max(64, triton.next_power_of_2(channels)),
                    ),
                    num_warps=4,
                )
            elif n <= 2 and channels <= 2048:
                _launch_fixed(
                    _triton_hpb_bwd_b_vecn_kernel,
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
                    bias is not None,
                    min(2048, max(64, triton.next_power_of_2(channels))),
                    num_warps=4,
                )
            else:
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
                    max(4, n),
                    bias is not None,
                    min(_DOT_B_BLOCK_C, max(64, 1 << (channels.bit_length() - 1))),
                    num_warps=8,
                )
    else:
        grid_a, config_a = _grid_a_config(sb, channels)
        with _disable_aabs(), _device_context(h_res.device):
            _triton_hpb_bwd_g_x_orig_kernel.fn.fn[grid_a](
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
            if sb > _DOT_LAUNCH_TILES:
                go_rows = go_flat.view(sb, n, channels)
                orig_rows = orig_flat.view(sb, n, channels)
                x_rows = x_flat.view(sb, channels)
                g_hr_rows = g_hr.view(sb, n, n)
                g_hp_rows = g_hp.view(sb, n)
            else:
                go_rows = go_flat
                orig_rows = orig_flat
                x_rows = x_flat
                g_hr_rows = g_hr
                g_hp_rows = g_hp
            for s_start in range(0, sb, _DOT_LAUNCH_TILES):
                batch_sb = min(sb - s_start, _DOT_LAUNCH_TILES)
                if s_start:
                    go_batch = go_rows[s_start:]
                    orig_batch = orig_rows[s_start:]
                    x_batch = x_rows[s_start:]
                    g_hr_batch = g_hr_rows[s_start:]
                    g_hp_batch = g_hp_rows[s_start:]
                else:
                    go_batch = go_rows
                    orig_batch = orig_rows
                    x_batch = x_rows
                    g_hr_batch = g_hr_rows
                    g_hp_batch = g_hp_rows
                _triton_hpb_bwd_g_hp_hr_kernel.fn.fn[(batch_sb,)](
                    go_batch,
                    orig_batch,
                    x_batch,
                    bias_flat,
                    g_hr_batch,
                    g_hp_batch,
                    batch_sb,
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

    if not bias_fused:
        g_bias = _bias_gradient(g_x, sb, channels, bias)
    if plan_key is not None and small_fused:
        with _device_context(h_res.device):
            _register_backward_plan(
                plan_key,
                go_flat,
                hr_flat,
                orig_flat,
                hp_flat,
                x_flat,
                bias,
                g_hr,
                g_res,
                g_hp,
                g_x,
                g_bias,
                sb,
                channels,
                n,
                small_bias_fused,
                config["BLOCK_C"],
                config["BLOCK_S"],
            )
    return g_hr, g_res, g_hp, g_x, g_bias
